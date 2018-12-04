using Images
using PyPlot
using Printf
using Random
using Statistics
using LinearAlgebra
using Interpolations

include("Common.jl")


#---------------------------------------------------------
# Loads keypoints from JLD2 container.
#
# INPUTS:
#   filename     JLD2 container filename
#
# OUTPUTS:
#   keypoints1   [n x 2] keypoint locations (of left image)
#   keypoints2   [n x 2] keypoint locations (of right image)
#
#---------------------------------------------------------
function loadkeypoints(filename::String)
    container = load(filename)
    keypoints1 = container["keypoints1"]
    keypoints2 = container["keypoints2"]

  @assert size(keypoints1,2) == 2
  @assert size(keypoints2,2) == 2
  return keypoints1::Array{Int64,2}, keypoints2::Array{Int64,2}
end


#---------------------------------------------------------
# Compute pairwise Euclidean square distance for all pairs.
#
# INPUTS:
#   features1     [128 x m] descriptors of first image
#   features2     [128 x n] descriptors of second image
#
# OUTPUTS:
#   D             [m x n] distance matrix
#
#---------------------------------------------------------
function euclideansquaredist(features1::Array{Float64,2},features2::Array{Float64,2})
    # vectorized notation
    # (a - b)^2 = a^2 - 2ab + b^2
    a_squared = transpose(sum(features1.^2, dims=(1)))
    b_squared = sum(features2.^2, dims=(1))
    ab = transpose(features1) * features2
    D = a_squared .- 2 * ab .+ b_squared
  @assert size(D) == (size(features1,2),size(features2,2))
  return D::Array{Float64,2}
end


#---------------------------------------------------------
# Find pairs of corresponding interest points given the
# distance matrix.
#
# INPUTS:
#   p1      [m x 2] keypoint coordinates in first image.
#   p2      [n x 2] keypoint coordinates in second image.
#   D       [m x n] distance matrix
#
# OUTPUTS:
#   pairs   [min(N,M) x 4] vector s.t. each row holds
#           the coordinates of an interest point in p1 and p2.
#
#---------------------------------------------------------
function findmatches(p1::Array{Int,2},p2::Array{Int,2},D::Array{Float64,2})
    m = size(p1, 1)
    n = size(p2, 1)
    
    # swap p1 and p2 if p2 is larger
    # in that case, we also need to transpose the distance matrix
    if m <= n
        p_small = p1
        p_large = p2
    else
        p_small = p2
        p_large = p1
        D = D'
    end
    
    # for each point p_i in the smaller list,
    # add the point from the larger list that is closest to p_i
    pairs = zeros(Int, size(p_small, 1), 4)
    for i in 1:size(p_small, 1)
        pairs[i,:] = [p_small[i,:] p_large[argmin(D[i,:]),:]]
    end 

  @assert size(pairs) == (min(size(p1,1),size(p2,1)),4)
  return pairs::Array{Int,2}
end


#---------------------------------------------------------
# Show given matches on top of the images in a single figure.
# Concatenate the images into a single array.
#
# INPUTS:
#   im1     first grayscale image
#   im2     second grayscale image
#   pairs   [n x 4] vector of coordinates containing the
#           matching pairs.
#
#---------------------------------------------------------
function showmatches(im1::Array{Float64,2},im2::Array{Float64,2},pairs::Array{Int,2})
    # concatenate images
    im = [im1 im2]
    
    # get matches in first and second image
    matches1 = pairs[:,1:2]
    matches2 = pairs[:,3:end]
    
    # adjust second image indices by first image's size (due to concatenation)
    matches2[:,1] = matches2[:,1] .+ size(im1,2)

    PyPlot.figure()
    PyPlot.imshow(im, "gray")
    
    # plot the matches (-1 due to python's zero indexing)
    plot(matches1[:,1].-1, matches1[:,2].-1, "xy", linewidth=8)
    plot(matches2[:,1].-1, matches2[:,2].-1, "xy", linewidth=8)

  return nothing::Nothing
end


#---------------------------------------------------------
# Computes the required number of iterations for RANSAC.
#
# INPUTS:
#   p    probability that any given correspondence is valid
#   k    number of samples drawn per iteration
#   z    total probability of success after all iterations
#
# OUTPUTS:
#   n   minimum number of required iterations
#
#---------------------------------------------------------
function computeransaciterations(p::Float64,k::Int,z::Float64)
    # n = log(1-z) / log(1-p^k), rounded up to an integer
    n = Int(ceil(log(1 - z) / log(1 - p^k)))
  return n::Int
end


#---------------------------------------------------------
# Randomly select k corresponding point pairs.
#
# INPUTS:
#   points1    given points in first image
#   points2    given points in second image
#   k          number of pairs to select
#
# OUTPUTS:
#   sample1    selected [kx2] pair in left image
#   sample2    selected [kx2] pair in right image
#
#---------------------------------------------------------
function picksamples(points1::Array{Int,2},points2::Array{Int,2},k::Int)
    # get a random permutation of the indices of points1
    # and pick k items of the permutation
    # this ensure that we alway get exactly k items
    rand_ind = randperm(size(points1,1))[1:k]
    
    # access both lists at the random indices
    sample1 = points1[rand_ind,:]
    sample2 = points2[rand_ind,:]

  @assert size(sample1) == (k,2)
  @assert size(sample2) == (k,2)
  return sample1::Array{Int,2},sample2::Array{Int,2}
end


#---------------------------------------------------------
# Conditioning: Normalization of coordinates for numeric stability.
#
# INPUTS:
#   points    unnormalized coordinates
#
# OUTPUTS:
#   U         normalized (conditioned) coordinates
#   T         [3x3] transformation matrix that is used for
#                   conditioning
#
#---------------------------------------------------------
function condition(points::Array{Float64,2})
    # s = 1/2 max(||x_i||)
    s = 0.5 * maximum(sqrt.(sum(points.^2, dims=2)))
    
    # t = mean(x_i)
    tx = mean(points[1,:])
    ty = mean(points[2,:])
    
    # build T matrix (see l8 slide 14)
    T = [1/s 0 -tx/s;
        0 1/s -ty/s;
        0 0 1]
    
    # transform the points (in homogeneous coordinates)
    U = T * Common.cart2hom(points')
    # transform back to cartesian coordinates
    U = collect(Common.hom2cart(U)')

  @assert size(U) == size(points)
  @assert size(T) == (3,3)
  return U::Array{Float64,2},T::Array{Float64,2}
end


#---------------------------------------------------------
# Estimates the homography from the given correspondences.
#
# INPUTS:
#   points1    correspondences in left image
#   points2    correspondences in right image
#
# OUTPUTS:
#   H         [3x3] estimated homography
#
#---------------------------------------------------------
function computehomography(points1::Array{Int,2}, points2::Array{Int,2})
    # condition both sets of points
    U1, T1 = condition(Float64.(points1))
    U2, T2 = condition(Float64.(points2))
    
    # work in homogeneous coordinates
    U1 = Common.cart2hom(U1')
    U2 = Common.cart2hom(U2')
    
    # build linear equation system
    # (see l8 slide 13)
    n_eqns = 2*size(points1,1)
    A = zeros(n_eqns,9)
    for j = 1:2:n_eqns
        i = Int((j+1)/2)
        # U1[:,i] = [x y 1]
        # U1[:,1]' * -U2[2,i] = [-xy' -yy' -y']
        # => A[j,:] = [0 0 0 x y 1 -xy' -yy' -y']
        A[j,:] = [0 0 0 U1[:,i]' U1[:,i]' .* -U2[2,i]]
        # -U1[:,i]' = [-x -y -1]
        # U1[:,i]' .* U2[1,i] = [U1[:,i]' .* U2[1,i]]
        # => A[j+1,:] = [-x -y 1 0 0 0 U1[:,i]' .* U2[1,i]]
        A[j+1,:] = [-U1[:,i]' 0 0 0  U1[:,i]' .* U2[1,i]]
    end

    # perform singular value decomposition
    U, S, V = svd(A, full=true)
    
    # take the last right singular vector and reshape into 3x3 matrix
    H = reshape(V[:,end], 3,3)'
    
    # undo the conditioning
    H = inv(T2) * H * T1

  @assert size(H) == (3,3)
  return H::Array{Float64,2}
end


#---------------------------------------------------------
# Computes distances for keypoints after transformation
# with the given homography.
#
# INPUTS:
#   H          [3x3] homography
#   points1    correspondences in left image
#   points2    correspondences in right image
#
# OUTPUTS:
#   d2         distance measure using the given homography
#
#---------------------------------------------------------
function computehomographydistance(H::Array{Float64,2},points1::Array{Int,2},points2::Array{Int,2})
    
    # convert to cartesian coordinates
    x1 = Common.cart2hom(points1')
    x2 = Common.cart2hom(points2')
    
    # d = ||H*x1 - x2||^2 + ||x1 - inv(H)*x2||^2
    leftdist = (Common.hom2cart(H * x1) - points2').^2
    rightdist = (points1' - Common.hom2cart(inv(H) * x2)).^2
    d2 = sum((leftdist + rightdist)', dims=2)

  @assert length(d2) == size(points1,1)
  return d2::Array{Float64,2}
end


#---------------------------------------------------------
# Compute the inliers for a given distances and threshold.
#
# INPUTS:
#   distance   homography distances
#   thresh     threshold to decide whether a distance is an inlier
#
# OUTPUTS:
#  n          number of inliers
#  indices    indices (in distance) of inliers
#
#---------------------------------------------------------
function findinliers(distance::Array{Float64,2},thresh::Float64)
    # find all indices where the distance is smaller than the threshold
    indices = [i[1] for i in findall(distance .< thresh)]
    # count the indices
    n = length(indices)
  return n::Int,indices::Array{Int,1}
end


#---------------------------------------------------------
# RANSAC algorithm.
#
# INPUTS:
#   pairs     potential matches between interest points.
#   thresh    threshold to decide whether a homography distance is an inlier
#   n         maximum number of RANSAC iterations
#
# OUTPUTS:
#   bestinliers   [n x 1 ] indices of best inliers observed during RANSAC
#
#   bestpairs     [4x4] set of best pairs observed during RANSAC
#                 i.e. 4 x [x1 y1 x2 y2]
#
#   bestH         [3x3] best homography observed during RANSAC
#
#---------------------------------------------------------
function ransac(pairs::Array{Int,2},thresh::Float64,n::Int)

    # initialize best values to zero
    bestinliers = zeros(n, 1)
    bestpairs = zeros(4, 4)
    bestH = zeros(3, 3)
    bestn = 0
    
    # get points from left and right image
    points1 = pairs[:,1:2]
    points2 = pairs[:,3:4]
    for i in 1:n
        # compute homography for a sample of 4 corresponding points
        sample1, sample2 = picksamples(points1, points2, 4)
        H = computehomography(sample1, sample2)
        
        # if H does not have full rank, we get numerical instabilities
        # so skip in this case
        if det(H) == 0
            i -= 1
            continue
        end
    
        # compute distance and number of inliers
        d = computehomographydistance(H, points1, points2)
        n, ind = findinliers(d, thresh)
        
        # if we have more inliers than before, update the record
        if n > bestn
            bestn = n
            bestinliers = ind
            bestpairs = [sample1 sample2]
            bestH = H
        end
    end



  @assert size(bestinliers,2) == 1
  @assert size(bestpairs) == (4,4)
  @assert size(bestH) == (3,3)
  return bestinliers::Array{Int,1},bestpairs::Array{Int,2},bestH::Array{Float64,2}
end


#---------------------------------------------------------
# Recompute the homography based on all inliers
#
# INPUTS:
#   pairs     pairs of keypoints
#   inliers   inlier indices.
#
# OUTPUTS:
#   H         refitted homography using the inliers
#
#---------------------------------------------------------
function refithomography(pairs::Array{Int64,2}, inliers::Array{Int64,1})
    # take only the pairs that are inliers
    H = computehomography(pairs[inliers,1:2], pairs[inliers,3:4])
  @assert size(H) == (3,3)
  return H::Array{Float64,2}
end


#---------------------------------------------------------
# Show panorama stitch of both images using the given homography.
#
# INPUTS:
#   im1     first grayscale image
#   im2     second grayscale image
#   H       [3x3] estimated homography between im1 and im2
#
#---------------------------------------------------------
function showstitch(im1::Array{Float64,2},im2::Array{Float64,2},H::Array{Float64,2})
    # initialize empty array for panorama image
    im = zeros(size(im1,1), 700)
    
    # create interpolatable version of im2
    im2_itp = interpolate(im2, BSpline(Linear()))
    
    # for each point in the panorama image
    for i in 1:size(im, 1)
        for j in 1:size(im, 2)
            # transform index from coordinates of first image
            # to coordinates of second image using the homography
            idx = Common.hom2cart(H * Common.cart2hom([j; i]))
            # if the index is valid for im2 (we're currently not doing extrapolation)
            if (1 <= idx[1] <= size(im2, 2)) && (1 <= idx[2] <= size(im2, 1))
                # set the pixel in the panorama image to the interpolated value
                im[i,j] = im2_itp(idx[2], idx[1])
            end
        end
    end
    
    # the leftmost 300 pixels from im1 are used directly
    im[1:size(im1,1),1:300] = im1[:,1:300]

    PyPlot.figure()
    PyPlot.imshow(im, "gray")
    PyPlot.show()

  return nothing::Nothing
end


#---------------------------------------------------------
# Problem 2: Image Stitching
#---------------------------------------------------------
function problem2()
  # SIFT Parameters
  sigma = 1.4             # standard deviation for presmoothing derivatives

  # RANSAC Parameters
  ransac_threshold = 50.0 # inlier threshold
  p = 0.5                 # probability that any given correspondence is valid
  k = 4                   # number of samples drawn per iteration
  z = 0.99                # total probability of success after all iterations

  # load images
  im1 = PyPlot.imread("a3p2a.png")
  im2 = PyPlot.imread("a3p2b.png")

  # Convert to double precision
  im1 = Float64.(im1)
  im2 = Float64.(im2)

  # load keypoints
  keypoints1, keypoints2 = loadkeypoints("keypoints.jld2")

  # extract SIFT features for the keypoints
  features1 = Common.sift(keypoints1,im1,sigma)
  features2 = Common.sift(keypoints2,im2,sigma)

  # compute chi-square distance  matrix
  D = euclideansquaredist(features1,features2)

  # find matching pairs
  pairs = findmatches(keypoints1,keypoints2,D)

  # show matches
  showmatches(im1,im2,pairs)
  title("Putative Matching Pairs")

  # compute number of iterations for the RANSAC algorithm
  niterations = computeransaciterations(p,k,z)

  # apply RANSAC
  bestinliers,bestpairs,bestH = ransac(pairs,ransac_threshold,niterations)
  @printf(" # of bestinliers : %d", length(bestinliers))
  
  # show best matches
  showmatches(im1,im2,bestpairs)
  title("Best 4 Matches")

  # show all inliers
  showmatches(im1,im2,pairs[bestinliers,:])
  title("All Inliers")

  # stitch images and show the result
  showstitch(im1,im2,bestH)

  # recompute homography with all inliers
  H = refithomography(pairs,bestinliers)
  showstitch(im1,im2,H)

  return nothing::Nothing
end
