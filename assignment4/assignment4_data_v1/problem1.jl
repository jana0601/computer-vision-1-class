using PyPlot
using FileIO
using Statistics
using LinearAlgebra

include("Common.jl")

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
    s = 0.5 * maximum(sqrt.(sum(points[1:2,:].^2, dims=1)))
    
    # t = mean(x_i)
    t = mean(points, dims=2)
    
    # build T matrix (see l8 slide 14)
    T = [1/s 0 -t[1]/s;
        0 1/s -t[2]/s;
        0 0 1]
    
    # transform the points (in homogeneous coordinates)
    U = T * points
  @assert size(U) == size(points)
  @assert size(T) == (3,3)
  return U::Array{Float64,2},T::Array{Float64,2}
end


#---------------------------------------------------------
# Enforce a rank of 2 to a given 3x3 matrix.
#
# INPUTS:
#   A     [3x3] matrix (of rank 3)
#
# OUTPUTS:
#   Ahat  [3x3] matrix of rank 2
#
#---------------------------------------------------------
# Enforce that the given matrix has rank 2
function enforcerank2(A::Array{Float64,2})
    # perform svd on A
    U, S, V = svd(A, full=true)
    # force the last singular value to be zero -> decreases rank by 1
    S[end] = 0
    # assamble the low-rank matrix
    Ahat = U * Diagonal(S) * V'
  @assert size(Ahat) == (3,3)
  return Ahat::Array{Float64,2}
end


#---------------------------------------------------------
# Compute fundamental matrix from conditioned coordinates.
#
# INPUTS:
#   p1     set of conditioned coordinates in left image
#   p2     set of conditioned coordinates in right image
#
# OUTPUTS:
#   F      estimated [3x3] fundamental matrix
#
#---------------------------------------------------------
# Compute the fundamental matrix for given conditioned points
function computefundamental(p1::Array{Float64,2},p2::Array{Float64,2})
    
    xl = p1[1,:]
    yl = p1[2,:]
    xr = p2[1,:]
    yr = p2[2,:]
    
    # create zero and 1 vectors to fill matrix
    I = ones(size(xr,1))
    
    # build equation system
    A = [xr.*xl yr.*xl xl xr.*yl yr.*yl yl xr yr I]

    # perform singular value decomposition
    U, S, V = svd(A, full=true)
    
    # take the last right singular vector and reshape into 3x3 matrix
    F = collect(reshape(V[:,9], 3,3)')
    
    F = enforcerank2(F)
  @assert size(F) == (3,3)
  return F::Array{Float64,2}
end


#---------------------------------------------------------
# Compute fundamental matrix from unconditioned coordinates.
#
# INPUTS:
#   p1     set of unconditioned coordinates in left image
#   p2     set of unconditioned coordinates in right image
#
# OUTPUTS:
#   F      estimated [3x3] fundamental matrix
#
#---------------------------------------------------------
function eightpoint(p1::Array{Float64,2},p2::Array{Float64,2})
    # condition both sets of points
    U1, T1 = condition(p1)
    U2, T2 = condition(p2)
    
    F = computefundamental(U1, U2)
    
    # undo the conditioning
    F = T1' * F * T2
  @assert size(F) == (3,3)
  return F::Array{Float64,2}
end


#---------------------------------------------------------
# Draw epipolar lines:
#   E.g. for a given fundamental matrix and points in first image,
#   draw corresponding epipolar lines in second image.
#
#
# INPUTS:
#   Either:
#     F         [3x3] fundamental matrix
#     points    set of coordinates in left image
#     img       right image to be drawn on
#
#   Or:
#     F         [3x3] transposed fundamental matrix
#     points    set of coordinates in right image
#     img       left image to be drawn on
#
#---------------------------------------------------------
function showepipolar(F::Array{Float64,2},points::Array{Float64,2},img::Array{Float64,3})
    lines = F' * Common.cart2hom(points')
    
    x = 1:size(img,2)
    x = repeat(x', 16)


    y = - lines[1,:] ./ lines[2,:] .* x .- lines[3,:] ./ lines[2,:]
    imshow(img)

    for i in 1:size(y,1)
        plot(x[i,:], y[i,:])
    end
    
    ax = gca()
    ax[:set_ylim]([size(img,1),0])
    ax[:set_xlim]([0,size(img,2)])
  return nothing::Nothing
end


#---------------------------------------------------------
# Compute the residual errors for a given fundamental matrix F,
# and set of corresponding points.
#
# INPUTS:
#    p1    corresponding points in left image
#    p2    corresponding points in right image
#    F     [3x3] fundamental matrix
#
# OUTPUTS:
#   residuals      residual errors for given fundamental matrix
#
#---------------------------------------------------------
function computeresidual(p1::Array{Float64,2},p2::Array{Float64,2},F::Array{Float64,2})
  residual = reshape(diag(p1' * F * p2), size(p1,2), 1)
  return residual::Array{Float64,2}
end


#---------------------------------------------------------
# Problem 1: Fundamental Matrix
#---------------------------------------------------------
function problem1()
  # Load images and points
  img1 = Float64.(PyPlot.imread("a4p1a.png"))
  img2 = Float64.(PyPlot.imread("a4p1b.png"))
  points1 = load("points.jld2", "points1")
  points2 = load("points.jld2", "points2")

  # Display images and correspondences
  figure()
  subplot(121)
  imshow(img1,interpolation="none")
  axis("off")
  scatter(points1[:,1],points1[:,2])
  title("Keypoints in left image")
  subplot(122)
  imshow(img2,interpolation="none")
  axis("off")
  scatter(points2[:,1],points2[:,2])
  title("Keypoints in right image")

  # compute fundamental matrix with homogeneous coordinates
  x1 = Common.cart2hom(points1')
  x2 = Common.cart2hom(points2')
  F = eightpoint(x1,x2)

  # draw epipolar lines
  figure()
  subplot(121)
  F_transposed = permutedims(F, [2, 1])
  showepipolar(F_transposed,points2,img1)
  scatter(points1[:,1],points1[:,2])
  title("Epipolar lines in left image")
  subplot(122)
  showepipolar(F,points1,img2)
  scatter(points2[:,1],points2[:,2])
  title("Epipolar lines in right image")


  # check epipolar constraint by computing the remaining residuals
  residual = computeresidual(x1, x2, F)
  println("Residuals:")
  println(residual)

  # compute epipoles
  U,_,V = svd(F)
  e1 = V[1:2,3]./V[3,3]
  println("Epipole 1: $(e1)")
  e2 = U[1:2,3]./U[3,3]
  println("Epipole 2: $(e2)")

  return
end
