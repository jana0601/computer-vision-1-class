module Common

using Images
using ImageFiltering

export
  rgb2gray,
  gauss2d,
  sift,
  cart2hom,
  hom2cart,
  findnonzero,
  nlfilter,
  correctindpad,
  sift


#---------------------------------------------------------
# Converts color images to grayscale.
#
# INPUTS:
#   img          [m x n x 3] color image
#
# OUTPUTS:
#   gray         [m x n] grayscale image
#
#---------------------------------------------------------
function rgb2gray(img)
  if size(img,3) != 3
    throw(DimensionMismatch("Input array must be of size NxMx3."))
  end
  gray = 0.299*img[:,:,1] + 0.587*img[:,:,2] + 0.114*img[:,:,3]
  return gray
end


#---------------------------------------------------------
# Creates Gaussian smoothing filter.
#
# INPUTS:
#   sigma        standard deviation
#   fsize        [h,w] 2-element filter size
#
# OUTPUTS:
#   g            [h x w] Gaussian filter with given std
#
#---------------------------------------------------------
function gauss2d(sigma, fsize)
    m, n = fsize[1], fsize[2]
    g = [exp(-(X.^2 + Y.^2) / (2*sigma.^2)) for X=-floor(m/2):floor(m/2), Y=-floor(n/2):floor(n/2)]
    g = g./sum(g)
    return g
end


#---------------------------------------------------------
# Converts Cartesian to homogeneous coordinates.
#
# INPUTS:
#   points_c      [dim x N] Cartesian points.
#
# OUTPUTS:
#   point_h       [(dim+1) x N] homogeneous points.
#
#---------------------------------------------------------
function cart2hom(points_c)
  points_h = [points_c; ones(1,size(points_c,2))]
  return points_h
end


#---------------------------------------------------------
# Converts homogeneous to Cartesian coordinates.
#
# INPUTS:
#   points_h      [dim x N] homogeneous points.
#
# OUTPUTS:
#   point_c       [(dim-1) x N] Cartesian points.
#
#---------------------------------------------------------
function hom2cart(points_h)
  points_c = points_h[1:end-1,:] ./ points_h[end:end,:]
  return points_c
end



#---------------------------------------------------------
# Returns indices of non-zero array elements.
#
# INPUTS:
#   A               [m x n] array
#
# OUTPUTS:
#   x               x-coordinates of non-zero elements
#   y               y-coordinates of non-zero elements
#
#---------------------------------------------------------
function findnonzero(A)
  cartesian_coordinates = findall(A .> 0)
  rows = map(i->i[1], cartesian_coordinates)
  columns = map(i->i[2], cartesian_coordinates)
  return rows, columns
end

#---------------------------------------------------------
# Applies nonlinear filter to grayscale image with a sliding window.
#
# Converts an image into MxN sliding windows and applies a function
# FUN to the 1D linearized view of these windows.
# Each pixel is then replaced by the result of this computation
# under the given boundary condition.
#
# INPUTS:
#   img             [m x n] grayscale image
#   fun             nonlinear function to be applied to each window
#   m               height of sliding window
#   n               width of sliding window
#   border          boundary conditions (outside the image domain)
#                   e.g. "replicate", "symmetric"
#
# OUTPUTS:
#   img_filtered    [m x n] nonlinear filtered image
#
#---------------------------------------------------------
function nlfilter(img, fun=median, m=3, n=3, border="replicate")

  # lambda for linearized view
  fun_linearized(x) = fun(x[:])
  # map windows
  img_filtered = ImageFiltering.mapwindow(fun_linearized, img, [m, n], border=border)
  return img_filtered

end

#---------------------------------------------------------
# Compute SIFT features for a given set of interest points.
#
# INPUTS:
#   points          [2 x n] interest point locations
#   img             grayscale image
#   sigma           standard deviation for presmoothing derivatives
#
# OUTPUTS:
#   features       [128 x n] SIFT feature descriptors
#                            (for all given interest points)
#
#---------------------------------------------------------
function sift(points,img,sigma)
  px = points[:,1]
  py = points[:,2]
  n = length(px)
  features = zeros(128,n)
  d = [[0, -0.5, 0] [0, 0, 0] [0, 0.5, 0]]
  g = gauss2d(sigma,[25 25])
  smoothed = imfilter(img,centered(g))
  dx = imfilter(smoothed,centered(d))
  dy = imfilter(smoothed,centered(d'))
  for i = 1:n
    # get patch
    r1 = [Int32(x) for x = (py[i]-7):(py[i]+8)]
    r2 = [Int32(x) for x = (px[i]-7):(px[i]+8)]
    dxp = dx[r1,r2]
    dyp = dy[r1,r2]
    # im2col adaption
    dxc = zeros(16,16)
    dyc = zeros(16,16)
    for c = 1:4
      for r = 1:4
        dxc[:,r+4*(c-1)] = dxp[1+4*(c-1):4*c,1+4*(r-1):4*r][:]
        dyc[:,r+4*(c-1)] = dyp[1+4*(c-1):4*c,1+4*(r-1):4*r][:]
      end
    end
    # compute histogram
    hist8 = zeros(8,16)
    hist8[1,:] = sum(dxc.*(dxc.>0),dims=1) # 0°
    hist8[3,:] = sum(dyc.*(dyc.>0),dims=1) # 90°
    hist8[5,:] = sum(-dxc.*(dxc.<0),dims=1) # 180°
    hist8[7,:] = sum(-dyc.*(dyc.<0),dims=1) # 270°
    idx = dyc .> -dxc
    hist8[2,:] = sum((dyc.*idx .+ dxc.*idx) ./sqrt(2),dims=1) # 45°
    idx = dyc .> dxc
    hist8[4,:] = sum((dyc.*idx .- dxc.*idx) ./sqrt(2),dims=1) # 135°
    idx = dyc .< -dxc
    hist8[6,:] = sum((-dyc.*idx .- dxc.*idx) ./sqrt(2),dims=1) # 225°
    idx = dyc .< dxc
    hist8[8,:] = sum((-dyc.*idx .+ dxc.*idx) ./sqrt(2),dims=1) # 315°
    features[:,i] = hist8[:]
  end
  # normalization
  features = features ./ sqrt.(sum(features.^2, dims=1))
  return features
end


end # module
