module Common

using Images
using PyPlot
using ImageFiltering

export
  detect_interestpoints,
  sift,
  meshgrid


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
function nlfilter(img, fun, m=3, n=3, border="replicate")

  # lambda for linearized view
  fun_linearized(x) = fun(x[:])
  # map windows
  img_filtered = ImageFiltering.mapwindow(fun_linearized, img, [m, n], border=border)
  return img_filtered

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
# Computes structure tensor.
#
# INPUTS:
#   img             grayscale color image
#   sigma           std for presmoothing derivatives
#   sigma_tilde     std for presmoothing coefficients
#   fsize           filter size to use for presmoothing
#
# OUTPUTS:
#   S_xx       first diagonal coefficient of structure tensor
#   S_yy       second diagonal coefficient of structure tensor
#   S_xy       off diagonal coefficient of structure tensor
#
#---------------------------------------------------------
function computetensor(img::Array{Float64,2},sigma::Float64,sigma_tilde::Float64,fsize::Int)

  # Smoothing and derivative filters
  g = gauss2d(sigma, [fsize, fsize])
  g_tilde = gauss2d(sigma_tilde, [fsize, fsize])
  dx = [[0, -0.5, 0] [0, 0, 0] [0, 0.5, 0]]

  # Compute derivatives
  img_smoothed = imfilter(img, centered(g), "replicate")
  img_dx = imfilter(img_smoothed, centered(dx), "replicate")
  img_dy = imfilter(img_smoothed, centered(dx'), "replicate")

  # compute coefficients
  S_xx = imfilter(img_dx.^2, centered(g_tilde), "replicate")
  S_yy = imfilter(img_dy.^2, centered(g_tilde), "replicate")
  S_xy = imfilter(img_dx.*img_dy, centered(g_tilde), "replicate")

  return S_xx::Array{Float64,2},S_yy::Array{Float64,2},S_xy::Array{Float64,2}
end


#---------------------------------------------------------
# Computes Harris function values.
#
# INPUTS:
#   S_xx       first diagonal coefficient of structure tensor
#   S_yy       second diagonal coefficient of structure tensor
#   S_xy       off diagonal coefficient of structure tensor
#   sigma      std that was used for presmoothing derivatives
#   alpha      weighting factor for trace
#
# OUTPUTS:
#   harris     Harris function score
#
#---------------------------------------------------------
function computeharris(S_xx::Array{Float64,2},S_yy::Array{Float64,2},S_xy::Array{Float64,2}, sigma::Float64, alpha::Float64)
  v_det = S_xx.*S_yy - S_xy.^2
  v_trace = S_xx + S_yy
  harris = sigma^4 * (v_det - alpha*v_trace.^2)
  return harris::Array{Float64,2}
end


#---------------------------------------------------------
# Non-maximum suppression of Harris function values.
#   Extracts local maxima within a 5x5 stencils.
#   Allows multiple points with equal values within the same window.
#   Applies thresholding with the given threshold.
#
# INPUTS:
#   harris     Harris function score
#   thresh     param for thresholding Harris function
#   boundary   number of pixels to ignore at boundaries
#
# OUTPUTS:
#   px        x-position of kept Harris interest points
#   py        y-position of kept Harris interest points
#
#---------------------------------------------------------
function nonmaxsupp(harris::Array{Float64,2}, thresh::Float64, boundary::Int)
  # apply maximum filter
  harris_max = nlfilter(harris, maximum, 5, 5, Fill(-Inf))

  # if value is local maximum and above threshold, keep it !
  mask = (harris .>= harris_max) .* (harris .> thresh)

  # Get rid of interest points close to the boundaries
  k = boundary-1
  mask[1:1+k,:] .= 0
  mask[end-k+1:end,:] .= 0
  mask[:,1:1+k] .= 0
  mask[:,end-k+1:end] .= 0
  rows,columns = findnonzero(mask)

  return rows::Array{Int,1}, columns::Array{Int,1}
end


#---------------------------------------------------------
# Detect Harris interest points.
#   Includes non-maximum suppression and thresholding.
#
# INPUTS:
#   img             [m x n x 1] gray scale image
#   sigma           base std for smoothing
#   fsize           filter size to use for presmoothing
#   threshold       threshold that defines which interest points
#                   are kept after non-maximum suppression
#   boundary        Number of boundary pixels to ignore
#
# OUTPUTS:
#   py              y-coordinates of interest points in [1, ..., m]
#   px              x-coordinates of interest points in [1, ..., n]
#
#---------------------------------------------------------
function detect_interestpoints(img, fsize, threshold, sigma, boundary)
  # Compute Harris function values
  S_xx, S_yy, S_xy = computetensor(img, sigma, 1.6*sigma, fsize)
  harris = computeharris(S_xx, S_yy, S_xy, sigma, 0.06)
  # non-maximum suppression
  py, px = nonmaxsupp(harris, threshold, boundary)
  return py, px
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


#---------------------------------------------------------
# Matlab like meshgrid function.
#
# INPUTS:
#   px         linspace in horizontal dimension
#   py         linspace in vertical dimension
#
#
# OUTPUTS:
#   XX         x-coordinates spanned by px x py
#   YY         x-coordinates spanned by py x py
#
#---------------------------------------------------------
function meshgrid(px::StepRangeLen{Float64},py::StepRangeLen{Float64})
  XX = [i for i in px, j in py]
  YY = [j for i in px, j in py]
  return XX::Array{Float64,2}, YY::Array{Float64,2}
end

end # module
