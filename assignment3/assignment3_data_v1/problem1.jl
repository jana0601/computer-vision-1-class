using Images
using PyPlot

include("Common.jl")

#---------------------------------------------------------
# Loads grayscale and color image given PNG filename.
#
# INPUTS:
#   filename     given PNG image file
#
# OUTPUTS:
#   gray         single precision grayscale image
#   rgb          single precision color image
#
#---------------------------------------------------------
function loadimage(filename)
    rgb = Float64.(PyPlot.imread(filename))
    gray = Common.rgb2gray(rgb)
  return gray::Array{Float64,2}, rgb::Array{Float64,3}
end


#---------------------------------------------------------
# Computes entries of Hessian matrix.
#
# INPUTS:
#   img             grayscale color image
#   sigma           std for presmoothing image
#   fsize           filter size for smoothing
#
# OUTPUTS:
#   I_xx       second derivative in x-direction
#   I_yy       second derivative in y-direction
#   I_xy       derivative in x- and y-direction
#
#---------------------------------------------------------
function computehessian(img::Array{Float64,2},sigma::Float64,fsize::Int)
    # filter image with a gaussian kernel
    g = Common.gauss2d(sigma, [fsize, fsize])
    img_filtered = imfilter(img, centered(g), "replicate")
    
    # central derivative filters
    fx = centered([1 0 -1] / 2)
    fy = centered([1 0 -1]' / 2)
    
    # first derivatives
    I_x = imfilter(img_filtered, fx, "replicate")
    I_y = imfilter(img_filtered, fy, "replicate")
    
    # second derivatives
    I_xx = imfilter(I_x, fx, "replicate")
    I_yy = imfilter(I_y, fy, "replicate")
    I_xy = imfilter(I_x, fy, "replicate")
  return I_xx::Array{Float64,2},I_yy::Array{Float64,2},I_xy::Array{Float64,2}
end


#---------------------------------------------------------
# Computes function values of Hessian criterion.
#
# INPUTS:
#   I_xx       second derivative in x-direction
#   I_yy       second derivative in y-direction
#   I_xy       derivative in x- and y-direction
#   sigma      std that was used for smoothing image
#
# OUTPUTS:
#   criterion  function score
#
#---------------------------------------------------------
function computecriterion(I_xx::Array{Float64,2},I_yy::Array{Float64,2},I_xy::Array{Float64,2}, sigma::Float64)
    # criterion = sigma^4 * det(H)
    criterion = sigma^4 .* (I_xx .* I_yy - I_xy .^2)
  return criterion::Array{Float64,2}
end


#---------------------------------------------------------
# Non-maximum suppression of criterion function values.
#   Extracts local maxima within a 5x5 window and
#   allows multiple points with equal values within the same window.
#   Discards interest points in a 5 pixel boundary.
#   Applies thresholding with the given threshold.
#
# INPUTS:
#   criterion  function score
#   thresh     param for thresholding
#
# OUTPUTS:
#   rows        row positions of kept interest points
#   columns     column positions of kept interest points
#
#---------------------------------------------------------
function nonmaxsupp(criterion::Array{Float64,2}, thresh::Float64)
    # compute maxima in a 5x5 region
    maxima = criterion .- Common.nlfilter(criterion, maximum, 5, 5) .== 0.0

    # throw away all interest points in a 5 pixel boundary at image edges
    maxima[1:5,:] .= false
    maxima[end-4:end,:] .= false
    maxima[:,1:5] .= false
    maxima[:,end-4:end] .= false
    
    # find all points that are larger than the threshold
    mask_thresh = criterion .> thresh

    # get all indices where the criterion is a local maximum and larger than the threshold
    indices = findall(maxima .& mask_thresh)
    
    # get rows and columns from indices
    rows = [index[1] for index in indices]
    columns = [index[2] for index in indices]
    
  return rows::Array{Int,1},columns::Array{Int,1}
end


#---------------------------------------------------------
# Problem 1: Interest point detector
#---------------------------------------------------------
function problem1()
  # parameters
  sigma = 4.5              # std for presmoothing image
  fsize = 25             # filter size for smoothing
  threshold = 1e-3      # Corner criterion threshold

  # Load both colored and grayscale image from PNG file
  gray,rgb = loadimage("a3p1.png")

  # Compute the three components of the Hessian matrix
  I_xx,I_yy,I_xy = computehessian(gray,sigma,fsize)

  # Compute Hessian based corner criterion
  criterion = computecriterion(I_xx,I_yy,I_xy,sigma)

  # Display Hessian criterion image
  figure()
  imshow(criterion,"jet",interpolation="none")
  axis("off")
  title("Determinant of Hessian")
  gcf()

  # Threshold corner criterion
  mask = criterion .> threshold
  rows, columns = Common.findnonzero(mask)
  figure()
  imshow(rgb)
  plot(columns.-1,rows.-1,"xy",linewidth=8)
  axis("off")
  title("Hessian interest points without non-maximum suppression")
  gcf()

  # Apply non-maximum suppression
  rows,columns = nonmaxsupp(criterion,threshold)
  print(size(rows))

  # Display interest points on top of color image
  figure()
  imshow(rgb)
  plot(columns.-1,rows.-1,"xy",linewidth=8)
  axis("off")
  title("Hessian interest points after non-maximum suppression")
  gcf()
  return nothing
end
