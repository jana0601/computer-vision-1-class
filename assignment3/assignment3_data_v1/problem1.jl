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

  return rows::Array{Int,1},columns::Array{Int,1}
end


#---------------------------------------------------------
# Problem 1: Interest point detector
#---------------------------------------------------------
function problem1()
  # parameters
  sigma = ?               # std for presmoothing image
  fsize = ?               # filter size for smoothing
  threshold = ?           # Corner criterion threshold

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

  # Display interest points on top of color image
  figure()
  imshow(rgb)
  plot(columns.-1,rows.-1,"xy",linewidth=8)
  axis("off")
  title("Hessian interest points after non-maximum suppression")
  gcf()
  return nothing
end
