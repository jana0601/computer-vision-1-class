module Common

using PyPlot
using Images
using ImageFiltering

export
  loadimage,
  rgb2gray,
  cart2hom


  #---------------------------------------------------------
  # Load PNG Images
  #
  # INPUTS:
  #   filename          string
  #
  # OUTPUTS:
  #   gray, rgb         [m x n] and [m x n x 3]
  #
  #---------------------------------------------------------
  function loadimage(filename)
    rgb = PyPlot.imread(filename)
    rgb = Float64.(rgb)
    gray = Common.rgb2gray(rgb)
    return gray::Array{Float64,2}, rgb::Array{Float64,3}
  end



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


end # module
