using Images
using PyPlot
using Statistics
using LinearAlgebra
using Printf

@inline isBigger(a, b) = (a > b)
@inline isSmaller(a, b) = (a < b)

include("Common.jl")

#---------------------------------------------------------
# Load ground truth disparity map
#
# Input:
#   filename:       the name of the disparity file
#
# Outputs:
#   disparity:      [h x w] ground truth disparity
#   mask:           [h x w] validity mask
#---------------------------------------------------------
function loadGTdisaprity(filename)

  @assert size(mask) == size(disparity_gt)
  return disparity_gt::Array{Float64,2}, mask::BitArray{2}
end

#---------------------------------------------------------
# Calculate NC between two image patches
#
# Inputs:
#   patch1 : an image patch from the left image
#   patch2 : an image patch from the right image
#
# Output:
#   nc_cost : Normalized Correlation cost
#
#---------------------------------------------------------
function computeNC(patch1, patch2)

  return nc_cost::Float64
end

#---------------------------------------------------------
# Calculate SSD between two image patches
#
# Inputs:
#   patch1 : an image patch from the left image
#   patch2 : an image patch from the right image
#
# Output:
#   ssd_cost : SSD cost
#
#---------------------------------------------------------
function computeSSD(patch1, patch2)

  return ssd_cost::Float64
end


#---------------------------------------------------------
# Calculate the error of estimated disparity
#
# Inputs:
#   disparity : estimated disparity result, [h x w]
#   disparity_gt : ground truth disparity, [h x w]
#   valid_mask : validity mask, [h x w]
#
# Output:
#   error_disparity : calculated disparity error
#   error_map:  error map, [h x w]
#
#---------------------------------------------------------
function calculateError(disparity, disparity_gt, valid_mask)

  @assert size(disparity) == size(error_map)
  return error_disparity::Float64, error_map::Array{Float64,2}
end


#---------------------------------------------------------
# Compute disparity
#
# Inputs:
#   gray_l : a gray version of the left image, [h x w]
#   gray_R : a gray version of the right image, [h x w]
#   max_disp: Maximum disparity for the search range
#   w_size: window size
#   cost_ftn: a cost function for caluclaing the cost between two patches.
#             It can be either computeSSD or computeNC.
#
# Output:
#   disparity : disparity map, [h x w]
#
#---------------------------------------------------------
function computeDisparity(gray_l, gray_r, max_disp, w_size, cost_ftn::Function)


  @assert size(disparity) == size(gray_l)
  return disparity::Array{Int64,2}
end

#---------------------------------------------------------
#   An efficient implementation
#---------------------------------------------------------
function computeDisparityEff(gray_l, gray_r, max_disp, w_size)


  @assert size(disparity) == size(gray_l)
  return disparity::Array{Int64,2}
end

#---------------------------------------------------------
# Problem 2: Stereo matching
#---------------------------------------------------------
function problem2()

  # Define parameters
  w_size = [5 5]
  max_disp = 100
  gt_file_name = "a4p2_gt.png"

  # Load both images
  gray_l, rgb_l = Common.loadimage("a4p2_left.png")
  gray_r, rgb_r = Common.loadimage("a4p2_right.png")

  # Load ground truth disparity
  disparity_gt, valid_mask = loadGTdisaprity(gt_file_name)

  # estimate disparity
  @time disparity_ssd = computeDisparity(gray_l, gray_r, max_disp, w_size, computeSSD)
  @time disparity_nc = computeDisparity(gray_l, gray_r, max_disp, w_size, computeNC)


  # Calculate Error
  error_disparity_ssd, error_map_ssd = calculateError(disparity_ssd, disparity_gt, valid_mask)
  @printf(" disparity_SSD error = %f \n", error_disparity_ssd)
  error_disparity_nc, error_map_nc = calculateError(disparity_nc, disparity_gt, valid_mask)
  @printf(" disparity_NC error = %f \n", error_disparity_nc)

  figure()
  subplot(2,1,1), imshow(disparity_ssd, interpolation="none"), axis("off"), title("disparity_ssd")
  subplot(2,1,2), imshow(error_map_ssd, interpolation="none"), axis("off"), title("error_map_ssd")
  gcf()

  figure()
  subplot(2,1,1), imshow(disparity_nc, interpolation="none"), axis("off"), title("disparity_nc")
  subplot(2,1,2), imshow(error_map_nc, interpolation="none"), axis("off"), title("error_map_nc")
  gcf()

  figure()
  imshow(disparity_gt)
  axis("off")
  title("disparity_gt")
  gcf()

  @time disparity_ssd_eff = computeDisparityEff(gray_l, gray_r, max_disp, w_size)
  error_disparity_ssd_eff, error_map_ssd_eff = calculateError(disparity_ssd_eff, disparity_gt, valid_mask)
  @printf(" disparity_SSD_eff error = %f \n", error_disparity_ssd_eff)

  figure()
  subplot(2,1,1), imshow(disparity_ssd_eff, interpolation="none"), axis("off"), title("disparity_ssd_eff")
  subplot(2,1,2), imshow(error_map_ssd_eff, interpolation="none"), axis("off"), title("error_map_ssd_eff")
  gcf()

  return nothing::Nothing
end
