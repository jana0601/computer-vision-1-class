using Images  # Basic image processing functions
using PyPlot  # Plotting and image loading
using FileIO  # Functions for loading and storing data in the ".jld2" format


# Load the image from the provided .jld2 file
function loaddata()
  data = load("imagedata.jld2")["data"]
  return data::Array{Float64,2}
end


# Separate the image data into three images (one for each color channel),
# filling up all unknown values with 0
function separatechannels(data::Array{Float64,2})
    
    red_mask = zeros(size(data))
    blue_mask = zeros(size(data))
    green_mask = zeros(size(data))
    for i in 1:size(data, 1)
        for j in 1:size(data, 2)
            if (i + j) % 2 == 0
                red_mask[i,j] = 1
            elseif i % 2 == 1
                green_mask[i,j] = 1
            else
                blue_mask[i,j] = 1
            end
        end
    end

    r = red_mask .* data
    g = green_mask .* data
    b = blue_mask .* data
    return r::Array{Float64,2},g::Array{Float64,2},b::Array{Float64,2}
end


# Combine three color channels into a single image
function makeimage(r::Array{Float64,2},g::Array{Float64,2},b::Array{Float64,2})
    @assert size(r) == size(g)
    @assert size(g) == size(b)
    
    image = Array{Float64}(undef, size(r,1), size(r,2), 3)
    image[:,:,1] = r
    image[:,:,2] = g
    image[:,:,3] = b
    return image::Array{Float64,3}
end



# Interpolate missing color values using bilinear interpolation
function interpolate(r::Array{Float64,2},g::Array{Float64,2},b::Array{Float64,2})
    h_r = centered([0 1 0;
                    1 4 1; 
                    0 1 0] / 4)

    h_gb = centered([1 2 1;
                     2 4 2;
                     1 2 1] / 4)
    
    r = imfilter(r, h_r, "symmetric")
    g = imfilter(g, h_gb, "symmetric")
    b = imfilter(b, h_gb, "symmetric")
    image = makeimage(r, g, b)
    return image::Array{Float64,3}
end



# Display two images in a single figure window
function displayimages(img1::Array{Float64,3}, img2::Array{Float64,3})
    f, axes = subplots(1,2)
    axes[1][:imshow](img1)
    axes[2][:imshow](img2)
    for ax in axes
        ax[:axis]("off")
    end
    show()
end

#= Problem 2
Bayer Interpolation =#

function problem2()
  # load raw data
  data = loaddata()
  # separate data
  r,g,b = separatechannels(data)
  # merge raw pattern
  img1 = makeimage(r,g,b)
  # interpolate
  img2 = interpolate(r,g,b)
  # display images
  displayimages(img1, img2)
  return
end
