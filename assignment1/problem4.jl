using Images
using PyPlot

# Create 3x3 derivative filters in x and y direction
function createfilters()
    d = [1 0 -1] / 2
    g = exp.(-0.5 * [-1, 0, 1].^2 / 0.9^2)
    g = g / sum(g)
    fx = g * d
    fy = transpose(d) * transpose(g)
    return fx::Array{Float64,2}, fy::Array{Float64,2}
end

# Apply derivate filters to an image and return the derivative images
function filterimage(I::Array{Float32,2},fx::Array{Float64,2},fy::Array{Float64,2})

    Ix = imfilter(I, centered(fx), "replicate")
    Iy = imfilter(I, centered(fy), "replicate")
    return Ix::Array{Float64,2},Iy::Array{Float64,2}
end

# Apply thresholding on the gradient magnitudes to detect edges
function detectedges(Ix::Array{Float64,2},Iy::Array{Float64,2}, thr::Float64)
    grad_magn = sqrt.(Ix.^2 + Iy.^2)
    edges = grad_magn .- thr
    return edges::Array{Float64,2}
end


# Apply non-maximum-suppression
function nonmaxsupp(edges::Array{Float64,2},Ix::Array{Float64,2},Iy::Array{Float64,2})
    maxima = zeros(size(edges))
    
    # compute direction angle in degrees between 0 and 360
    directions = mod.(atand.(Iy, Ix), 360)
    
    # round to 45 degrees and divide into 4 possible cases (E/W, N/S, NW/SE, NE/SW)
    rounded_dir = mod.(Int.(round.(directions / 45)), 4)
    for i in 2:size(edges,1)-1
        for j in 2:size(edges,2)-1

            if edges[i,j] > 0
                # 0 / 180 deg
                if rounded_dir[i,j] == 0.0
                    if edges[i,j] >= edges[i,j + 1] && edges[i,j] >= edges[i,j - 1]
                        maxima[i,j] = edges[i,j]
                    end
                end
                # 45 / 225 deg
                if rounded_dir[i,j] == 1.0
                    if edges[i,j] >= edges[i - 1,j + 1] && edges[i,j] >= edges[i + 1,j - 1]
                        maxima[i,j] = edges[i,j]
                    end

                end
                # 90 / 270 deg
                if rounded_dir[i,j] == 2.0
                    if edges[i,j] >= edges[i-1,j] && edges[i,j] >= edges[i+1,j]
                        maxima[i,j] = edges[i,j]
                    end
                end
                # 135 / 315 deg
                if rounded_dir[i,j] == 3.0
                    if edges[i,j] >= edges[i + 1,j - 1] && edges[i,j] >= edges[i - 1,j + 1]
                        maxima[i,j] = edges[i,j]
                    end
                end
            end
        end
    end
    edges = maxima
    return edges::Array{Float64,2}
end


#= Problem 4
Image Filtering and Edge Detection =#

function problem4()

  # load image
  img = PyPlot.imread("a1p4.png")

  # create filters
  fx, fy = createfilters()

  # filter image
  imgx, imgy = filterimage(img, fx, fy)

  # show filter results
  figure()
  subplot(121)
  imshow(imgx, "gray", interpolation="none")
  title("x derivative")
  axis("off")
  subplot(122)
  imshow(imgy, "gray", interpolation="none")
  title("y derivative")
  axis("off")
  gcf()

  # show gradient magnitude
  figure()
  imshow(sqrt.(imgx.^2 + imgy.^2),"gray", interpolation="none")
  axis("off")
  title("Derivative magnitude")
  gcf()

  # threshold derivative
  threshold = 38. / 255.
  edges = detectedges(imgx,imgy,threshold)
  figure()
  imshow(edges.>0, "gray", interpolation="none")
  axis("off")
  title("Binary edges")
  gcf()

  # non maximum suppression
  edges2 = nonmaxsupp(edges,imgx,imgy)
  figure()
  imshow(edges2,"gray", interpolation="none")
  axis("off")
  title("Non-maximum suppression")
  gcf()
  return
    
  # TODO: add answer to question from assignment sheet
  # Question:
  # Experiment with various threshold
  # values and choose one that shows the "important" image  
  # edges. Briey explain with a comment in the code
  # how you found this threshold and why you chose it.

end
