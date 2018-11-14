using Images
using PyPlot


# Create a gaussian filter
function makegaussianfilter(size::Array{Int,2},sigma::Float64)
    # compute width of filter
    wx, wy = (size.-1) ./ 2
    # make two 1d filters
    gx = reshape([exp(- x^2 / (2 * sigma^2)) for x in -wx:wx], (size[1],1))
    gy = reshape([exp(- y^2 / (2 * sigma^2)) for y in -wy:wy], (1,size[2]))
    # normalize
    gx = gx / sum(gx)
    gy = gy / sum(gy)
    # outer product
    f = gx * gy
    return f::Array{Float64,2}
end

# Create a binomial filter
function makebinomialfilter(size::Array{Int,2})
    # binomial filter in x direction
    bx = reshape([binomial(size[1]-1, k) for k in 0:size[1]-1], (size[1],1))
    bx = bx / sum(bx)
    # binomial filter in y direction
    by = reshape([binomial(size[2]-1, k) for k in 0:size[2]-1], (1, size[2]))
    by = by / sum(by)
    # outer product
    f = bx * by
    return f::Array{Float64,2}
end

# Downsample an image by a factor of 2
function downsample2(A::Array{Float64,2})
    # skip every other row
    D = A[1:2:end, 1:2:end]
    return D::Array{Float64,2}
end

# Upsample an image by a factor of 2
function upsample2(A::Array{Float64,2},fsize::Array{Int,2})
    U = zeros(size(A) .* 2)
    # fill every second entry
    U[1:2:end, 1:2:end] = A
    # filter with a binomial filter of given size
    U = 4 * imfilter(U, centered(makebinomialfilter(fsize)), "symmetric")
  return U::Array{Float64,2}
end

# Build a gaussian pyramid from an image.
# The output array should contain the pyramid levels in decreasing sizes.
function makegaussianpyramid(im::Array{Float32,2},nlevels::Int,fsize::Array{Int,2},sigma::Float64)
    # initialize array of arrays
    G = Array{Array{Float64,2},1}(undef,nlevels)
    # the first element of the pyramid is the original image
    G[1] = Float64.(im)
    # for the remaining nlevels-1 elements
    for i in 2:nlevels
        # filter with a gaussian
        fim = imfilter(G[i-1], makegaussianfilter(fsize, sigma), "symmetric")
        # downsample by a factor of two
        G[i] = downsample2(fim)
    end
    return G::Array{Array{Float64,2},1}
end

# Display a given image pyramid (laplacian or gaussian)
function displaypyramid(P::Array{Array{Float64,2},1})
    # initialized a empty
    filled = Array{Array{Float64,2},1}(undef,size(P))
    for i in 1:size(P)[1]
        # current pyramid image
        pimg = copy(P[i])
        # normalize the current level -> (0,1)
        pimg = (pimg .- minimum(pimg)) ./ (maximum(pimg) - minimum(pimg))
        # fill the rest of the image with zeros (black)
        A = zeros(size(P[1],1), size(pimg,2))
        A[1:size(pimg,1),1:size(pimg,2)] = pimg
        filled[i] = A
    end
    # concatenate into one image and display that
    img = hcat(filled...)
    PyPlot.imshow(img, cmap="gray")
    PyPlot.axis("off")
    return nothing
end

# Build a laplacian pyramid from a gaussian pyramid.
# The output array should contain the pyramid levels in decreasing sizes.
function makelaplacianpyramid(G::Array{Array{Float64,2},1},nlevels::Int,fsize::Array{Int,2})
    # L_n = G_n
    L = Array{Array{Float64,2},1}(undef,size(G))
    L[nlevels] = copy(G[nlevels])
    # for level n-1 down to level 1
    for i in nlevels-1:-1:1
        L[i] = G[i] .- upsample2(G[i+1], fsize)
    end
    return L::Array{Array{Float64,2},1}
end

# Build a laplacian pyramid from a gaussian pyramid.
# The output array should contain the pyramid levels in decreasing sizes.
function makelaplacianpyramid(G::Array{Array{Float64,2},1},nlevels::Int,fsize::Array{Int,2})
    # L_n = G_n
    L = Array{Array{Float64,2},1}(undef,size(G))
    L[nlevels] = copy(G[nlevels])
    # for level n-1 down to level 1
    for i in nlevels-1:-1:1
        L[i] = G[i] .- upsample2(G[i+1], fsize)
    end
    return L::Array{Array{Float64,2},1}
end
