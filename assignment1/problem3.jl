using Images
using PyPlot
using Test
using LinearAlgebra
using FileIO

# Transform from Cartesian to homogeneous coordinates
function cart2hom(points::Array{Float64,2})
    points_hom = [points; ones(1, size(points,2))]
    return points_hom::Array{Float64,2}
end

# Transform from homogeneous to Cartesian coordinates
function hom2cart(points::Array{Float64,2})
    points_cart = points[1:end-1,:] ./ transpose(points[end,:])
    return points_cart::Array{Float64,2}
end

# Translation by v
function gettranslation(v::Array{Float64,1})
    T = [Matrix{Float64}(I, 3, 3) v; zeros(1, 3) 1] 
    return T::Array{Float64,2}
end

# Rotation of d degrees around x axis
function getxrotation(d::Int)
    theta = deg2rad(d)
    Rx = [1 0 0 0; 
        0 cos(theta) -sin(theta) 0;
        0 sin(theta) cos(theta) 0;
        0 0 0 1]
    return Rx::Array{Float64,2}
end

# Rotation of d degrees around y axis
function getyrotation(d::Int)
    theta = deg2rad(d)
    Ry = [cos(theta) 0 sin(theta) 0;
        0 1 0 0;
        -sin(theta) 0 cos(theta) 0;
        0 0 0 1]
    return Ry::Array{Float64,2}
end

# Rotation of d degrees around z axis
function getzrotation(d::Int)
    theta = deg2rad(d)
    Rz = [cos(theta) -sin(theta) 0 0
        sin(theta) cos(theta) 0 0
        0 0 1 0;
        0 0 0 1]
  return Rz::Array{Float64,2}
end


# Central projection matrix (including camera intrinsics)
function getcentralprojection(principal::Array{Int,1}, focal::Int)
    K = Float64.([focal 0 principal[1] 0;
        0 focal principal[2] 0;
        0 0 1 0])
    return K::Array{Float64,2}
end


# Return full projection matrix P and full model transformation matrix M
function getfullprojection(T::Array{Float64,2},Rx::Array{Float64,2},Ry::Array{Float64,2},Rz::Array{Float64,2},V::Array{Float64,2})
    M = Rz * Rx * Ry * T
    P = V * M
    return P::Array{Float64,2},M::Array{Float64,2}
end


# Load 2D points
function loadpoints()
    points = load("obj2d.jld2")["x"]
    return points::Array{Float64,2}
end


# Load z-coordinates
function loadz()
    z = load("zs.jld2")["Z"]
    return z::Array{Float64,2}
end



# Invert just the central projection P of 2d points *P2d* with z-coordinates *z*
function invertprojection(P::Array{Float64,2}, P2d::Array{Float64,2}, z::Array{Float64,2})
    P3d = pinv(P) * cart2hom(P2d) .* z .+ [0; 0; 0 ; 1]
    return P3d::Array{Float64,2}
end


# Invert just the model transformation of the 3D points *P3d*
function inverttransformation(A::Array{Float64,2}, P3d::Array{Float64,2})
    X = A \ P3d
    return X::Array{Float64,2}
end


# Plot 2D points
function displaypoints2d(points::Array{Float64,2})
    PyPlot.figure()
    PyPlot.scatter(points[1,:], points[2,:])
    return gcf()::Figure
end

# Plot 3D points
function displaypoints3d(points::Array{Float64,2})
    PyPlot.figure()
    PyPlot.scatter3D(points[1,:], points[2,:], points[3,:])
    return gcf()::Figure
end

# Apply full projection matrix *C* to 3D points *X*
function projectpoints(P::Array{Float64,2}, X::Array{Float64,2})
    P2d = hom2cart(P * cart2hom(X))
    return P2d::Array{Float64,2}
end



#= Problem 2
Projective Transformation =#

function problem3()
  # parameters
  t               = [6.7; -10; 4.2]
  principal_point = [9; -7]
  focal_length    = 8

  # model transformations
  T = gettranslation(t)
  Ry = getyrotation(-45)
  Rx = getxrotation(120)
  Rz = getzrotation(-10)

  # central projection including camera intrinsics
  K = getcentralprojection(principal_point,focal_length)

  # full projection and model matrix
  P,M = getfullprojection(T,Rx,Ry,Rz,K)

  # load data and plot it
  points = loadpoints()
  displaypoints2d(points)

  # reconstruct 3d scene
  z = loadz()
  Xt = invertprojection(K,points,z)
  Xh = inverttransformation(M,Xt)

  worldpoints = hom2cart(Xh)
  displaypoints3d(worldpoints)

  # reproject points
  points2 = projectpoints(P,worldpoints)
  displaypoints2d(points2)

  @test points ≈ points2
  
  # Question 1
  # Yes, the points match! This should be the case, since we
  # apply a transformation and then its inverse.
  
  # Question 2
  # Why is it necessary to provide the z-coordinates?
  # When applying the perspective projection, the z-coordinate is essentially lost.
  # Points that lie on the same ray from the camera center are projected onto the same point.
  # Thus, a 2D image of points in the world is not enough to reconstruct 3D structure.

  # Question 3
  # Are the rotation and translation operations commutative?
  # The rotation and translation operations are not commutative.
  # This can be seen from the example below in the code or understood intuitively.
  # If we first translate an object in world coordinates and then rotate it,
  # it is not the same as applying the same translation after rotating 
  # an object to align with the camera coordinate system.  This is because after the rotation,
  # the x, y and z axes point in different directions as before.
  R = Rz * Ry * Rx
  @test ~ isapprox(R * T, T * R)
  return
end
