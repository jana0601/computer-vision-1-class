using Images
using PyPlot
using Clustering
using MultivariateStats
using Printf
using Random

include("Common.jl")

#---------------------------------------------------------
# Type aliases for arrays of images/features
#---------------------------------------------------------
const ImageList = Array{Array{Float64,2},1}
const FeatureList = Array{Array{Float64,2},1}


#---------------------------------------------------------
# Structure for storing datasets
#
# Fields:
#   images      images associated with this dataset
#   labels      corresponding labels
#   n           number of examples
#---------------------------------------------------------
struct Dataset
  images::Array{Array{Float64,2},1}
  labels::Array{Float64,1}
  n::Int
end

#---------------------------------------------------------
# Provides Dataset.length() method.
#---------------------------------------------------------
import Base.length
function length(x::Dataset)
  @assert length(x.images) == length(x.labels) == x.n "The length of the dataset is inconsistent."
  return x.n
end


#---------------------------------------------------------
# Structure for storing SIFT parameters.
#
# Fields:
#   fsize         filter size
#   sigma         standard deviation for filtering
#   threshold     SIFT threshold
#   boundary      number of boundary pixels to ignore
#---------------------------------------------------------
struct Parameters
  fsize::Int
  sigma::Float64
  threshold::Float64
  boundary::Int
end


#---------------------------------------------------------
# Helper: Concatenates two datasets.
#---------------------------------------------------------
function concat(d1::Dataset, d2::Dataset)
  return Dataset([d1.images; d2.images], [d1.labels; d2.labels], d1.n + d2.n)
end


#---------------------------------------------------------
# Helper: Create a train/test split of a dataset.
#---------------------------------------------------------
function traintestsplit(d::Dataset, p::Float64)
  ntrain = Int(floor(d.n * p))
  ntest = d.n - ntrain
  permuted_idx = randperm(d.n)

  train = Dataset(d.images[permuted_idx[1:ntrain]], d.labels[permuted_idx[1:ntrain]], ntrain)
  test = Dataset(d.images[permuted_idx[1+ntrain:end]], d.labels[permuted_idx[1+ntrain:end]], ntest)

  return train,test
end


#---------------------------------------------------------
# Create input data by separating planes and bikes randomly
# into two equally sized sets.
#
# Note: Use the Dataset type from above.
#
# OUTPUTS:
#   trainingset      Dataset of length 120, contraining bike and plane images
#   testingset       Dataset of length 120, contraining bike and plane images
#
#---------------------------------------------------------
function loadimages()

  nbikes = 106 # number of planes
  nplanes = 134 # number of bikes

  ### Your implementations for loading images here -------

    bike_imgs = [imread("bikes/$(lpad(string(i),3,"0")).png") for i in 1:nbikes]
    bikes = Dataset(bike_imgs, [0 for i in 1:nbikes], nbikes)
    
    plane_imgs = [imread("planes/$(lpad(string(i),3,"0")).png") for i in 1:nplanes]
    planes = Dataset(plane_imgs, [1 for i in 1:nplanes], nplanes)

  ### ----------------------------------------------------

  trainplanes, testplanes = traintestsplit(planes, 0.5)
  trainbikes, testbikes = traintestsplit(bikes, 0.5)

  trainingset = concat(trainbikes, trainplanes)
  testingset = concat(testbikes, testplanes)

  @assert length(trainingset) == 120
  @assert length(testingset) == 120
  return trainingset::Dataset, testingset::Dataset

end


#---------------------------------------------------------
# Extract SIFT features for all images
# For each image in the images::ImageList, first find interest points by applying the Harris corner detector.
# Then extract SIFT to compute the features at these points.
# Use params.sigma for the Harris corner detector and SIFT together.
#---------------------------------------------------------
function extractfeatures(images::ImageList, params::Parameters)
    
    # init feature list
    features = FeatureList()
    
    # for each image
    for i = 1:length(images)
        img = images[i]
        
        # detect interest points
        px, py = Common.detect_interestpoints(img, params.fsize, params.threshold, params.sigma, params.boundary)
        # compute sift features
        img_features = Common.sift([py px], img, params.sigma)
        push!(features, img_features) 
    end


  @assert length(features) == length(images)
  for i = 1:length(features)
    @assert size(features[i],1) == 128
  end
  return features::FeatureList
end


#---------------------------------------------------------
# Build a concatenated feature matrix from all given features
#---------------------------------------------------------
function concatenatefeatures(features::FeatureList)

    # concatenate horizontally
    X = hcat(features...)

  @assert size(X,1) == 128
  return X::Array{Float64,2}
end

#---------------------------------------------------------
# Build a codebook for a given feature matrix by k-means clustering with K clusters
#---------------------------------------------------------
function computecodebook(X::Array{Float64,2},K::Int)
    kmeans_result = Clustering.kmeans(X, K)
    codebook = kmeans_result.centers

  @assert size(codebook) == (size(X,1),K)
  return codebook::Array{Float64,2}
end


#---------------------------------------------------------
# Compute a histogram over the codebook for all given features
#---------------------------------------------------------
function computehistogram(features::FeatureList,codebook::Array{Float64,2},K::Int)
    
    # initialize histogram matrix
    H = zeros(K, length(features))
    
    # for each image
    for i in 1:length(features)
        # get current image
        img = features[i]
        # for each sift feature in the image
        for j in 1:size(img,2)
            feature = img[:,j]
            # compute which codeword the feature is closest to
            distances = sqrt.(sum((codebook .- feature).^2, dims=1))
            closest_codeword = argmin(distances)[2]
            # increase the counter for the closest codeword
            H[closest_codeword, i] += 1
        end
    end
    
    # normalize the histogram for each image
    H = H ./ sum(H, dims=1)

  @assert size(H) == (K,length(features))
  return H::Array{Float64,2}
end


#---------------------------------------------------------
# Visualize a feature matrix by projection to the first
# two principal components. Points get colored according to class labels y.
#---------------------------------------------------------
function visualizefeatures(X::Array{Float64,2}, y)

    # fit PCA
    M = fit(PCA, X, maxoutdim=2)
    
    Xt = transform(M, X)
    
    # plot in two colors
    PyPlot.figure()
    PyPlot.scatter(Xt[1,y.==0], Xt[2,y.==0])
    PyPlot.scatter(Xt[1,y.==1], Xt[2,y.==1])
    PyPlot.show()
    


  return nothing::Nothing
end


# Problem 1: Bag of Words Model: Codebook

function problem1()
  # make results reproducable
  Random.seed!(0)

  # parameters
  params = Parameters(15, 1.4, 1e-7, 10)
  K = 50

  # load trainging and testing data
  traininginputs,testinginputs = loadimages()

  # extract features from images
  trainingfeatures = extractfeatures(traininginputs.images, params)
  testingfeatures = extractfeatures(testinginputs.images, params)

  # construct feature matrix from the training features
  X = concatenatefeatures(trainingfeatures)

  # write codebook
  codebook = computecodebook(X,K)

  # compute histogram
  traininghistogram = computehistogram(trainingfeatures,codebook,K)
  testinghistogram = computehistogram(testingfeatures,codebook,K)

  # # visualize training features
  visualizefeatures(traininghistogram, traininginputs.labels)

  return nothing::Nothing
end
