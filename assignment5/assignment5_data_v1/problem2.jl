using Images
using PyPlot
using FileIO
using Optim
using Random
using Statistics

include("Common.jl")

#---------------------------------------------------------
# Load features and labels from file.
#---------------------------------------------------------
function loaddata(path::String)

  @assert length(labels) == size(features,1)
  return features::Array{Float64,2}, labels::Array{Float64,1}
end

#---------------------------------------------------------
# Show a 2-dimensional plot for the given features with
# different colors according to the labels.
#---------------------------------------------------------
function showbefore(features::Array{Float64,2},labels::Array{Float64,1})

  return nothing::Nothing
end


#---------------------------------------------------------
# Show a 2-dimensional plot for the given features along
# with the decision boundary.
#---------------------------------------------------------
function showafter(features::Array{Float64,2},labels::Array{Float64,1},Ws::Vector{Any}, bs::Vector{Any})

  return nothing::Nothing
end


#---------------------------------------------------------
# Implements the sigmoid function.
#---------------------------------------------------------
function sigmoid(z)

  return s
end


#---------------------------------------------------------
# Implements the derivative of the sigmoid function.
#---------------------------------------------------------
function dsigmoid_dz(z)

  return ds
end


#---------------------------------------------------------
# Evaluates the loss function of the MLP.
#---------------------------------------------------------
function nnloss(theta::Array{Float64,1}, X::Array{Float64,2}, y::Array{Float64,1}, netdefinition::Array{Int, 1})

  return loss::Float64
end


#---------------------------------------------------------
# Evaluate the gradient of the MLP loss w.r.t. Ws and Bs
# The gradient should be stored in the vector 'storage'
#---------------------------------------------------------
function nnlossgrad(storage::Array{Float64,1}, theta::Array{Float64,1}, X::Array{Float64,2}, y::Array{Float64,1}, netdefinition::Array{Int, 1})

  return storage::Array{Float64,1}
end


#---------------------------------------------------------
# Use LBFGS to optimize the MLP loss
#---------------------------------------------------------
function train(trainfeatures::Array{Float64,2}, trainlabels::Array{Float64,1}, netdefinition::Array{Int, 1})

  return Ws::Vector{Any},bs::Vector{Any}
end


#---------------------------------------------------------
# Predict the classes of the given data points using Ws and Bs.
# p, N x 1 array of Array{Float,2}, contains the output class scores (continuous value) for each input feature.
# c, N x 1 array of Array{Float,2}, contains the output class label (either 0 or 1) for each input feature.
#---------------------------------------------------------
function predict(X::Array{Float64,2}, Ws::Vector{Any}, bs::Vector{Any})

  return p::Array{Float64,2}, c::Array{Float64,2}
end


#---------------------------------------------------------
# A helper function which concatenates weights and biases into a variable theta
#---------------------------------------------------------
function weightsToTheta(Ws::Vector{Any}, bs::Vector{Any})

  return theta::Vector{Float64}
end


#---------------------------------------------------------
# A helper function which decomposes and reshapes weights and biases from the variable theta
#---------------------------------------------------------
function thetaToWeights(theta::Vector{Float64}, netdefinition::Array{Int,1})

  return Ws::Vector{Any}, bs::Vector{Any}
end


#---------------------------------------------------------
# Initialize weights and biases from Gaussian distributions
#---------------------------------------------------------
function initWeights(netdefinition::Array{Int,1}, sigmaW::Float64, sigmaB::Float64)

  return Ws::Vector{Any}, bs::Vector{Any}
end


# Problem 2: Multilayer Perceptron

function problem2()
  # make results reproducable
  Random.seed!(10)

  # LINEAR SEPARABLE DATA
  # load data
  features,labels = loaddata("separable.jld2")

  # show data points
  showbefore(features,labels)
  title("Data for Separable Case")

  # train MLP
  Ws,bs = train(features,labels, [2,4,1])

  # show optimum and plot decision boundary
  showafter(features,labels,Ws,bs)
  title("Learned Decision Boundary for Separable Case")


  ## LINEAR NON-SEPARABLE DATA
  # load data
  features2,labels2 = loaddata("nonseparable.jld2")

  # show data points
  showbefore(features2,labels2)
  title("Data for Non-Separable Case")

  # train MLP
  Ws,bs = train(features2,labels2, [2,4,1])

  # show optimum and plot decision boundary
  showafter(features2,labels2,Ws, bs)
  title("Learned Decision Boundary for Non-Separable Case")

  # PLANE-BIKE-CLASSIFICATION FROM PROBLEM 2
  # load data
  trainfeatures,trainlabels = loaddata("imgstrain.jld2")
  testfeatures,testlabels = loaddata("imgstest.jld2")

  # train MLP and predict classes
  Ws,bs = train(trainfeatures,trainlabels, [50,40,30,1])
  _,trainpredictions = predict(trainfeatures, Ws, bs)
  _,testpredictions = predict(testfeatures, Ws, bs)

  # show error
  trainerror = sum(trainpredictions.!=trainlabels)/length(trainlabels)
  testerror = sum(testpredictions.!=testlabels)/length(testlabels)
  println("Training Error Rate: $(round(100*trainerror,digits=2))%")
  println("Testing Error Rate: $(round(100*testerror,digits=2))%")

  return
end
