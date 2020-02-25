if !isdefined(:KernelTypes); include("KernelTypes.jl"); end;

module UQb

using Distributions
using KernelTypes
using Distributions 
using PyPlot 
using StatsBase 
using GaussianMixtures 
using Clustering 
using ScikitLearn 
using PyCall 
export Train, PredicUQb, Uncertaintypredic 


function Train(X::Array{Float64,2}, y::Array{Float64,1}; kernel=0,tuning_param::Float64=1.0, epoch::Int64=100,thresh::Float64 = 1e-5,theta=[1.0],verbose=false)
 
    n = size(X,1)
    k = size(X,1)
    Y = Diagonal(y)
    f = randn(k)
    invlatentvariable = abs(rand(k))
    if kernel == 0
      kernel = Kernel("rbf",theta[1],params=theta[2])
    elseif typeof(kernel)== AbstractString
      if length(theta)>1
        kernel = Kernel(kernel,theta[1],params=theta[2])
      else
        kernel = Kernel(kernel,theta[1])
      end
    end

    K = KernelMatrix(X,kernel.kernel_compute)
    i = 1
    div = Inf;
    while i < epoch && div > thresh
        prev_latentvariable = 1./invlatentvariable; prev_f = f;
        invlatentvariable = sqrt(1.0+2.0/tuning_param)./(abs(1.0-y.*f))

        f = K*inv((K+1.0/tuning_param*diagm(1./invlatentvariable)))*Y*(1+1./invlatentvariable)
        div = norm(f-prev_f);
        i += 1
    end
    return (invlatentvariable,K,kernel,y,f)
end

function PredicUQb(X,y,X_test,invlatentvariable,K,tuning_param,kernel)
  n = size(X,1)
  n_t = size(X_test,1)
  predic = zeros(n_t)
  sig = inv(K+1/tuning_param*diagm(1./invlatentvariable))
  upper = sig*diagm(y)*(1+1./invlatentvariable)
  for i in 1:n_t
    k_optimal = zeros(n)
    for j in 1:n
      k_optimal[j] = kernel.kernel_compute(X[j,:],X_test[i,:])
    end
    predic[i] = dot(k_optimal,upper)
  end
  return predic
end

function Uncertaintypredic(X,y,X_test,invlatentvariable,K,tuning_param,kernel)
  n = size(X,1)
  n_t = size(X_test,1)
  predic = zeros(n_t)
  sig = inv(K+1/tuning_param*diagm(1./invlatentvariable))
  upper = sig*diagm(y)*(1+1./invlatentvariable)
  for i in 1:n_t
    k_optimal = zeros(n)
    for j in 1:n
      k_optimal[j] = kernel.kernel_compute(X[j,:],X_test[i,:])
    end
    k_optimalstar = kernel.kernel_compute(X_test[i,:],X_test[i,:])
    predic[i] = cdf(Normal(),dot(k_optimal,upper)/(1+k_optimalstar-dot(k_optimal,sig*k_optimal)))
  end
  return predic
end
end