 

module KernelTypes

export Kernel
export KernelMatrix, DiagKernelMatrix
export delta_k  


type Kernel
    kfunction::Function  
    kernel_compute::Function  
    kernel_compute_grad::Function     
    kenel_grad::Function  
    param::Float64  
    weight::Float64 
    Npar::Int64  
 

    function Kernel(kernel, weight::Float64; params=0)
      this = new()
      this.weight = weight
      this.Npar = 1
      if kernel=="rbf"
        this.kfunction = rbf
        this.kenel_grad = rbf_grad
        this.param = params 
      elseif kernel=="linear"
        this.kfunction = linear
        this.Npar = 0
      end

      if this.Npar > 0
        this.kernel_compute = function(X1,X2)
            this.kfunction(X1,X2,this.param)
          end
        this.kernel_compute_grad = function(X1,X2)
            this.kenel_grad(X1,X2,this.param)
          end
      else
        this.kernel_compute = function(X1,X2)
            this.kfunction(X1,X2)
          end
      end
      return this
    end
end

function KernelMatrix(X1,kfunction;X2=0)  
  if X2 == 0
    kernel_size = size(X1,1)
    K = zeros(kernel_size,kernel_size)
    for i in 1:kernel_size
      for j in 1:i
        K[i,j] = kfunction(X1[i,:],X1[j,:])
        if i != j
          K[j,i] = K[i,j]
        end
      end
    end
    return K
  else
    kernel_size1 = size(X1,1)
    kernel_size2 = size(X2,1)
    K = zeros(kernel_size1,kernel_size2)
    for i in 1:kernel_size1
      for j in 1:kernel_size2
        K[i,j] = kfunction(X1[i,:],X2[j,:])
      end
    end
    return K
  end
end


function DiagKernelMatrix(X,kfunction;MatrixFormat=false)
  n = size(X,1)
  kermatrix = zeros(n)
  for i in 1:n
    kermatrix[i] = kfunction(X[i,:],X[i,:])
  end
  if MatrixFormat
    return diagm(kermatrix)
  else
    return kermatrix
  end
end

function delta_k(X1::Array{Float64,1},X2::Array{Float64,1})
  return X1==X2 ? 1 : 0
end

 
function rbf(X1::Array{Float64,1},X2::Array{Float64,1},theta)
  if X1 == X2
    return 1
  end
  exp(-(norm(X1-X2))^2/(theta^2))
end


function rbf_grad(X1::Array{Float64,1},X2::Array{Float64,1},theta)
  a = norm(X1-X2)
  if a != 0
    return 2*a^2/(theta^3)*exp(-a^2/(theta^2))
  else
    return 0
  end
end

 
function linear(X1::Array{Float64,1},X2::Array{Float64,1})
  dot(X1,X2)
end

 

end 