This is a discriminative feature-learning model, in which the labels are utilized within a max-margin classifier.
The max-margin idea, which is closely related to support vector machines (SVMs)
The objective of such model is to: minimizing a discriminant function by estimating the posterior of the vector of coefficients, given the prior distribution for the vector of coefficients is known see [1] and choosing coefficient that maximize the log of likelihood function see [1].

In this work we set this prior distribution a Gaussian distribution to satisfy the integrality condition of parameter Z (the normalization parameter) see [1-3].

We now assume that the decision function is drawn from a zero-mean Gaussian process with some kernel function. In the previous work we used a linear kernel, but you can check whether other choices would help or not. 

In the first document I used  conjugate gradient modelthod is going to minimize l_1elihood of the training, I followed this paper [1] and solved it through EM, again the optimization algorithm is not important since it is just used to scale the process up. The maxmargin idea is the main issue. Please check [1] for the estimation process. 


Code:

This program containing Julia language code:
if you want to run the algorithm in Julia,do the follwoing: 
# brew cask install julia
# or 
# sudo ln -s /Applications/Julia-1.1.app/Contents/Resources/julia/bin/julia /usr/local/bin/julia
import julia
# after downloading Julia you need to install packages in Julia
# Don't run this in .py
# run it in the julia terminal 
# @ julia
import Pkg
Pkg.add("Distributions")
Pkg.add("PyPlot")
Pkg.add("StatsBase")
Pkg.add("GaussianMixtures")
Pkg.add("Clustering")
Pkg.add("ScikitLearn")
Pkg.add("PyCall")
# cd(Pkg.dir("PyCall")) check paclage path in Julia
# homedir() to check homedirectory
# cd("$(homedir())/Desktop/UQ")
#pwd()

In python, you have to import Tensorflow and GPflow.  Tensorflow and GPflow, they must be included in the search path of PyCall:

push!(pyimport("sys")["path"], pwd())
unshift!(PyVector(pyimport("sys")["path"]), pwd())
Pkg.build("PyCall")

# pip install gpflow==1.0 #if you use python 2
# check your tensorflow version: tensorflow.__version__
#import gpflow as gp #  pip install gpflow==1.0

#import os.path
#print(os.path.abspath(tensorflow.__file__))

UQb.jl = has the main code
KernelTypes = is the type of decision function, if you choose linear its a simple f(x): w^Tx, you also have the option to userbf too
ExportData = is generating the dataset compatible with the code
Main = contains the running of the model
runtest = choose the dataset, change the path, parameters, and run the file:

j = julia.Julia()
j.include("runtest.jl") 

 
Model = Train(X_training,y_training; kernel=0,tuning_param = tm.Param["tuning_param"], epoch = tm.Param["epoch"],thresh =tm.Param["thresh"]  ,theta=tm.Param["theta"],verbose=false)
yhat  = sign(Model.Predict(X_test))
y_uncertain= Model.PredictProb(X_test)


Dataset:
once you found the processed data, please make sure the format of your data is in such a way that your training data has m samples and d features and your labels are in the form of 1,-1. for multiclass, please use OVA. Leave the last column for the labels -- Processed.data is the imput to the 


[1] Jaakkola, Tommi, Marina Meila, and Tony Jebara. "Maximum entropy discrimination." Advances in neural information processing systems. 2000.
[2] Uncertainty Quantification for Deep Context-Aware Mobile Activity Recognition and Unknown Context Discovery
[3] UQ-CHI: An Uncertainty Quantification-Based Contemporaneous Health Index for Degenerative Disease Monitoring
