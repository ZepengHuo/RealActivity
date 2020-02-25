 

module Main

using UQb
using Distributions
using KernelTypes

export TestingModel
export InitialParameters 
export CreateModel, TrainModel, RunTests, Process , WriteRes
export Prediction, PredictionAccuracy

 
type TestingModel
  MethodName::String  
  DatasetName::String  
  ExperimentType::String  
  MethodType:: String 
  Param::Dict{String,Any}  
  Results::Dict{String,Any} 
  Model::Any
  TestingModel(methname,dataset,exp,methtype) = new(methname,dataset,exp,methtype,Dict{String,Any}(),Dict{String,Any}())
  TestingModel(methname,dataset,exp,methtype,params) = new(methname,dataset,exp,methtype,params,Dict{String,Any}())
end

 
function InitialParameters()
  param = Dict{String,Any}()
  param["thresh"]= 1e-8  
  param["epoch"] = 500
  param["Kernel"] = "rbf" 
  param["theta"] = 1.0  
  param["tuning_param"] = 1.0  
  param["Verbose"] = 0  
  return param
end

 
function CreateModel(tm::TestingModel,X_training,y_training)  
  tm.Model = Train(X_training,y_training; kernel=0,tuning_param = tm.Param["tuning_param"], epoch = tm.Param["epoch"],thresh =tm.Param["thresh"]  ,theta=tm.Param["theta"],verbose=false)
end

function TrainModel(tm::TestingModel,iterations)
  training_time = 0;
  tm.Model.nEpochs = iterations
  training_time = @elapsed tm.Model.Train()
  return training_time;
end


function RunTests(tm::TestingModel,X_test,y_test;accuracy::Bool=false,logscore::Bool=false)
  if accuracy
    push!(tm.Results["accuracy"],TestAccuracy(y_test,Prediction(tm,X_test)))
  end
  y_hat_acc = 0
  if logscore
    if y_hat_acc == 0
      y_hat_acc = PredictionAccuracy(tm::TestingModel,X_test)
    end
    push!(tm.Results["logscore"],LogScore(y_test,y_hat_acc))
  end
end


function Process(tm::TestingModel,writing_order)
  all_results = Array{Float64,1}()
  names = Array{String,1}()
  for name in writing_order
    result = [mean(tm.Results[name]), std(tm.Results[name])]
    all_results = vcat(all_results,result)
    names = vcat(names,name)
  end
  tm.Results["names"] = names
end
 

function WriteRes(tm::TestingModel,location)
  fold = String(location*"/"*tm.ExperimentType*"Experiment_"*tm.DatasetName*"Dataset")
  if !isdir(fold); mkdir(fold); end;
  writedlm(String(fold*"/Results_"*tm.MethodName*".txt"),tm.Results["allresults"])
end

 
function Prediction(tm::TestingModel, X_test)
  return sign.(tm.Model.Predict(X_test))
end

 
function PredictionAccuracy(tm::TestingModel, X_test)
  return tm.Model.PredictProba(X_test)
end

 
function TestAccuracy(y_test, y_hat)
  return 1-sum(1-y_test.*y_hat)/(2*length(y_test))
end
 
 
function LogScore(y_test, y_hat)
  return sum((y_test+1)./2.*log(y_hat)+(1-(y_test+1)./2).*log(1-y_hat))/length(y_test)
end
 
function ROC(y_test,y_hat)
    nt = length(y_test)
    truepositive = zeros(npoints); falsepositive = zeros(npoints)
    truenegative = zeros(npoints); falsenegative = zeros(npoints)
    thresh = collect(linspace(0,1,npoints))
    for i in 1:npoints
      for j in 1:nt
        truepositive[i] += (yp[j]>=thresh[i] && y_test[j]>=0.9) ? 1 : 0;
        truenegative[i] += (yp[j]<=thresh[i] && y_test[j]<=-0.9) ? 1 : 0;
        falsepositive[i] += (yp[j]>=thresh[i] && y_test[j]<=-0.9) ? 1 : 0;
        falsenegative[i] += (yp[j]<=thresh[i] && y_test[j]>=0.9) ? 1 : 0;
      end
    end
    return (truepositive./(truepositive+falsenegative),falsepositive./(truenegative+falsepositive))
end

end  