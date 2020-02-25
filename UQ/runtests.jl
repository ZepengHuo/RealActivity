

if !isdefined(:DataAccess); include("../test/ExportData.jl"); end;
if !isdefined(:Main); include("../test/Main.jl");end;


using Main
using ExportData


@test function()

(X_data,y_data,DatasetName) = get_Data()
(nSamples,nFeatures) = size(X_data);
nFold = 10;  
split = collect(1:nSamples÷nFold:nSamples+1)  
doAccuracy = true  
doLogScore = false  
doWrite = false  
MaxIter = 3000  

main_param = InitialParameters();
main_param["nFeatures"] = nFeatures; 
main_param["nSamples"] = nSamples
main_param["thresh"] = 1e-5  
main_param["Kernel"] = "rbf"
main_param["theta"] = 5.0 
main_param["Verbose"] = false;


UQParam = InitialParameters(main_param=main_param)
Model = TestingModel("MXE",DatasetName,"Prediction","UQb" ,UQParam)

writing_order = Array{String,1}();                    
 
if doAccuracy;
  push!(writing_order,"accuracy"); 
end;  
 
if doLogScore;
  push!(writing_order,"logscore");
end;
 
if doAccuracy;   
  Model.Results["accuracy"]   = Array{Float64,1}();
end;

if doLogScore;
  Model.Results["logscore"]   = Array{Float64,1}();
end;

for i in 1:nFold 

  X_test = X_data[split[i]:(split[i+1])-1,:]
  y_test = y_data[split[i]:(split[i+1])-1]
  X = X_data[vcat(collect(1:split[i]-1),collect(split[i+1]:nSamples)),:]
  y = y_data[vcat(collect(1:split[i]-1),collect(split[i+1]:nSamples))]
  CreateModel(Model,X,y)
  time = TrainModel(Model,MaxIter)
  RunTests(Model,X_test,y_test,accuracy=doAccuracy,logscore=doLogScore)
end

Process(Model,writing_order) 
if doWrite
  upper_fold = "data";
  if !isdir(upper_fold); mkdir(upper_fold); end;
  WriteRes(Model,upper_fold)
end

return true
end 