 

module ExportData

export get_Data

function get_Data()
    data = readdlm("../data/Processed.data",',')
    X = convert(Array{Float64,2},data[:,1:end-1])
    y = convert(Array{Float64,1},data[:,end])
    nSamples = size(X,1)
    shuffling = true
    DatasetName = "sensor"
    Z = hcat(X,y)
    if shuffling
      Z = Z[shuffle(collect(1:nSamples)),:]  
    end
    (Z[:,1:end-1],Z[:,end], DatasetName)
end
 

end  