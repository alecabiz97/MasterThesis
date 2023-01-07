include("./utils.jl")

using Pkg
using Flux,
    MLDataPattern,
    Mill,
    JsonGrinder,
    JSON,
    Statistics,
    IterTools,
    StatsBase,
    ThreadTools
using JsonGrinder: suggestextractor, ExtractDict
using Mill: reflectinmodel
using CSV, DataFrames
using Random
using Dates
using Plots
using Printf
using ROC: roc, AUC
using Distributions: quantile, Normal
using BSON: @save, @load
Random.seed!(1234)
THREADS = Threads.nthreads()

PATH_TO_REDUCED_REPORTS = "../data/Avast/public_small_reports/"
PATH_TO_LABELS = "../data/Avast/subset_100.csv" ;
minibatchsize = 50
iterations = 200
epochs=5
split_choose="time" # "time" or "random"
training=false # If true training the models, if false load the trained model

df_labels=CSV.read(PATH_TO_LABELS,DataFrame);
targets=df_labels.classification_family;
labels=Set(df_labels.classification_family);
n_classes=length(labels);

jsons = tmap(df_labels.sha256) do s
    try 
        x=open(JSON.parse, "$(PATH_TO_REDUCED_REPORTS)$(s).json")
        delete!(x,"static") # Take only the behavioral info
        #delete!(x,"behavior") # Take only the static info
    catch e
        @error "Error when processing sha $s: $e"
    end
end ;

n_samples=length(jsons)
println("N samples: $(n_samples)")
println("N classes: $(n_classes)")
    
@assert size(jsons, 1) == length(targets)

if split_choose =="time"
    train_indexes,test_indexes=time_split(n_samples,"2019-07-01")
elseif split_choose =="random"
    train_indexes,test_indexes=random_split(n_samples,tr_frac -> 0.8)
end

train_size = length(train_indexes)
test_size = length(test_indexes)
println("Train size: $(train_size)")
println("Test size: $(test_size)")

# Select feature
reg_keys=map(jsons) do j  x = Dict("keys" => j["behavior"]["summary"]["keys"]) end
api=map(jsons) do j  x = Dict("resolved_apis" => j["behavior"]["summary"]["resolved_apis"]) end
executed_commands=map(jsons) do j  x = Dict("executed_commands" => j["behavior"]["summary"]["executed_commands"]) end
write_keys=map(jsons) do j  x = Dict("write_keys" => j["behavior"]["summary"]["write_keys"]) end
files=map(jsons) do j  x = Dict("files" => j["behavior"]["summary"]["files"]) end
read_files=map(jsons) do j  x = Dict("read_files" => j["behavior"]["summary"]["read_files"]) end
write_files=map(jsons) do j  x = Dict("write_files" => j["behavior"]["summary"]["write_files"]) end
delete_keys=map(jsons) do j  x = Dict("delete_keys" => j["behavior"]["summary"]["delete_keys"]) end
read_keys=map(jsons) do j  x = Dict("read_keys" => j["behavior"]["summary"]["read_keys"]) end
delete_files=map(jsons) do j  x = Dict("delete_files" => j["behavior"]["summary"]["delete_files"]) end
mutexes=map(jsons) do j  x = Dict("mutexes" => j["behavior"]["summary"]["mutexes"]) end

behavior=map(jsons) do x x=x end

#features = [reg_keys,api,executed_commands,write_keys,files,read_files,write_files,delete_keys,read_keys,delete_files,mutexes]
#features_names = ["reg_keys","api","executed_commands","write_keys","files","read_files","write_files","delete_keys","read_keys","delete_files","mutexes"]

features=[behavior]
features_names=["all"]

test_acc = []
train_acc = []
for (jsons, name) in zip(features, features_names)

    data,complete_schema,extractor=select_schema(jsons,train_indexes,THREADS)
  
    labelnames = sort(unique(df_labels.classification_family))
    neurons = 32
    model = reflectinmodel(
        complete_schema,
        extractor,
        k -> Dense(k, neurons, relu),
        d -> SegmentedMeanMax(d),
        fsm = Dict("" => k -> Dense(k, n_classes)),
    )

    if training
        eval_trainset = shuffle(train_indexes)
        eval_testset = shuffle(test_indexes)

        ps = Flux.params(model)
        loss = (x,y) -> Flux.logitcrossentropy(model(x), y)
        opt = ADAM()

        function minibatch()
            idx = StatsBase.sample(train_indexes, minibatchsize, replace = false)
            reduce(catobs, data[idx]), Flux.onehotbatch(df_labels.classification_family[idx], labelnames)
        end

        cb = () -> begin
            train_acc = calculate_accuracy(model,data[eval_trainset], df_labels.classification_family[eval_trainset],labelnames)
            test_acc = calculate_accuracy(model,data[eval_testset], df_labels.classification_family[eval_testset],labelnames)
            println("accuracy: train = $train_acc, test = $test_acc")
        end
        
        #=
        function calculate_accuracy(x,y) 
            vals = tmap(x) do s
                Flux.onecold(softmax(model(s)), labelnames)[1]
            end
            mean(vals .== y)
        end     
         =#   
        #printtree(model)

        for i in 1:epochs
            println("Epoch $(i)")
            Flux.Optimise.train!(loss, ps, repeatedly(minibatch, iterations), opt, cb = Flux.throttle(cb, 2))
        end

        # Save model
        @save "./trained_models/Avast/JsonGrinder_classification_$(name)_$(split_choose).bson" model
    else
        # Load model
        @load "./trained_models/Avast/JsonGrinder_classification_$(name)_$(split_choose).bson" model
    end    

    full_train_accuracy = calculate_accuracy(model,data[train_indexes], df_labels.classification_family[train_indexes],labelnames)
    full_test_accuracy = calculate_accuracy(model,data[test_indexes], df_labels.classification_family[test_indexes],labelnames)
    append!(train_acc,full_train_accuracy)
    append!(test_acc,full_test_accuracy)
    
    println("Final evaluation:")
    println("Accuratcy on train data: $(full_train_accuracy)")
    println("Accuratcy on test data: $(full_test_accuracy)") 
    

    # Save data for confusion matrix
    #=
    y_true=String[]
    y_pred=String[]
    for true_label in labelnames
        family_indexes = filter(i -> df_labels.classification_family[i] == true_label, test_indexes)
        predictions = tmap(data[family_indexes]) do s
            Flux.onecold(softmax(model(s)), labelnames)[1]
        end

        tmp1=[true_label for i in 1:length(predictions)]

        append!(y_true,tmp1)
        append!(y_pred,predictions)
    end 

    d=Dict("y_true" => y_true, "y_pred" => y_pred )
    json_string = JSON.json(d)

    open("jsongrinder_confmat_all_$(split_choose).json","w") do f 
        write(f, json_string) 
    end
    =#

    # Confusion matrix
    test_predictions = Dict()
    for true_label in labelnames
        current_predictions = Dict()
        [current_predictions[pl]=0.0 for pl in labelnames]
        family_indexes = filter(i -> df_labels.classification_family[i] == true_label, test_indexes)
        predictions = tmap(data[family_indexes]) do s
            Flux.onecold(softmax(model(s)), labelnames)[1]
        end
        [current_predictions[pl] += 1.0 for pl in predictions]
        [current_predictions[pl] = current_predictions[pl] ./ length(predictions) for pl in labelnames]
        test_predictions[true_label] = current_predictions
    end

    @printf "%8s\t" "TL\\PL"
    [@printf " %8s" s for s in labelnames]
    print("\n")
    for tl in labelnames
        @printf "%8s\t" tl 
        for pl in labelnames
            @printf "%9s" @sprintf "%.2f" test_predictions[tl][pl]*100
        end
        print("\n")
    end
    

end

println("\nFinal evaluation")
for i in 1:length(features_names)
    println("$(features_names[i])")
    println("  Train acc: $(train_acc[i])")
    println("  Test acc: $(test_acc[i])")
end