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

THREADS = Threads.nthreads()
PATH_BEN_REPORTS,PATH_MAL_REPORTS = "../dataset1/ben_preproc/","../dataset1/mal_preproc/"
PATH_TO_LABELS = "../dataset1/labels_preproc.csv";

df = CSV.read(PATH_TO_LABELS, DataFrame);

#df_labels_ben = filter("label" => x -> x == 0, df)[1:100, :]
#df_labels_mal = filter("label" => x -> x == 1, df)[1:100, :]
#df_labels = vcat(df_labels_ben, df_labels_mal)

df_labels=df

n_classes = length(Set(df_labels.label));

# Import data
jsons = read_data(df_labels, PATH_BEN_REPORTS, PATH_MAL_REPORTS)

n_samples = length(jsons)
println("N samples: $(n_samples)")
println("N labels: $(n_classes)")
@assert n_samples == length(df_labels.label)

split_choose="time" # 'time' or 'random'
if split_choose =="time"
    train_indexes,test_indexes=time_split(n_samples,"2013-08-09")
elseif split_choose =="random"
    train_indexes,test_indexes=random_split(n_samples,tr_frac -> 0.8)
end

train_size = length(train_indexes)
test_size = length(test_indexes)
println("Train size: $(train_size)")
println("Test size: $(test_size)")

# Select feature
#behavior = map(jsons) do j  x = Dict("behavior" => j["behavior"]) end
static= map(jsons) do j x = Dict("static" => j["static"]) end
api = map(jsons) do j  x = Dict("apistats" => j["behavior"]["apistats"]) end
api_opt = map(jsons) do j  x = Dict("apistats_opt" => j["behavior"]["apistats_opt"]) end
dll = map(jsons) do j  x = Dict("dll_loaded" => j["behavior"]["summary"]["dll_loaded"]) end
regop = map(jsons) do j  x = Dict("regkey_opened" => j["behavior"]["summary"]["regkey_opened"]) end
regre = map(jsons) do j x = Dict("regkey_read" => j["behavior"]["summary"]["regkey_read"]) end
mutex = map(jsons) do j x = Dict("mutex" => j["behavior"]["summary"]["mutex"]) end

api_opt_regre = map(jsons) do j  x = Dict("apistats_opt" => j["behavior"]["apistats_opt"],
                                        "regkey_read" => j["behavior"]["summary"]["regkey_read"]) end
api_regre = map(jsons) do j  x = Dict("apistats" => j["behavior"]["apistats"],
                                            "regkey_read" => j["behavior"]["summary"]["regkey_read"]) end

api_dll_regre = map(jsons) do j  x = Dict("apistats" => j["behavior"]["apistats"],
                                        "dll_loaded" => j["behavior"]["summary"]["dll_loaded"],
                                        "regkey_read" => j["behavior"]["summary"]["regkey_read"]) end

behavior=map(jsons) do j
    #delete!(j["behavior"],"apistats")
    j = j["behavior"]
end


x = [behavior,api,api_opt,regop,regre,dll,mutex]
y = ["All","API","API OPT","Regkey Opened","Regkey Read","DLL Loaded","Mutex"]

#x = [behavior,api,api_opt,dll]
#y = ["All","API","API OPT","DLL"]

#p = plot()
p1=plot(title="Receiver Operating Characteristic")
p2=plot(title="Detection Error Tradeoff")
colors=["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd","#8c564b","#e377c2"]
global j=1
for (jsons, name) in zip(x, y)

    # Select scheme
    data,complete_schema,extractor=select_schema(jsons,train_indexes,THREADS)
    printtree(data[1])

    # Create model
    labelnames = sort(unique(df_labels.label))
    neurons = 32
    model = reflectinmodel(
        complete_schema,
        extractor,
        k -> Dense(k, neurons, relu),
        d -> SegmentedMeanMax(d),
        fsm = Dict("" => k -> Dense(k, n_classes)),
    )

    minibatchsize = 50 # 50
    iterations = 200 # 200

    eval_trainset = shuffle(train_indexes)
    eval_testset = shuffle(test_indexes)

    ps = Flux.params(model)
    loss = (x, y) -> Flux.logitcrossentropy(model(x), y)
    opt = ADAM()

    function minibatch()
        idx = StatsBase.sample(train_indexes, minibatchsize, replace = false)
        reduce(catobs, data[idx]),  Flux.onehotbatch(df_labels.label[idx], labelnames)
    end

    function calculate_accuracy(x, y)
        vals = tmap(x) do s
            Flux.onecold(softmax(model(s)), labelnames)[1]
        end
        mean(vals .== y)
    end

    cb = () -> begin
            train_acc = calculate_accuracy(data[eval_trainset],df_labels.label[eval_trainset])
            test_acc = calculate_accuracy(data[eval_testset],df_labels.label[eval_testset])
            println("accuracy: train = $train_acc, test = $test_acc")
        end

    epochs = 5 # 5
    for i = 1:epochs
        println("Epoch $(i)")
        Flux.Optimise.train!(loss,ps,repeatedly(minibatch, iterations),opt,cb = Flux.throttle(cb, 2))
    end

    full_train_accuracy = calculate_accuracy(data[train_indexes], df_labels.label[train_indexes])
    full_test_accuracy = calculate_accuracy(data[test_indexes], df_labels.label[test_indexes])
    println("Final evaluation:")
    println("Accuratcy on train data: $(full_train_accuracy)")
    println("Accuratcy on test data: $(full_test_accuracy)")

    scores = softmax(model(data[test_indexes]))[2, :]
    roc_curve = roc(scores, df_labels.label[test_indexes], true)

    auc=AUC(roc_curve)
    println("AUC: $auc")
    #plot!(p, roc_curve, lw = 3, label = "$name -- AUC: $(round(auc,digits = 3))")
    
    # ROC
    plot!(p1,roc_curve,label = "$name (AUC: $(round(auc,digits = 2)))",color=colors[j], legend=:bottomright)
    xaxis!(widen=true)
    yaxis!(widen=true)
    xlabel!("False Positive Rate (Positive label: 1)")
    ylabel!("True Positive Rate (Positive label: 1)")
    
    # DET
    fpr=roc_curve.FPR;
    tpr=roc_curve.TPR;
    fnr= 1. .- tpr;

    ticks = [0.001, 0.01, 0.05, 0.20, 0.5, 0.80, 0.95, 0.99, 0.999]
    tick_locations = quantile(Normal(0.0, 1.0),ticks)
    tick_labels=["$(trunc(Int,100*i))%" for i in ticks]
    tick_labels[1]=""; tick_labels[end]="";

    plot!(p2,quantile(Normal(0.0, 1.0),fpr),quantile(Normal(0.0, 1.0),fnr),label = "$name",color=colors[j], legend=:topright)
    xaxis!(ticks=(tick_locations,tick_labels),lims=(-3,3))
    yaxis!(ticks=(tick_locations,tick_labels),lims=(-3,3))
    xlabel!("False Positive Rate (Positive label: 1)")
    ylabel!("False Negative Rate (Positive label: 1)")
    j += 1
end
#display(p)
p3=plot(p1,p2,layout=(1,2),size=(1100,400),dpi=400,lw=2,
    left_margin = 5Plots.mm, right_margin = 5Plots.mm, up_margin=5Plots.mm,  bottom_margin=5Plots.mm
)
display(p3)
savefig("jsonGrinder_$split_choose.pdf")