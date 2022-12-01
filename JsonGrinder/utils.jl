function calculate_accuracy(model,x, y,labelnames)
    vals = tmap(x) do s
        Flux.onecold(softmax(model(s)), labelnames)[1]
    end
    mean(vals .== y)
end

function read_data(df_labels,PATH_BEN_REPORTS,PATH_MAL_REPORTS)
    jsons = map(df_labels.name,df_labels.label) do n,y
        try
            #=if y==1
                path=PATH_MAL_REPORTS
            elseif y==0
                path=PATH_BEN_REPORTS
            end =#
            x=open(JSON.parse, "..\\data\\$(n).json")
            #x=Dict("behavior" => Dict("apistats_opt" => x["behavior"]["apistats_opt"], "summary" => Dict("dll_loaded" => x["behavior"]["summary"]["dll_loaded"]) ))
            #delete!(x["behavior"],"apistats")
            #delete!(x["behavior"],"apistats_opt")
            x=x
        catch e
            @error "Error when processing sha $n: $e"
        end
    end ;
    return jsons
end

function random_split(n_samples,tr_frac)
    idx = shuffle(collect(1:n_samples))
    tr_frac = 0.8
    train_indexes = idx[1:round(Int, tr_frac * n_samples)]
    test_indexes = setdiff(idx, train_indexes)

    return train_indexes,test_indexes
end

function time_split(n_samples,date)
    # Time-based split
    timesplit = Date(date)
    train_indexes = findall(i -> df_labels.date[i] < timesplit, 1:n_samples)
    test_indexes = [setdiff(Set(1:n_samples), Set(train_indexes))...] ;

    return train_indexes,test_indexes
end


function select_schema(jsons,train_indexes,THREADS)
    train_size = length(train_indexes)
    chunks = Iterators.partition(train_indexes, div(train_size, THREADS))
    sch_parts = tmap(chunks) do ch
        JsonGrinder.schema(jsons[ch])
    end
    complete_schema = merge(sch_parts...)
    printtree(complete_schema)

    extractor = suggestextractor(complete_schema)
    data = map(extractor, jsons)
    return data,complete_schema,extractor
end
