import json
from utils import *
import csv
from tqdm import tqdm
from collections import Counter

# Creates json files in two folder (ben_preproc and mal_preproc) with the preprocessed data of dataset1.
# In each json file there are only the behavior and static info.
# In the behavior part there are:
#   -apistats (pid independent, merge of dictionary)
#   -apistats_opt (optimized)
#   -summary:
#       -regkey_opened
#       -regkey_read
#       -dll_loaded
#       -mutex

def merge_dict(dicts):
    new_d = {}
    for d in dicts:
        for k, v in d.items():
            if k in new_d.keys():
                new_d[k] += v
            else:
                new_d[k] = v
    return new_d


def opt_api(d):
    apis = list(d.keys())
    for i in range(len(apis)):
        api_name = apis[i]
        n = d[api_name]
        if n <= 10:
            apis[i] = f"Low_{api_name}"
        elif 10 < n <= 100:
            apis[i] = f"Med_{api_name}"
        elif 100 < n <= 1000:
            apis[i] = f"High_{api_name}"
        elif n > 1000:
            apis[i] = f"VeryHigh_{api_name}"
    return apis

if __name__ == '__main__':
    benign_root = 'data\\dataset1\\ben_reports'
    malign_root = 'data\\dataset1\\mal_reports'

    ben_files = getListOfFiles(benign_root)
    mal_files = getListOfFiles(malign_root)

    df = pd.DataFrame({'name': str(), 'label': int(), 'date': str()}, index=[])
    i = 0
    for filepath in tqdm(ben_files + mal_files):
        try:
            with open(filepath, 'r') as fp:
                data = json.load(fp)
            data_keys = data.keys()
            if "behavior" in data_keys:
                beh_keys = data["behavior"].keys()
                if "apistats" in beh_keys and "summary" in beh_keys:
                    pids = list(data["behavior"]["apistats"].keys())

                    # API
                    apis_stats = [data["behavior"]["apistats"][pids[i]] for i in range(len(pids))]
                    apis_stats = merge_dict(apis_stats)

                    sum_keys=data["behavior"]["summary"].keys()

                    # REG
                    regkey_read = data["behavior"]["summary"]["regkey_read"] if "regkey_read" in sum_keys else []
                    regkey_opened = data["behavior"]["summary"]["regkey_opened"] if "regkey_opened" in sum_keys else []
                    regkey_deleted = data["behavior"]["summary"]["regkey_deleted"] if "regkey_deleted" in sum_keys else []
                    regkey_written = data["behavior"]["summary"]["regkey_written"] if "regkey_written" in sum_keys else []

                    # FILES
                    file_deleted = data["behavior"]["summary"]["file_deleted"] if "file_deleted" in sum_keys else []
                    file_failed = data["behavior"]["summary"]["file_failed"] if "file_failed" in sum_keys else []
                    file_read = data["behavior"]["summary"]["file_read"] if "file_read" in sum_keys else []
                    file_opened = data["behavior"]["summary"]["file_opened"] if "file_opened" in sum_keys else []
                    file_exists = data["behavior"]["summary"]["file_exists"] if "file_exists" in sum_keys else []
                    file_written = data["behavior"]["summary"]["file_written"] if "file_written" in sum_keys else []
                    file_created = data["behavior"]["summary"]["file_created"] if "file_created" in sum_keys else []


                    # DLL
                    dll_loaded = data["behavior"]["summary"]["dll_loaded"] if "dll_loaded" in sum_keys else []

                    # MUTEX
                    mutex = data["behavior"]["summary"]["mutex"] if "mutex" in sum_keys else []


                    d = {"static":data["static"],
                        "behavior": {"apistats": list(apis_stats),
                                     "apistats_opt": opt_api(apis_stats),
                                     "regkey_read":regkey_written,
                                     "regkey_opened":regkey_opened,
                                     "regkey_deleted":regkey_deleted,
                                     "regkey_written":regkey_written,
                                     "file_deleted": file_deleted,
                                     "file_failed": file_failed,
                                     "file_read": file_read,
                                     "file_opened": file_opened,
                                     "file_exists": file_exists,
                                     "file_written": file_written,
                                     "file_created": file_created,
                                      "dll_loaded": dll_loaded,
                                      "mutex": mutex
                                     }
                         }

                    name = filepath.split("\\")[-1].split(".")[0]
                    label = 0 if filepath.split("\\")[-2] == 'ben_reports' else 1  # 0 -> benign , 1 -> malign
                    date = data['static']['pe_timestamp'].split(" ")[0]

                    p = "data\\dataset1\\ben_preproc\\" if label == 0 else "data\\dataset1\\mal_preproc\\"

                    json_object = json.dumps(d, indent=4)
                    with open(f"{p}{name}.json", "w") as outfile:
                        outfile.write(json_object)

                    df_tmp = pd.DataFrame({'name': f"{p}{name}",
                                           'label': label,
                                           'date': date},
                                          index=[i])
                    i += 1
                    df = pd.concat([df, df_tmp], ignore_index=True)

        except:
            pass

    n_ben = len(df.query("label == 0"))
    n_mal = len(df.query("label == 1"))
    s1 = f"Created {n_ben}/{len(ben_files)} ben files"
    s2 = f"Created {n_mal}/{len(mal_files)} mal files"

    print(s1)
    print(s2)
    # df.to_csv(f'data\\dataset1\\labels_preproc.csv', index=False)








