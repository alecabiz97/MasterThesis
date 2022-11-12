# %%
import json
import pickle
import numpy as np
import pandas as pd
import time
from utils import *
from tqdm import tqdm
from transformers import DistilBertTokenizer,DistilBertForSequenceClassification,DistilBertTokenizerFast
from DatasetClasses import CustomDataset
from torch.nn.functional import softmax
import copy
import csv

# Creates two .csv (Avast and dataset1) files with 2 columns labels: label and text. They are used in neurlux.py

# %% DATASET1
meta = pd.read_csv("dataset1\\labels_preproc.csv")
df=pd.DataFrame({"label":int(),
                 "date":str(),
                "text":str()},index=[])
for i,(filepath,label,date) in enumerate(tqdm(meta[['name','label','date']].values)):
    with open(f"{filepath}.json", 'r') as fp:
        data = json.load(fp)

    text = preprocessing_data(str(data["behavior"]))

    df_tmp = pd.DataFrame({'label':label,
                           'date':date,
                           'text': text}, index=[i])
    df = pd.concat([df, df_tmp], ignore_index=True)

print(len(df))
df.to_csv('data.csv',index=False)


# %% AVAST DATASET
# meta = pd.read_csv("Avast\\public_labels.csv")
meta = pd.read_csv("Avast\\subset_100.csv")
root='Avast\\public_small_reports'
classes = ['Adload','Emotet', 'HarHar', 'Lokibot','njRAT','Qakbot','Swisyn','Trickbot','Ursnif','Zeus']
df=pd.DataFrame({"label":int(),
                "date":str(),
                "text":str()},index=[])
for i,(sha,family,date) in enumerate(tqdm(meta[["sha256","classification_family","date"]].values)):
    filepath=f"{root}\\{sha}.json"
    try:
        with open(filepath, 'r') as fp:
            data = json.load(fp)

        text = preprocessing_data(str(data["behavior"]))
        #text=" ".join(data["behavior"]["apistats"])

        y = classes.index(family)

        df_tmp = pd.DataFrame({'label': y,
                               'date': date,
                               'text': text}, index=[i])
        df = pd.concat([df, df_tmp], ignore_index=True)
        pass
    except:
        pass

print(len(df))
df.to_csv('data_avast_100.csv',index=False)

















