import json
import os
from utils import *
import pandas as pd
import numpy as np
from tqdm import tqdm
import csv
from collections import Counter
from transformers import DistilBertTokenizerFast,AutoTokenizer
import re

def create_avast_dataframe(files_path,labels_file,n_sample=None,feature=None):
    avast_files = getListOfFiles(files_path)
    if n_sample:
        avast_files = avast_files[0:n_sample]

    f = open(labels_file, 'r')
    data = np.array(list(csv.reader(f, delimiter=',')))

    df = pd.DataFrame({'sha256': str(),
                       'classification_family': str(),
                       'classification_type': str(),
                       'date': str(),
                       'words': str()},
                        index=[])

    print('Creating Avast dataframe ...')
    for i, filename in enumerate(tqdm(avast_files)):
        try:
            f = open(filename, 'r')
            d = json.load(f)

            hash = filename.split('\\')[1].split('.')[0]
            index = data[:, 0] == hash
            hash, family, class_type, date = data[index][0]

            if feature:
                words = str(d['behavior']['summary'][feature])
            else:
                words=str(d['behavior']['summary'])
            words = preprocessing_data(words) # Delete special characters

            df_tmp = pd.DataFrame({'sha256': hash,
                                   'classification_family': family,
                                   'classification_type': class_type,
                                   'date': date,
                                   'words': words},
                                  index=[i])

            df = pd.concat([df, df_tmp], ignore_index=True)
        except:
            pass

    # print(len(df))
    print('Avast Dataframe created')
    return df


if __name__ == '__main__':

    avast_root = 'Avast/public_small_reports'
    labels_file = 'Avast/public_labels.csv'
    
    df_path='Avast/avast_dataframe.pkl'
    df_train_path='Avast/avast_dataframe_train.pkl'
    df_test_path='Avast/avast_dataframe_test.pkl'
    vocab_file='vocab_avast.txt'

    feat='resolved_apis'

    df=create_avast_dataframe(avast_root,labels_file,feature=None) # Create dataframe
    df.to_pickle(df_path)
    df = pd.read_pickle(df_path)
    df_tr, df_ts = split_train_test(df,train_test_date='2019-08-01') # Split train and test
    df_tr.to_pickle(df_train_path)
    df_ts.to_pickle(df_test_path)

    df_train = pd.read_pickle(df_train_path)
    df_ts = pd.read_pickle(df_test_path)


    # Create vocab file
    all_words=[]
    x = df_train['words'].values
    for i in range(len(df_train)):
        all_words.extend(x[i].split(' '))
    print('Creating the vocab file ...')
    create_vocab_file(vocab_file,all_words,n_most_common=10000)

    # Read vocab file
    with open(vocab_file, 'r') as fp:
        vocab = fp.read()
        vocab = vocab.split('\n')

    # Try Tokenizer
    example = preprocessing_data("HKEY_CURRENT_USER\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Run")
    # example = preprocessing_data("kernel32.dll.GetProcAddress")
    print(example)
    # tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    x = tokenizer(example)
    print(len(tokenizer.convert_ids_to_tokens(x['input_ids'])),tokenizer.convert_ids_to_tokens(x['input_ids']))

    tokenizer = tokenizer.train_new_from_iterator(vocab, vocab_size=20000)
    x = tokenizer(example)
    print(len(tokenizer.convert_ids_to_tokens(x['input_ids'])),tokenizer.convert_ids_to_tokens(x['input_ids']))


    # benign_root = 'dataset1/ben_reports'
    # malign_root = 'dataset1/mal_reports'

    # ben_files = getListOfFiles(benign_root)
    # mal_files = getListOfFiles(malign_root)

    # new_ben='dataset1/ben_reports_cleaned_api'
    # new_mal='dataset1/mal_reports_cleaned_api'

    # Data cleaning
    #create_cleaned_files(ben_files,new_ben)
    #create_cleaned_files(mal_files,new_mal)

    # Data cleaning api
    #print('Creating api call files ...')
    # create_cleaned_files_api(ben_files,new_ben)
    # create_cleaned_files_api(mal_files,new_mal)

    # print('Creating dataframe ...')
    # df = create_dataframe(benign_root, malign_root)
    # print('Dataframe created.')
    # df.to_pickle('dataset1/dataframe.pkl')
    #
    # df = pd.read_pickle('dataset1/dataframe.pkl')
    #
    # y=df['label'].to_numpy()
    # x=df['data'].tolist()







