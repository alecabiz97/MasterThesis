import json
import os
from utils import *
import pandas as pd
import numpy as np
from tqdm import tqdm
import csv
from collections import Counter
from transformers import DistilBertTokenizerFast

def create_dataframe(benign_root,malign_root,n_sample_per_class=None):

    # list of filenames
    ben_filenames = getListOfFiles(benign_root)
    mal_filenames = getListOfFiles(malign_root)

    if n_sample_per_class:
        ben_filenames = ben_filenames[0:n_sample_per_class]
        mal_filenames = mal_filenames[0:n_sample_per_class]

    # 0 -> ben, 1 -> mal
    df = pd.DataFrame({'label': int(), 'data': str()}, index=[])
    for i, filename in enumerate(tqdm(ben_filenames + mal_filenames)):
        try:
            f = open(filename)
            data = json.load(f)
            f.close()

            wl = get_all_words_from_dict(data)

            words = ''
            for x in wl:
                words += x + ' '

            if filename.split('\\')[0] == 'ben_reports':
                label = 0
            elif filename.split('\\')[0] == 'mal_reports':
                label = 1

            df_tmp = pd.DataFrame({'label': label, 'data': words}, index=[i])

            df = pd.concat([df, df_tmp], ignore_index=True)
        except json.decoder.JSONDecodeError:
            pass

    return df

def create_cleaned_files(files,output_dir):
    cnt_problem_files=0
    for file in tqdm(files):
        try:
            f=open(file,'r')
            data=json.load(f)
            f.close()

            words=get_all_words_from_dict(data)

            new_file=os.path.join(output_dir,file.split('\\')[1].split('.')[0]+'.txt')
            f=open(new_file,'w')
            for w in words:
                f.write(w+'\n')
            f.close()
        except json.decoder.JSONDecodeError:
            cnt_problem_files +=1

    print(f'Number of problems: {cnt_problem_files}')

def create_cleaned_files_api(files,output_dir):
    cnt_problem_files=0
    for file in tqdm(files):
        try:
            f=open(file,'r')
            data=json.load(f)
            k = list(data['behavior']['apistats'].keys())[0]
            api=data['behavior']['apistats'][k]
            f.close()

            new_file=os.path.join(output_dir,file.split('\\')[1].split('.')[0]+'.txt')
            f=open(new_file,'w')
            for k,v in api.items():
                s=f'{k} {v} \n'
                f.write(s)
            f.close()
        except json.decoder.JSONDecodeError:
            cnt_problem_files +=1
        except KeyError:
            cnt_problem_files += 1

    print(f'Number of problems: {cnt_problem_files}')

def create_avast_dataframe(files_path,labels_file,n_sample=None):
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

            words = '  '.join(get_all_words_from_dict(d))

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

    # df=create_avast_dataframe(avast_root,labels_file) # Create dataframe (all samples)
    # df.to_pickle('Avast/avast_dataframe.pkl')
    # df = pd.read_pickle('Avast/avast_dataframe.pkl')
    # df_tr, df_ts = split_train_test(df,train_test_date='2019-08-01') # Split train and test
    # df_tr.to_pickle('Avast/avast_dataframe_train.pkl')
    # df_ts.to_pickle('Avast/avast_dataframe_test.pkl')

    df_train = pd.read_pickle('Avast/avast_dataframe_train.pkl')
    df_ts = pd.read_pickle('Avast/avast_dataframe_test.pkl')

    df_tr,df_val=split_train_val(df_train,val_frac=0.1) # Split train and validation

    print('Train', len(df_tr))
    print(set(df_tr['classification_family']))

    print('Validation', len(df_val))
    print(set(df_val['classification_family']))

    print('Test', len(df_ts))
    print(set(df_ts['classification_family']))

    # Create vocab file
    # all_words=[]
    # x = df_tr['words'].values
    # for i in range(len(df_tr)):
    #     all_words.extend(x[i].split('  '))
    # create_vocab_file('vocab_avast.txt',all_words,n_most_common=10000)

    # Read vocab file
    with open('vocab_avast.txt', 'r') as fp:
        vocab = fp.read()
        vocab = vocab.split('\n')

    # Try Tokenizer
    example = "ole32.dll.CoRegisterInitializeSpy"
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    x = tokenizer(example, truncation=True, padding=True)
    print(tokenizer.convert_ids_to_tokens(x['input_ids']))

    tokenizer = tokenizer.train_new_from_iterator(vocab, vocab_size=20000)
    x = tokenizer(example, truncation=True, padding=True)
    print(tokenizer.convert_ids_to_tokens(x['input_ids']))


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







