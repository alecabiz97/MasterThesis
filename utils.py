import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt

def plot_history(history):
    for phase in ['Train', 'Val']:
        data = np.array(history[phase])
        x = np.arange(1, data.shape[0] + 1)
        plt.subplot(2, 1, 1)
        plt.plot(x, data[:, 0], label='{}'.format(phase), marker='.')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.legend()
        plt.subplot(2, 1, 2)
        plt.plot(x, data[:, 1], label='{}'.format(phase), marker='.')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
    plt.legend()
    plt.show()

def create_subset(df,n_sample_per_class):
    """Create a subset of Datafram with respect to the number of samples for each class (malware family) """
    labels = set(df['classification_family'])

    df_sub=pd.DataFrame(columns=df.columns)
    for label in labels:
        df_tmp = df[df['classification_family'] == label][0:n_sample_per_class]
        df_tmp=df_tmp.sample(frac=1) # To shuffle the dataframe
        df_sub=pd.concat([df_sub,df_tmp])

    return df_sub.reset_index(drop=True)


def split_train_test(df,train_test_date='2019-08-01'):
    """Split a Dataframe in train and test with respect to a date threshold """
    df_train = df[df['date'] < train_test_date]
    df_ts = df[df['date'] >= train_test_date]

    # Reset index
    df_train = df_train.reset_index(drop=True)
    df_ts = df_ts.reset_index(drop=True)
    return df_train, df_ts

def split_train_val(df,val_frac=0.1):
    """Split a Dataframe in train and validation with respect to the validation size"""
    # Initialization of dataframes
    df_tr = pd.DataFrame(columns=df.columns)
    df_val = pd.DataFrame(columns=df.columns)

    labels = set(df['classification_family'])
    for label in labels:
        df_tmp = df[df['classification_family'] == label]
        df_tmp=df_tmp.sample(frac=1) # To shuffle the dataframe
        df_tr_tmp, df_val_tmp = train_test_split(df_tmp, test_size=val_frac)
        df_tr = pd.concat([df_tr, df_tr_tmp])
        df_val = pd.concat([df_val, df_val_tmp])

    # Reset index
    df_tr = df_tr.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    return df_tr, df_val


def create_vocab_file(filename, all_words, n_most_common):
    """Create a .txt file with the n_most_common words in the list all_words """
    counts = Counter(all_words)
    vocab = counts.most_common(n=n_most_common)
    f = open(filename, 'w')
    for w, _ in vocab:
        f.write(f'{w}\n')
    f.close()

def getListOfFiles(dirName):
    """ Create a list of all files give the directory"""

    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles

def get_all_words_from_dict(d):
    """Return all the words(keys and values) inside a dictionary"""
    words=[]
    for k, v in d.items():
        if isinstance(k, str) or isinstance(k, int) or isinstance(k, float):
            words.append(str(k))
        if isinstance(v, str) or isinstance(v, int) or isinstance(v, float):
            words.append(str(v))

        if isinstance(v, list):
            for v2 in v:
                if isinstance(v2, dict):
                    words.extend(get_all_words_from_dict(v2))
                elif isinstance(v2, str) or isinstance(v2, int) or isinstance(v2, float):
                    words.append(str(v2))

        elif isinstance(v, dict):
                words.extend(get_all_words_from_dict(v))

    return words


def compute_accuracy(model, data_loader, device):
    """Return the accuracy given the model, the dataloader and the device"""
    with torch.no_grad():
        correct_pred, num_examples = 0, 0

        for batch_idx, batch in enumerate(data_loader):
            ### Prepare data
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss, logits = outputs['loss'], outputs['logits']

            _, predicted_labels = torch.max(logits, 1)

            num_examples += labels.size(0)

            correct_pred += (predicted_labels == labels).sum()
    return float(correct_pred.float() / num_examples * 100)