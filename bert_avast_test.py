import torch
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel,\
    BertTokenizer, BertModel,DistilBertModel, DistilBertConfig, DistilBertTokenizer,DistilBertForSequenceClassification,DistilBertTokenizerFast
import time
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from DatasetClasses import AvastDataset
from utils import *
import copy
from torch.nn.functional import softmax
import pickle

# Hyperparameters
MAX_LEN = 256
BATCH_SIZE = 2

with open('vocab_avast.txt', 'r') as fp:
    vocab = fp.read()
    vocab = vocab.split('\n')

tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
tokenizer = tokenizer.train_new_from_iterator(vocab, vocab_size=10000)

# Plot the train history
# with open(f'Avast/history_{MAX_LEN}.pkl','rb') as fp:
#     history=pickle.load(fp)
# plot_history(history)

df_ts = pd.read_pickle('Avast/avast_dataframe_test.pkl')

# Subset
df_ts=create_subset(df_ts,n_sample_per_class=10)

print(f'Test samples: {len(df_ts)}')
print(len(set(df_ts['classification_family'])))

test_dataset = AvastDataset(df_ts, tokenizer, MAX_LEN)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',num_labels=10)
model.load_state_dict(torch.load(f'Avast/best_model_{MAX_LEN}.pt'))
model.to('cpu')
print('\nSTART TESTING')

acc=compute_accuracy(model, test_loader, device='cpu')
print(f' ---- Test Accuracy: {acc:.2f}%')