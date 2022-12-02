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
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay

# Hyperparameters
MAX_LEN = 128
N_SAMPLE_PER_CLASS=10

API_FEAT=False # if True uses API feature otherwise it uses every run-time info

if API_FEAT:
    DATAFRAME_FILENAME='Avast/avast_dataframe_test_api.pkl'
    MODEL_FILENAME=f'Avast/best_model_api_{MAX_LEN}.pt'
    HISTORY_FILENAME=f'Avast/history_api_{MAX_LEN}.pkl'
    VOCAB_FILENAME = 'vocab_avast_api.txt'
else:
    DATAFRAME_FILENAME = 'Avast/avast_dataframe_train.pkl'
    MODEL_FILENAME = f'Avast/best_model_{MAX_LEN}.pt'
    HISTORY_FILENAME = f'Avast/history_{MAX_LEN}.pkl'
    VOCAB_FILENAME = 'vocab_avast.txt'

with open(VOCAB_FILENAME, 'r') as fp:
    vocab = fp.read()
    vocab = vocab.split('\n')


tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
tokenizer = tokenizer.train_new_from_iterator(vocab, vocab_size=50000)

# Plot the train history
# with open(HISTORY_FILENAME,'rb') as fp:
#     history=pickle.load(fp)
# plot_history(history)

df_ts = pd.read_pickle(DATAFRAME_FILENAME)
# Subset
df_ts=create_subset(df_ts,n_sample_per_class=N_SAMPLE_PER_CLASS)

print(f'Test samples: {len(df_ts)}')
print(len(set(df_ts['classification_family'])))

test_dataset = AvastDataset(df_ts, tokenizer, MAX_LEN)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',num_labels=10)
model.load_state_dict(torch.load(MODEL_FILENAME))
model.to('cpu')
print('\nSTART TESTING')

acc,y_pred,y=compute_accuracy(model, test_loader, device='cpu')
print(f' ---- Test Accuracy: {acc:.2f}%')

y_pred=[AvastDataset.classes[i] for i in y_pred]
y=[AvastDataset.classes[i] for i in y]

#Confusion matrix
# plt.figure(figsize=(10,10))
ConfusionMatrixDisplay.from_predictions(y,
                                        y_pred,
                                        normalize='true',
                                        labels=AvastDataset.classes,
                                        cmap='Blues',
                                        colorbar=False,
                                        xticks_rotation='vertical')
plt.tight_layout()
plt.show()

# plt.savefig('conf_mat.pdf')