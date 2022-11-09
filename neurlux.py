from keras.preprocessing.text import Tokenizer
from keras.layers import Activation, Permute, RepeatVector, LSTM, Bidirectional, Multiply, Lambda, Dense, Dropout, \
    Input,Flatten,Embedding
import warnings
import IPython
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Activation, Permute, RepeatVector, LSTM, Bidirectional, Multiply, Lambda, Dense, Dropout, \
    Input,Flatten,Embedding,Attention
from keras.callbacks import History, CSVLogger, ModelCheckpoint, EarlyStopping
#from keras.engine.topology import Layer, InputSpec
from keras.models import Model, load_model, Sequential
from keras.utils import CustomObjectScope
import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import os
import keras
#from Attention import *
from sklearn.metrics import accuracy_score, precision_recall_curve, roc_curve, f1_score, confusion_matrix, recall_score, precision_score, ConfusionMatrixDisplay
from keras.layers import Conv1D, MaxPooling1D, CuDNNLSTM
import sys
import time
from torch import nn
import torch
from torch.utils.data import Dataset,DataLoader
import copy
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, plot_roc_curve, classification_report, RocCurveDisplay
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from utils import get_label_text_dataframe_dataset1,get_label_text_dataframe_avast

warnings.simplefilter(action='ignore', category=FutureWarning)

def tokenize_data(x_tr,x_val,x_ts,maxlen):
    tokenizer = Tokenizer(num_words=10000)

    tokenizer.fit_on_texts(x_tr.values)
    x_tr = tokenizer.texts_to_sequences(x_tr)
    x_tr=pad_sequences(x_tr,maxlen=maxlen,padding='post')
    vocab_size = len(tokenizer.word_index) + 1

    x_val = tokenizer.texts_to_sequences(x_val.values)
    x_val = pad_sequences(x_val, maxlen=MAXLEN, padding='post')

    x_ts = tokenizer.texts_to_sequences(x_ts.values)
    x_ts = pad_sequences(x_ts, maxlen=MAXLEN, padding='post')
    return x_tr,x_val,x_ts, vocab_size, tokenizer


def get_neurlux(vocab_size,EMBEDDING_DIM,MAXLEN,mode,n_classes=None):
    inp=keras.layers.Input(shape=(MAXLEN))
    x=keras.layers.Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM,input_length=MAXLEN)(inp)
    x = keras.layers.Conv1D(filters=100, kernel_size=4, padding='same', activation='relu')(x)
    x = keras.layers.MaxPooling1D(pool_size=4,data_format="channels_first")(x)
    # x = keras.layers.Bidirectional(keras.layers.CuDNNLSTM(64, return_sequences=True))(x)
    x = keras.layers.Bidirectional(keras.layers.CuDNNLSTM(32))(x)
    # x,att_score=keras.layers.Attention(name='attention_vec')([x,x], return_attention_scores=True)
    x = keras.layers.Dense(10, activation="relu")(x)
    x = keras.layers.Dropout(0.25)(x)


    if mode=='classification':
        x = keras.layers.Dense(n_classes, activation="softmax")(x)
        loss="sparse_categorical_crossentropy"
    elif mode=='detection':
        x = keras.layers.Dense(1, activation="sigmoid")(x)
        loss = "binary_crossentropy"

    model=keras.models.Model(inputs=inp,outputs=x)

    model.compile(loss=loss, optimizer='adam', metrics='accuracy')
    return model

if __name__ == '__main__':

    # Hyperparameters
    MAXLEN = 500
    EMBEDDING_DIM=256
    BATCH_SIZE = 50
    EPOCHS = 10
    LEARNING_RATE = 0.0001
    MODE='detection' # 'classification' or 'detection'

    # Import data
    # df = pd.read_csv("spamdata_v2.csv")
    if MODE=='classification':
        classes = ['Adload', 'Emotet', 'HarHar', 'Lokibot', 'njRAT', 'Qakbot', 'Swisyn', 'Trickbot', 'Ursnif', 'Zeus']
        df = pd.read_csv("data_avast_100.csv")
        # df = pd.read_csv("data_avast.csv")
        # df = get_label_text_dataframe_avast("Avast\\subset_100.csv")
    elif MODE=='detection':
        classes=["Benign","Malign"]
        df = pd.read_csv("data.csv")
        # df = get_label_text_dataframe_dataset1("dataset1\\labels_preproc.csv")

    df = df.sample(frac=1) # Shuffle dataset
    df=df.iloc[0:1000, :].reset_index(drop=True) # Subset
    print(df.head())
    n_classes=len(set(df['label']))
    print(f"Number of classes: {n_classes}")

    # Create training, validation and test set
    x_tr, x_tmp, y_tr, y_tmp = train_test_split(df['text'], df['label'], random_state=2018, test_size=0.2,stratify=df['label'])
    x_val, x_ts, y_val, y_ts = train_test_split(x_tmp, y_tmp, random_state=2018,test_size=0.6,stratify=y_tmp)

    print(f"Train size: {len(y_tr)} -- n_classes:{len(set(y_tr))}")
    print(f"Validation size: {len(y_val)} -- n_classes:{len(set(y_val))}")
    print(f"Test size: {len(y_ts)} -- n_classes:{len(set(y_ts))}")

    # Tokenize
    x_tr_tokens,x_val_tokens,x_ts_tokens, vocab_size, tokenizer = tokenize_data(x_tr,x_val,x_ts,maxlen=MAXLEN)
    print(f"Vocab size: {vocab_size}")

    # Model definition
    model=get_neurlux(vocab_size,EMBEDDING_DIM,MAXLEN,mode=MODE,n_classes=n_classes)
    #print(model.summary())

    es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 10)
    mc = ModelCheckpoint('./model.h5', monitor = 'val_accuracy', mode = 'max', verbose = 1, save_best_only = True)
    print("START TRAINING")
    history_embedding = model.fit(tf.constant(x_tr_tokens),tf.constant(y_tr),
                                  epochs = EPOCHS, batch_size = BATCH_SIZE,
                                  validation_data=(tf.constant(x_val_tokens),tf.constant(y_val)),
                                  verbose = 1, callbacks= [es, mc])

    #Test
    print("TEST")
    if MODE=='classification':
        y_pred = np.argmax(model.predict(tf.constant(x_ts_tokens)), axis=1)
    elif MODE=='detection':
        scores=model.predict(tf.constant(x_ts_tokens)).squeeze()
        y_pred=scores.round().astype(int)
        print(f"AUC: {roc_auc_score(y_ts,scores)}")
        RocCurveDisplay.from_predictions(y_ts,scores)
        plt.show()


    print(classification_report(y_pred, y_ts))
    #print(confusion_matrix(y_ts,y_pred))

    # Confusion matrix
    plt.figure(figsize=(10,10))

    ConfusionMatrixDisplay.from_predictions([classes[i] for i in y_ts],
                                            [classes[i] for i in y_pred],
                                            normalize='true',
                                            labels=classes,
                                            cmap='Blues',
                                            colorbar=False,
                                            # values_format=".2g",
                                            xticks_rotation='vertical')
    plt.title("Confusion matrix")
    plt.tight_layout()
    plt.show()

