import warnings
import IPython
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.layers import Activation, Permute, RepeatVector, LSTM, Bidirectional, Multiply, Lambda, Dense, Dropout, \
    Input,Flatten,Embedding,Attention
from keras.callbacks import History, CSVLogger, ModelCheckpoint, EarlyStopping
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
from utils import get_label_date_text_dataframe_dataset1,get_label_date_text_dataframe_avast
from keras import initializers, regularizers
from keras import constraints
from lime import lime_text
from tqdm import tqdm
import json
from utils import *

warnings.simplefilter(action='ignore', category=FutureWarning)


def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


class AttentionWithContext(keras.layers.Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

    def get_config(self):
        config = {
            "W_regularizer": self.W_regularizer,
            "u_regularizer": self.u_regularizer,
            "b_regularizer": self.b_regularizer,
            "W_constraint": self.W_constraint,
            "u_constraint": self.u_constraint,
            "b_constraint": self.b_constraint,
            "bias": self.bias,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Addition(keras.layers.Layer):
    """
    This layer is supposed to add of all activation weight.
    We split this from AttentionWithContext to help us getting the activation weights
    follows this equation:
    (1) v = \sum_t(\alpha_t * h_t)

    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    """

    def __init__(self, **kwargs):
        super(Addition, self).__init__(**kwargs)

    def build(self, input_shape):
        self.output_dim = input_shape[-1]
        super(Addition, self).build(input_shape)

    def call(self, x):
        return K.sum(x, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

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
    # x=AttentionWithContext()(x)
    # x=Addition(x)
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
    MAXLEN = 500 # 500
    EMBEDDING_DIM=256 # 256
    BATCH_SIZE = 50
    EPOCHS = 15 # 10
    LEARNING_RATE = 0.0001
    MODE='detection' # 'classification' or 'detection'
    TYPE_SPLIT='random' # 'time' or 'random'

    # Import data
    # df = pd.read_csv("spamdata_v2.csv")
    if MODE=='classification':
        split_date='2019-08-01'
        classes = ['Adload', 'Emotet', 'HarHar', 'Lokibot', 'njRAT', 'Qakbot', 'Swisyn', 'Trickbot', 'Ursnif', 'Zeus']
        # df = pd.read_csv("data_avast_100.csv")
        df = get_label_date_text_dataframe_avast("Avast\\subset_100.csv")
        # df = pd.read_csv("data_avast.csv")
    elif MODE=='detection':
        split_date = "2013-08-09"
        classes=["Benign","Malign"]
        df = get_label_date_text_dataframe_dataset1("dataset1\\labels_preproc.csv")
        # df = pd.read_csv("data.csv")
        # df = pd.read_csv("spamdata_v2.csv")

    df = df.sample(frac=1) # Shuffle dataset
    df=df.iloc[0:2000, :].reset_index(drop=True) # Subset
    print(df.head())
    n_classes=len(set(df['label']))
    print(f"Number of classes: {n_classes}")

    # Create training, validation and test set
    if TYPE_SPLIT == 'random':
        x_tr, x_tmp, y_tr, y_tmp = train_test_split(df['text'], df['label'], test_size=0.2,stratify=df['label'])
    elif TYPE_SPLIT == 'time':
        x_tr,y_tr = df[df['date'] < split_date]['text'],df[df['date'] < split_date]['label']
        x_tmp,y_tmp = df[df['date'] >= split_date]['text'],df[df['date'] >= split_date]['label']
    x_val, x_ts, y_val, y_ts = train_test_split(x_tmp, y_tmp, test_size=0.6,stratify=y_tmp)

    print(f"Split train-test: {TYPE_SPLIT}")
    print(f"Train size: {len(y_tr)} -- n_classes:{len(set(y_tr))}")
    print(f"Validation size: {len(y_val)} -- n_classes:{len(set(y_val))}")
    print(f"Test size: {len(y_ts)} -- n_classes:{len(set(y_ts))}")

    # Tokenize
    x_tr_tokens,x_val_tokens,x_ts_tokens, vocab_size, tokenizer = tokenize_data(x_tr,x_val,x_ts,maxlen=MAXLEN)
    print(f"Vocab size: {vocab_size}")

    # Save tokenizer
    with open(f'tokenizer_{MODE}.pickle', 'wb') as fp:
        pickle.dump(tokenizer, fp)


    # Model definition
    model=get_neurlux(vocab_size,EMBEDDING_DIM,MAXLEN,mode=MODE,n_classes=n_classes)
    #print(model.summary())


    es = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 10)
    mc = ModelCheckpoint(f'./model_{MODE}.h5', monitor = 'val_accuracy', mode = 'max', verbose = 1, save_best_only = True)
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

    ConfusionMatrixDisplay.from_predictions([classes[i] for i in y_ts],[classes[i] for i in y_pred],normalize='true',
                                            labels=classes, cmap='Blues',colorbar=False,xticks_rotation='vertical')
    plt.title("Confusion matrix")
    plt.tight_layout()
    plt.show()








