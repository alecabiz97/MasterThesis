import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import re
from tqdm import tqdm
import json
from sklearn.metrics import ConfusionMatrixDisplay
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import keras
from keras import constraints, initializers, regularizers
import keras.backend as K
import tensorflow as tf


def get_top_feature_attention(attention_model,tokenizer, x_tokens, topk=10):
    score = attention_model.predict(x_tokens)
    score = np.expand_dims(score, axis=0) if x_tokens.shape[0] == 1 else score
    top_score = np.argsort(-score)[:, 0:topk]
    top_feat_att = []
    for i in range(top_score.shape[0]):
        top_feat_att.append(tokenizer.sequences_to_texts([x_tokens[i, top_score[i, :]]])[0].split())
    return top_feat_att



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
        return K.sum(weighted_input, axis=1),tf.squeeze(a)

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



def tokenize_data(x_tr,x_val,x_ts,maxlen):
    tokenizer = Tokenizer(num_words=10000)

    tokenizer.fit_on_texts(x_tr.values)
    x_tr = tokenizer.texts_to_sequences(x_tr)
    x_tr=pad_sequences(x_tr,maxlen=maxlen,padding='post')
    vocab_size = len(tokenizer.word_index) + 1

    x_val = tokenizer.texts_to_sequences(x_val.values)
    x_val = pad_sequences(x_val, maxlen=maxlen, padding='post')

    x_ts = tokenizer.texts_to_sequences(x_ts.values)
    x_ts = pad_sequences(x_ts, maxlen=maxlen, padding='post')
    return x_tr,x_val,x_ts, vocab_size, tokenizer


def plot_confusion_matrix(y_true,y_pred,classes):
    plt.figure(figsize=(10, 10))

    ConfusionMatrixDisplay.from_predictions([classes[i] for i in y_true], [classes[i] for i in y_pred], normalize='true',
                                            labels=classes, cmap='Blues', colorbar=False, xticks_rotation='vertical')
    plt.title("Confusion matrix")
    plt.tight_layout()
    plt.show()
def get_label_date_text_dataframe_avast(meta_path,feature_maxlen=None):
    meta = pd.read_csv(meta_path)
    root = 'Avast\\public_small_reports'
    classes = ['Adload', 'Emotet', 'HarHar', 'Lokibot', 'njRAT', 'Qakbot', 'Swisyn', 'Trickbot', 'Ursnif', 'Zeus']
    df = pd.DataFrame({"label": int(),
                       "date": str(),
                       "text": str()}, index=[])
    for i, (sha, family,date) in enumerate(tqdm(meta[["sha256", "classification_family","date"]].values)):
        filepath = f"{root}\\{sha}.json"
        try:
            with open(filepath, 'r') as fp:
                data = json.load(fp)

            if feature_maxlen is None:
                text = preprocessing_data(str(data["behavior"]))
            else:
                text=[]
                for feat in feature_maxlen.keys():
                    x=data["behavior"]["summary"][feat]
                    text.append(x[0:min(len(x),feature_maxlen[feat])])

                text = preprocessing_data(str(text))
            # text = preprocessing_data(str(data["static"]))
            # text=" ".join(data["behavior"]["apistats"])

            y = classes.index(family)
            df_tmp = pd.DataFrame({'label': y,
                                   "date": date,
                                   'text': text}, index=[i])
            df = pd.concat([df, df_tmp], ignore_index=True)
            pass
        except:
            pass

    return df

def get_label_date_text_dataframe_dataset1(meta_path,feature_maxlen):
    meta = pd.read_csv(meta_path)
    df = pd.DataFrame({"label": int(),
                       "date": str(),
                       "text": str()}, index=[])
    for i, (filepath, label,date) in enumerate(tqdm(meta[['name', 'label','date']].values)):
        with open(f"{filepath}.json", 'r') as fp:
            data = json.load(fp)

        feat=[]
        if 'apistats' in feature_maxlen.keys():
            x = data["behavior"]['apistats']
            feat.append(x[0:min(len(x), feature_maxlen["apistats"])])
        if 'apistats_opt' in feature_maxlen.keys():
            x = data["behavior"]["apistats_opt"]
            feat.append(x[0:min(len(x),feature_maxlen["apistats_opt"])])
        if 'regkey_opened' in feature_maxlen.keys():
            x = data["behavior"]["summary"]["regkey_opened"]
            feat.append(x[0:min(len(x),feature_maxlen["regkey_opened"])])
        if 'regkey_read' in feature_maxlen.keys():
            x = data["behavior"]["summary"]["regkey_read"]
            feat.append(x[0:min(len(x),feature_maxlen["regkey_read"])])
        if 'dll_loaded' in feature_maxlen.keys():
            x = data["behavior"]["summary"]["dll_loaded"]
            feat.append(x[0:min(len(x),feature_maxlen["dll_loaded"])])
        if 'mutex' in feature_maxlen.keys():
            x = data["behavior"]["summary"]["mutex"]
            feat.append(x[0:min(len(x),feature_maxlen["mutex"])])


        text = preprocessing_data(str(feat))
        df_tmp = pd.DataFrame({'label': label,
                               "date": date,
                               'text': text}, index=[i])
        df = pd.concat([df, df_tmp], ignore_index=True)

    # print(len(df))
    return df

def preprocessing_data(text):
    # return re.sub('[^a-zA-Z0-9,:]','',text).replace(',',' ').replace(':',' ').lower()
    return re.sub('[^a-zA-Z0-9,:]', '', text).lower().replace("c:","c").replace(',', ' ').replace(':', ' ')

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
    return float(correct_pred.float() / num_examples * 100),predicted_labels,labels


