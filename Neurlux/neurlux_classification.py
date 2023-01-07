import warnings

import numpy as np
import pandas as pd
from keras.layers import Activation, LSTM, Bidirectional, Dense, Dropout, Input,Embedding, Conv1D,MaxPooling1D,CuDNNLSTM
import keras.backend as K
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import keras
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import initializers, regularizers
from keras import constraints
from utils import *
import shap
from collections import Counter

warnings.simplefilter(action='ignore', category=FutureWarning)


def get_neurlux(vocab_size,EMBEDDING_DIM,MAXLEN,n_classes=None,with_attention=False):
    inp=Input(shape=(MAXLEN))
    x=Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM,input_length=MAXLEN)(inp)
    x = Conv1D(filters=100, kernel_size=4, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=4,data_format="channels_first")(x)
    # x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)

    if with_attention:
        x = Bidirectional(CuDNNLSTM(32,return_sequences=True))(x)
        x, attention_out = AttentionWithContext()(x)
    else:
        x = Bidirectional(CuDNNLSTM(32))(x)

    x = Dense(10, activation="relu")(x)
    x = Dropout(0.25)(x)
    x = Dense(n_classes, activation="softmax")(x)

    model=keras.models.Model(inputs=inp,outputs=x)
    model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics='accuracy')

    if with_attention:
        attention_model = keras.models.Model(inputs=inp, outputs=attention_out)
        return model, attention_model
    else:
        return model, None

if __name__ == '__main__':

    # Hyperparameters
    feature_maxlen = {
        'keys': 500,
        'resolved_apis': 200,
        'executed_commands': 20,
        'write_keys': 20,
        'files': 100,
        'read_files': 200,
        'write_files': 100,
        'delete_keys': 100,
        'read_keys': 250,
        'delete_files':50,
        'mutexes': 20,
        # "started_services":5,
        # "created_services":5,
    }
    MAXLEN = sum(feature_maxlen.values())

    EMBEDDING_DIM=256 # 256
    BATCH_SIZE = 50
    EPOCHS = 30 # 10
    LEARNING_RATE = 0.0001
    TYPE_SPLIT='random' # 'time' or 'random'
    SPLIT_DATE_VAL_TS = "2019-07-01"
    SPLIT_DATE_TR_VAL = "2019-05-01"
    SUBSET_N_SAMPLES=None # if None takes all data
    WITH_ATTENTION = True
    TRAINING = False
    meta_path = "..\\data\\Avast\\subset_100.csv"
    # model_name = "Neurlux_Avast"
    model_name = f"neurlux_avast_all_{EPOCHS}_{TYPE_SPLIT}"
    classes = ['Adload', 'Emotet', 'HarHar', 'Lokibot', 'njRAT', 'Qakbot', 'Swisyn', 'Trickbot', 'Ursnif', 'Zeus']

    # Explanation
    SHAP=True
    LIME = True
    EXP_MODE = 'multi'  # single or multi
    TOPK_FEATURE = 10
    N_SAMPLES_EXP = 2
    SAVE_EXP_DICT=False
    feature_set_path = "../data/Avast/Avast_feature_set.json"

    # Import data
    df = import_data(meta_path=meta_path,subset_n_samples=SUBSET_N_SAMPLES,feature_maxlen=feature_maxlen,
                              callback=get_label_date_text_dataframe_avast)
    n_classes = len(classes)

    # Split Train-Test-Validation
    x_tr, y_tr, x_val, y_val, x_ts, y_ts = split_train_val_test_dataframe(df, type_split=TYPE_SPLIT,
                                                                          split_dates=[SPLIT_DATE_VAL_TS,SPLIT_DATE_TR_VAL], tr=0.8)

    # Tokenize
    x_tr_tokens, x_val_tokens, x_ts_tokens, vocab_size, tokenizer = tokenize_data(x_tr, x_val, x_ts, maxlen=MAXLEN)
    print(f"Vocab size: {vocab_size}")


    # Model definition
    model,attention_model = get_neurlux(vocab_size, EMBEDDING_DIM, MAXLEN, n_classes=n_classes,with_attention=WITH_ATTENTION)
    print(model.summary())

    if TRAINING:
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        mc = ModelCheckpoint(f"./trained_models/avast/{model_name}.h5", monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
        print("START TRAINING")
        history_embedding = model.fit(tf.constant(x_tr_tokens), tf.constant(y_tr),
                                      epochs=EPOCHS, batch_size=BATCH_SIZE,
                                      validation_data=(tf.constant(x_val_tokens), tf.constant(y_val)),
                                      verbose=1, callbacks=[es, mc])
    else:
        model.load_weights(f"./trained_models/avast/{model_name}.h5")

    #Test
    print("TEST")

    # print(classification_report(y_pred, y_ts))
    print(f"Train accuracy: {model.evaluate(x_tr_tokens, np.array(y_tr),verbose=False)[1]}")
    print(f"Test accuracy: {model.evaluate(x_ts_tokens, np.array(y_ts),verbose=False)[1]}")
    # print(confusion_matrix(y_ts,y_pred))
    y_pred = np.argmax(model.predict(tf.constant(x_ts_tokens)), axis=1)

    # Confusion matrix
    plot_confusion_matrix(y_true=y_ts, y_pred=y_pred, classes=classes)

# %% Explanation

    hash="b773c3406e289cd100237bec78642bf0cbc95f0c408b20165cc3d02b89d35081" # Emotet

    #     hash="f7a4a26c10c86ce3c1e9b606ed3e59c4c12758c24de95bd68016200b28e6b06b" # Emotet
    #     hash="6847bd9c431b65456654ce635ce365ca4c66bb056f648eab54e87ad7b7269c60" # Trickbot
    #     hash="2e1640fb12cc66af4428e3e8e2a0de4d768f2e4085144a3f04aafc79fd53c38a" # Trickbot
    #     hash="1b6ca28027f62a4922348c55b891472c9530da1b0ab2af1ab615a491612bea01" # Trickbot
    #     hash="efb793eafd7993152fcb0075887584cd65bab183d0ebea0bbbcf05255c8be8db" # njRAT
    #     hash="32c58040d3d6ec5305a1a0ebb48ba05aebe3ac2f905a7f152f32fc9170e16711" # Trickbot

    y_true="Emotet"

    # LIME Explanation
    if LIME:

        if EXP_MODE == "single":
            idx_true = np.array([classes.index(y_true)])
            with open(f"..\\data\\Avast\\public_small_reports\\{hash}.json", "r") as fp:
                data = json.load(fp)

            text = []
            for feat in feature_maxlen.keys():
                x = data["behavior"]['summary'][feat]
                text.append(x[0:min(len(x), feature_maxlen[feat])])

            x = pd.Series(preprocessing_data(str(text)))
            x_tokens = tokenizer.texts_to_sequences(x)
            x_tokens = pad_sequences(x_tokens, maxlen=MAXLEN, padding='post')
            y = pd.Series(idx_true)
        elif EXP_MODE == "multi":

            # Subset
            x=[]
            x_tokens = np.zeros(shape=(N_SAMPLES_EXP * len(classes), x_ts_tokens.shape[1]))
            y = np.zeros(shape=(N_SAMPLES_EXP * len(classes)), dtype=int)
            for i in range(len(classes)):
                idx = (y_ts == i).to_numpy()
                x_tokens[i * N_SAMPLES_EXP:i * N_SAMPLES_EXP + N_SAMPLES_EXP, :] = x_ts_tokens[idx, :][0:N_SAMPLES_EXP,:]
                y[i * N_SAMPLES_EXP:i * N_SAMPLES_EXP + N_SAMPLES_EXP] = y_ts.values[idx][0:N_SAMPLES_EXP]
                x.extend(x_ts.values[idx][0:N_SAMPLES_EXP].tolist())

            x=pd.Series(x)
            y=pd.Series(y.tolist())

            # x = x_ts[0:N_SAMPLES_EXP]
            # x_tokens = x_ts_tokens[0:N_SAMPLES_EXP]
            # y = y_ts[0:N_SAMPLES_EXP]

            print(y.shape, x_tokens.shape)

        top_feat_dict_lime = lime_explanation_avast(x=x, x_tokens=x_tokens, y=y, model=model,tokenizer=tokenizer,
                                              feature_maxlen=feature_maxlen,classes=classes,
                                              num_features=TOPK_FEATURE, save_html=False)

        # Save top feat dict
        if SAVE_EXP_DICT:
            with open(f"top_feat_dict_lime_{model_name}.json", "w") as outfile:
                json.dump(top_feat_dict_lime, outfile, indent=4)

        # Load top feat dict
        # with open(f"top_feat_dict_lime_{model_name}.json", "r") as outfile:
        #     top_feat_dict_lime = json.load(outfile)
        print_top_feature_avast(top_feat_dict_lime,feature_set_path=feature_set_path)


# %%
    if SHAP:
        explainer = shap.KernelExplainer(model.predict, np.zeros((1,x_ts_tokens.shape[1])))
        # explainer = shap.KernelExplainer(model.predict,  shap.sample(x_tr_tokens,10))

        if EXP_MODE == "single":
            idx_true = np.array([classes.index(y_true)])
            with open(f"..\\data\\Avast\\public_small_reports\\{hash}.json", "r") as fp:
                data = json.load(fp)

            text = []
            for feat in feature_maxlen.keys():
                x = data["behavior"]['summary'][feat]
                text.append(x[0:min(len(x), feature_maxlen[feat])])
            sample = preprocessing_data(str(text))
            sample_tokens = tokenizer.texts_to_sequences([sample])
            sample_tokens = pad_sequences(sample_tokens, maxlen=MAXLEN, padding='post')

        elif EXP_MODE == "multi":

            # Subset
            sample_tokens = np.zeros(shape=(N_SAMPLES_EXP * len(classes), x_ts_tokens.shape[1]))
            idx_true = np.zeros(shape=(N_SAMPLES_EXP * len(classes)),dtype=int)
            for i in range(len(classes)):
                idx = (y_ts == i).to_numpy()
                sample_tokens[i * N_SAMPLES_EXP:i * N_SAMPLES_EXP + N_SAMPLES_EXP, :] = x_ts_tokens[idx, :][0:N_SAMPLES_EXP, :]
                idx_true[i * N_SAMPLES_EXP:i * N_SAMPLES_EXP + N_SAMPLES_EXP] = y_ts.values[idx][0:N_SAMPLES_EXP]

            print(idx_true.shape, sample_tokens.shape)

            # sample = x_ts.iloc[0:0 + N_SAMPLES_EXP]
            # sample_tokens = x_ts_tokens[0:0 + N_SAMPLES_EXP]
            # idx_true = y_ts.iloc[0:0 + N_SAMPLES_EXP].values

        top_feat_dict_shap = shap_explanation_avast(explainer=explainer, sample_tokens=sample_tokens,
                                                    classes=classes,
                                                    tokenizer=tokenizer, model=model, idx_true=idx_true,
                                                    summary_plot_feat=False,
                                                    summary_plot=False, dependence_plot=False)

        # Save top feat dict
        if SAVE_EXP_DICT:
            with open(f"top_feat_dict_shap_{model_name}.json", "w") as outfile:
                json.dump(top_feat_dict_shap, outfile,indent = 4)

        # Load top feat dict
        # with open(f"top_feat_dict_shap_{model_name}.json", "r") as outfile:
        #     top_feat_dict_shap=json.load(outfile)

        print_top_feature_avast(top_feat_dict_shap,feature_set_path=feature_set_path)


# %% Check feature max len
# import pandas as pd
# import json
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm
#
# # meta_path = "..\\data\\Avast\\subset_100.csv"
# meta_path = "data\\Avast\\public_labels.csv"
#
# meta=pd.read_csv(meta_path)
# d = {
#     'keys': [],
#     'resolved_apis': [],
#     'executed_commands': [],
#     'write_keys': [],
#     'files': [],
#     'read_files': [],
#     "started_services":[],
#     "created_services":[],
#     'write_files': [],
#     'delete_keys': [],
#     'read_keys': [],
#     'delete_files':[],
#     'mutexes': []
# }
# cnt=0
# for i, (filepath, label,date) in enumerate(tqdm(meta[['sha256', 'classification_family','date']].values)):
#     try:
#         with open(f"data\\Avast\\public_small_reports\\{filepath}.json", 'r') as fp:
#             data = json.load(fp)
#         d_tmp=data["behavior"]["summary"]
#         for k in d_tmp.keys():
#             d[k].append(len(d_tmp[k]))
#         cnt += 1
#     except:
#         pass
#
# print(f"N: {cnt}")
# for k in d.keys():
#     print(k)
#     print(f"    Min: {min(d[k])}")
#     print(f"    Max: {max(d[k])}")
#     print(f"    Mean: {np.mean(d[k])}")
#     plt.hist(d[k],bins=100)
#     plt.title(f'{k}')
#     plt.show()