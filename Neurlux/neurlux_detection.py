import warnings

import matplotlib.pyplot as plt
from keras.layers import LSTM, Bidirectional, Dense, Dropout, Input,Embedding,Conv1D,MaxPooling1D,CuDNNLSTM
import pickle
import keras
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, plot_roc_curve,\
    classification_report, RocCurveDisplay, DetCurveDisplay
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from utils import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
import shap

# TF_GPU_ALLOCATOR=cuda_malloc_async
warnings.simplefilter(action='ignore', category=FutureWarning)

def get_neurlux(vocab_size,EMBEDDING_DIM,MAXLEN,with_attention=False):
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
    x = Dense(1, activation="sigmoid")(x)
    model=keras.models.Model(inputs=inp,outputs=x)
    model.compile(loss="binary_crossentropy", optimizer='adam', metrics='accuracy')

    if with_attention:
        attention_model = keras.models.Model(inputs=inp, outputs=attention_out)
        return model,attention_model
    else:
        return model, None


if __name__ == '__main__':

    # Hyperparameters
    feature_maxlen = {
        # "apistats": 200, #200
        "apistats_opt": 200, #200
        "regkey_opened": 500, #500
        "regkey_read": 500,
        # "dll_loaded": 120,
        # "mutex": 100,
        #  "regkey_deleted": 100,
        #  "regkey_written": 50,
         # "file_deleted": 100,
         # "file_failed": 50,
         # "file_read": 50,
         # "file_opened": 50,
         # "file_exists": 50,
         # "file_written": 50,
         # "file_created": 50
    }

    MAXLEN = sum(feature_maxlen.values())
    EMBEDDING_DIM=256 # 256
    BATCH_SIZE = 50
    EPOCHS = 20 # 30
    LEARNING_RATE = 0.0001
    TYPE_SPLIT='random' # 'time' or 'random'
    SPLIT_DATE_VAL_TS="2013-08-09"
    SPLIT_DATE_TR_VAL="2012-12-09"
    SUBSET_N_SAMPLES=None # if None takes all data
    WITH_ATTENTION=True
    TRAINING=True
    meta_path="..\\data\\dataset1\\labels_preproc.csv"
    model_name="Neurlux_detection"
    classes = ["Benign", "Malign"]

    # Shap Explanation
    SHAP=True

    # Lime Explanation
    LIME_EXPLANATION = True
    TOPK_FEATURE=10
    N_SAMPLES_EXP=1

    # Import data
    df = import_data(meta_path=meta_path,subset_n_samples=SUBSET_N_SAMPLES,feature_maxlen=feature_maxlen,
                              callback=get_label_date_text_dataframe_dataset1)
    n_classes = len(classes)

    # Split Train-Test-Validation
    x_tr, y_tr, x_val, y_val, x_ts, y_ts= split_train_val_test_dataframe(df, type_split=TYPE_SPLIT,
                                                                         split_dates=[SPLIT_DATE_VAL_TS,SPLIT_DATE_TR_VAL], tr=0.8)

    # Tokenize
    x_tr_tokens, x_val_tokens, x_ts_tokens, vocab_size, tokenizer = tokenize_data(x_tr, x_val, x_ts, maxlen=MAXLEN)
    print(f"Vocab size: {vocab_size}")

    # Model definition
    model,attention_model = get_neurlux(vocab_size, EMBEDDING_DIM, MAXLEN,with_attention=WITH_ATTENTION)
    print(model.summary())

    if TRAINING:
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        mc = ModelCheckpoint(f'./trained_models/dataset1/{model_name}.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
        print("START TRAINING")

        history_embedding = model.fit(tf.constant(x_tr_tokens), tf.constant(y_tr),
                                      epochs=EPOCHS, batch_size=BATCH_SIZE,
                                      validation_data=(tf.constant(x_val_tokens), tf.constant(y_val)),
                                      verbose=1, callbacks=[es, mc])
    else:
        model.load_weights(f"./trained_models/dataset1/{model_name}.h5")

    #Test
    print("TEST")

    # print(classification_report(y_pred, y_ts))
    print(f"Train accuracy: {model.evaluate(x_tr_tokens, np.array(y_tr), verbose=False)[1]}")
    print(f"Test accuracy: {model.evaluate(x_ts_tokens, np.array(y_ts),verbose=False)[1]}")

    # print(confusion_matrix(y_ts,y_pred))
    scores=model.predict(tf.constant(x_ts_tokens),verbose=False).squeeze()
    y_pred=scores.round().astype(int)

    # Plot ROC and DET curves
    RocCurveDisplay.from_predictions(y_ts, scores,name="Neurlux")
    plt.title("Receiver Operating Characteristic (ROC) curves")
    plt.grid(linestyle="--")
    plt.legend(loc='lower right')
    plt.show()

    DetCurveDisplay.from_predictions(y_ts, scores,name="Neurlux")
    plt.title("Detection Error Tradeoff (DET) curves")
    plt.grid(linestyle="--")
    plt.legend(loc='upper right')
    plt.show()

    # Confusion matrix
    plot_confusion_matrix(y_true=y_ts, y_pred=y_pred, classes=classes)


# %% Explanation

    # hash = "00b26c8964bf6c20183d13867e6dbcb0"
    # hash = "00a0f5fe1ba0102ed789b2aa85c3e316"
    # hash = "1a9ab9e924a6856d642bbe88064e4236" # tesla-crypt
    # hash = "000ed458b787e6841c103a694d11962c"
    # hash = "000f6d35ea5397e40ffaa7931a9ae1d3"
    hash = "00aae6c41996b3d1631f605ad783f1f8" #

    with open(f"..\\data\\dataset1\\mal_preproc\\{hash}.json", "r") as fp:
        data = json.load(fp)

    text = []
    for feat in feature_maxlen.keys():
        x = data["behavior"][feat]
        text.append(x[0:min(len(x), feature_maxlen[feat])])

    # LIME Explanation
    if LIME_EXPLANATION:
        # x=x_ts[0:N_SAMPLES_EXP]
        # x_tokens=x_ts_tokens[0:N_SAMPLES_EXP]
        # y=y_ts[0:N_SAMPLES_EXP]
        # idx=3
        # x = x_ts.iloc[idx:idx + 1]
        # x_tokens = x_ts_tokens[idx:idx + 1]
        # y = y_ts.iloc[idx:idx + 1]

        # hash = "0fb068a699abfe607e4e9f99c1aad3ab"
        x = pd.Series(preprocessing_data(str(text)))
        x_tokens = tokenizer.texts_to_sequences(x)
        x_tokens = pad_sequences(x_tokens, maxlen=MAXLEN, padding='post')
        y = pd.Series(1)

        print(x)
        explanations = lime_explanation_dataset1(x=x, x_tokens=x_tokens, y=y, model=model,tokenizer=tokenizer,
                                                 feature_maxlen=feature_maxlen,classes=classes,
                                                 num_features=TOPK_FEATURE, feature_stats=False)

        # top_feat_lime=[val[0] for val in explanation.as_list(label=explanation.available_labels()[0])]
        # top_feat_lime = [[val[0] for val in exp.as_list(label=exp.available_labels()[0])] for exp in explanations]
        #
        # # Attention
        # if WITH_ATTENTION:
        #     top_feat_att=get_top_feature_attention(attention_model,tokenizer,x_tokens,topk=TOPK_FEATURE)
        #     for i in range(N_SAMPLES_EXP):
        #         cnt = 0
        #         for val in top_feat_att[i]:
        #             if val in top_feat_lime[i]:
        #                 #print(val)
        #                 cnt += 1
        #         print(f"[Sample {i}] Common feature Attention/LIME: {cnt}/{TOPK_FEATURE}")

# %% SHAP Explanation
    if SHAP:
        explainer = shap.KernelExplainer(model.predict, np.zeros((1,x_tr_tokens.shape[1])))
        # explainer = shap.KernelExplainer(model.predict, shap.sample(x_tr_tokens,100))

        sample = preprocessing_data(str(text))
        sample_tokens = tokenizer.texts_to_sequences([sample])
        sample_tokens = pad_sequences(sample_tokens, maxlen=MAXLEN, padding='post')
        y_true="Malign"

        # idx = 3
        # sample = x_ts.iloc[idx]
        # sample_tokens = x_ts_tokens[idx:idx + 1]
        # y_true = classes[y_ts.iloc[idx]]

        y_pred = classes[model.predict(tf.constant(sample_tokens),verbose=False).squeeze().round().astype(int)]
        print(f"Score: {model.predict(sample_tokens)}")
        print(f"True: {y_true}")
        print(f"Predicted: {y_pred}")

        shap_values = explainer.shap_values(sample_tokens, nsamples="auto")
        shap.initjs()

        text = []
        for i in sample_tokens[0]:
            if i != 0:
                text.append(tokenizer.index_word[i])
            else:
                text.append("PAD")
        print(sample)
        print(text)

        # p = shap.force_plot(explainer.expected_value[0], shap_values[0], text)
        # p.matplotlib(figsize=(15, 5), show=True, text_rotation=None)

        shap.summary_plot(shap_values[0], sample_tokens, feature_names=text, class_names=classes,plot_size=(20.,5.))

        # Dependence plot
        feature=text[np.argmax(shap_values[0])]
        id=tokenizer.word_index[feature]

        fig,ax=plt.subplots(1,figsize=(10,5))
        shap.partial_dependence_plot(feature, model.predict, sample_tokens, ice=False,
                                     model_expected_value=True, feature_expected_value=True, feature_names=text,
                                     xmin=id-10, xmax=id + 10,ax=ax)
        print(model.predict(sample_tokens))


# %% Check feature max len
# import pandas as pd
# import json
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm
#
# # meta = pd.read_csv("data\\dataset1\\labels.csv")
# meta = pd.read_csv("..\\data\\dataset1\\labels_preproc.csv")
#
# d = {
#         "apistats": [],
#         "apistats_opt": [],
#         "regkey_opened": [],
#         "regkey_read": [],
#         "dll_loaded": [],
#         "mutex": [],
#         "regkey_deleted": [],
#         "regkey_written": [],
#         "file_deleted": [],
#         "file_failed": [],
#         "file_read": [],
#         "file_opened": [],
#         "file_exists": [],
#         "file_written": [],
#         "file_created": []
# }
# cnt=0
# for i, (filepath, label,date) in enumerate(tqdm(meta[['name', 'label','date']].values)):
#     try:
#         with open(f"data\\{filepath}.json", 'r') as fp:
#             data = json.load(fp)
#         d_tmp=data["behavior"]
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

# %%
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# from utils import *
#
# ben_files=getListOfFiles("data/dataset1/ben_reports")
# mal_files=getListOfFiles("data/dataset1/mal_reports")
# # keys=set()
# info_keys={"file_deleted":[],
#     "wmi_query":[],
#     "mutex":[],
#     "file_failed":[],
#     "file_read":[],
#     "directory_created":[],
#     "directory_enumerated":[],
#     "command_line":[],
#     "regkey_deleted":[],
#     "file_opened":[],
#     "fetches_url":[],
#     "file_recreated":[],
#     "resolves_host":[],
#     "file_exists":[],
#     "regkey_opened":[],
#     "connects_host":[],
#     "file_written":[],
#     "guid":[],
#     "file_created":[],
#     "dll_loaded":[],
#     "file_moved":[],
#     "file_copied":[],
#     "regkey_written":[],
#     "directory_removed":[],
#     "connects_ip":[]
#       }
# cnt=0
# for fname in tqdm(ben_files+mal_files):
#     try:
#         with open(fname,'r') as fp:
#             data=json.load(fp)
#         # keys.update(list(data['behavior']['summary'].keys()))
#         summary_keys=list(data['behavior']['summary'].keys())
#         for k in info_keys.keys():
#             if k in summary_keys:
#                 n=len(data['behavior']['summary'][k])
#                 info_keys[k].append(n)
#         cnt+=1
#     except:
#         pass
#
# for k,val in info_keys.items():
#     print(k, len(val))
#     print(f"    Min: {min(val)}")
#     print(f"    Max: {max(val)}")
#     print(f"    Mean: {np.mean(val)}")
#     plt.hist(val, bins=100)
#     plt.title(f'{k}')
#     plt.show()
#
# print(f"\nN_sample: {cnt}")

# %%
# files=getListOfFiles("../data/dataset1/mal_reports")
# files2=getListOfFiles("../data/dataset1/mal_preproc")
# hashes_preproc=[files2[i].split('\\')[1].split('.')[0] for i in range(len(files2))]
# fp1=open("hashes.txt","w")
# for f in tqdm(files):
#     try:
#         if f.split('\\')[1].split('.')[0] in hashes_preproc:
#             with open(f,'r') as fp:
#                 data=json.load(fp)
#             hash=data["target"]["file"]["sha256"]
#             fp1.write(f"{hash}\n")
#     except:
#         pass
#
# fp1.close()
# print("Done")









