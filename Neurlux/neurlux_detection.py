import warnings

import matplotlib.pyplot as plt
from keras.layers import LSTM, Bidirectional, Dense, Dropout, Input, Embedding, Conv1D, MaxPooling1D, CuDNNLSTM
import pickle
import keras
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, plot_roc_curve, \
    classification_report, RocCurveDisplay, DetCurveDisplay
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from utils import *
from tensorflow.keras.preprocessing.sequence import pad_sequences
import shap

# TF_GPU_ALLOCATOR=cuda_malloc_async
warnings.simplefilter(action='ignore', category=FutureWarning)


def get_neurlux(vocab_size, EMBEDDING_DIM, MAXLEN, with_attention=False):
    inp = Input(shape=(MAXLEN))
    x = Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=MAXLEN)(inp)
    x = Conv1D(filters=100, kernel_size=4, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=4, data_format="channels_first")(x)
    # x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
    if with_attention:
        x = Bidirectional(CuDNNLSTM(32, return_sequences=True))(x)
        x, attention_out = AttentionWithContext()(x)
    else:
        x = Bidirectional(CuDNNLSTM(32))(x)

    x = Dense(10, activation="relu")(x)
    x = Dropout(0.25)(x)
    x = Dense(1, activation="sigmoid")(x)
    model = keras.models.Model(inputs=inp, outputs=x)
    model.compile(loss="binary_crossentropy", optimizer='adam', metrics='accuracy')

    if with_attention:
        attention_model = keras.models.Model(inputs=inp, outputs=attention_out)
        return model, attention_model
    else:
        return model, None


if __name__ == '__main__':

    # Hyperparameters
    feature_maxlen = {
        "apistats": 200,  # 200
        "apistats_opt": 200,  # 200
        "regkey_opened": 500,  # 500
        "regkey_read": 500,
        "dll_loaded": 120,
        "mutex": 100,
        "regkey_deleted": 100,
        "regkey_written": 50,
        "file_deleted": 100,
        "file_failed": 50,
        "file_read": 50,
        "file_opened": 50,
        "file_exists": 50,
        "file_written": 50,
        "file_created": 50
    }

    MAXLEN = sum(feature_maxlen.values())
    EMBEDDING_DIM = 256  # 256
    BATCH_SIZE = 50
    EPOCHS = 30  # 30
    LEARNING_RATE = 0.0001
    TYPE_SPLIT = 'time'  # 'time' or 'random'
    SPLIT_DATE_VAL_TS = "2013-08-09"
    SPLIT_DATE_TR_VAL = "2012-12-09"
    SUBSET_N_SAMPLES = None  # if None takes all data
    WITH_ATTENTION = True
    TRAINING = False
    meta_path = "..\\data\\dataset1\\labels_preproc.csv"
    # model_name = "Neurlux_detection"
    model_name=f"neurlux_detection_all_{EPOCHS}_{TYPE_SPLIT}"
    classes = ["Benign", "Malign"]

    # Explanation
    SHAP = True
    LIME = True
    EXP_MODE = 'multi'  # single or multi
    TOPK_FEATURE = 10
    N_SAMPLES_EXP = 50
    SAVE_EXP_DICT = True
    feature_set_path = "../data/dataset1/dataset1_feature_set.json"

    # Import data
    df = import_data(meta_path=meta_path, subset_n_samples=SUBSET_N_SAMPLES, feature_maxlen=feature_maxlen,
                     callback=get_label_date_text_dataframe_dataset1)
    n_classes = len(classes)

    # Split Train-Test-Validation
    x_tr, y_tr, x_val, y_val, x_ts, y_ts = split_train_val_test_dataframe(df, type_split=TYPE_SPLIT,
                                                                          split_dates=[SPLIT_DATE_VAL_TS,
                                                                                       SPLIT_DATE_TR_VAL], tr=0.8)

    # Tokenize
    x_tr_tokens, x_val_tokens, x_ts_tokens, vocab_size, tokenizer = tokenize_data(x_tr, x_val, x_ts, maxlen=MAXLEN)
    print(f"Vocab size: {vocab_size}")

    # Model definition
    model, attention_model = get_neurlux(vocab_size, EMBEDDING_DIM, MAXLEN, with_attention=WITH_ATTENTION)
    print(model.summary())

    if TRAINING:
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        mc = ModelCheckpoint(f'./trained_models/dataset1/{model_name}.h5', monitor='val_accuracy', mode='max',
                             verbose=1, save_best_only=True)
        print("START TRAINING")

        history_embedding = model.fit(tf.constant(x_tr_tokens), tf.constant(y_tr),
                                      epochs=EPOCHS, batch_size=BATCH_SIZE,
                                      validation_data=(tf.constant(x_val_tokens), tf.constant(y_val)),
                                      verbose=1, callbacks=[es, mc])
    else:
        model.load_weights(f"./trained_models/dataset1/{model_name}.h5")

    # Test
    print("TEST")

    # print(classification_report(y_pred, y_ts))
    print(f"Train accuracy: {model.evaluate(x_tr_tokens, np.array(y_tr), verbose=False)[1]}")
    print(f"Test accuracy: {model.evaluate(x_ts_tokens, np.array(y_ts), verbose=False)[1]}")

    # print(confusion_matrix(y_ts,y_pred))
    scores = model.predict(tf.constant(x_ts_tokens), verbose=False).squeeze()
    y_pred = scores.round().astype(int)

    # Save scores data
    d = {'scores': scores.tolist(), 'y': y_ts.to_list()}
    json_object = json.dumps(d, indent=4)
    with open(f"Neurlux_scores_y_All_{TYPE_SPLIT}.json", "w") as outfile:
        outfile.write(json_object)

    # Plot ROC and DET curves
    RocCurveDisplay.from_predictions(y_ts, scores, name="Neurlux")
    plt.title("Receiver Operating Characteristic (ROC) curves")
    plt.grid(linestyle="--")
    plt.legend(loc='lower right')
    plt.show()

    DetCurveDisplay.from_predictions(y_ts, scores, name="Neurlux")
    plt.title("Detection Error Tradeoff (DET) curves")
    plt.grid(linestyle="--")
    plt.legend(loc='upper right')
    plt.show()

    # Confusion matrix
    plot_confusion_matrix(y_true=y_ts, y_pred=y_pred, classes=classes)

    # %% Explanation
    hash = "00aae6c41996b3d1631f605ad783f1f8"
    # hash = "00b26c8964bf6c20183d13867e6dbcb0"
    # hash = "00a0f5fe1ba0102ed789b2aa85c3e316"
    # hash = "1a9ab9e924a6856d642bbe88064e4236" # tesla-crypt
    # hash = "000ed458b787e6841c103a694d11962c"
    # hash = "000f6d35ea5397e40ffaa7931a9ae1d3"

    # LIME Explanation
    if LIME:

        # Explanation of a single sample given is hash
        if EXP_MODE == "single":

            with open(f"..\\data\\dataset1\\mal_preproc\\{hash}.json", "r") as fp:
                data = json.load(fp)

            text = []
            for feat in feature_maxlen.keys():
                x = data["behavior"][feat]
                text.append(x[0:min(len(x), feature_maxlen[feat])])

            x = pd.Series(preprocessing_data(str(text)))
            x_tokens = tokenizer.texts_to_sequences(x)
            x_tokens = pad_sequences(x_tokens, maxlen=MAXLEN, padding='post')
            y = pd.Series(1)

        # Explanation of multiple samples
        elif EXP_MODE == "multi":
            # Subset
            x = []
            x_tokens = np.zeros(shape=(N_SAMPLES_EXP * len(classes), x_ts_tokens.shape[1]),dtype=int)
            y = np.zeros(shape=(N_SAMPLES_EXP * len(classes)), dtype=int)
            for i in range(len(classes)):
                idx = (y_ts == i).to_numpy()
                x_tokens[i * N_SAMPLES_EXP:i * N_SAMPLES_EXP + N_SAMPLES_EXP, :] = x_ts_tokens[idx, :][0:N_SAMPLES_EXP,
                                                                                   :]
                y[i * N_SAMPLES_EXP:i * N_SAMPLES_EXP + N_SAMPLES_EXP] = y_ts.values[idx][0:N_SAMPLES_EXP]
                x.extend(x_ts.values[idx][0:N_SAMPLES_EXP].tolist())

            x = pd.Series(x)
            y = pd.Series(y.tolist())

        top_feat_dict_lime = lime_explanation_dataset1(x=x, x_tokens=x_tokens, y=y, model=model, tokenizer=tokenizer,
                                                 feature_maxlen=feature_maxlen, classes=classes,
                                                 num_features=TOPK_FEATURE, save_html=False)
        # Save top feat dict
        if SAVE_EXP_DICT:
            with open(f"top_feat_dict_lime_{model_name}.json", "w") as outfile:
                json.dump(top_feat_dict_lime, outfile, indent=4)
        # Print most frequents feature
        print("\nLIME RESULTS")
        print_top_feature_dataset1(top_feat_dict_lime,feature_set_path=feature_set_path)

    # %% SHAP Explanation
    if SHAP:
        explainer = shap.KernelExplainer(model.predict, np.zeros((1, x_tr_tokens.shape[1])))
        # explainer = shap.KernelExplainer(model.predict, shap.sample(x_tr_tokens,1))

        # Explanation of a single sample given is hash
        if EXP_MODE == "single":
            with open(f"..\\data\\dataset1\\mal_preproc\\{hash}.json", "r") as fp:
                data = json.load(fp)

            text = []
            for feat in feature_maxlen.keys():
                x = data["behavior"][feat]
                text.append(x[0:min(len(x), feature_maxlen[feat])])

            sample = preprocessing_data(str(text))
            sample_tokens = tokenizer.texts_to_sequences([sample])
            sample_tokens = pad_sequences(sample_tokens, maxlen=MAXLEN, padding='post')
            # y_true="Malign"
            idx_true=np.array(1)

        # Explanation of multiple samples
        elif EXP_MODE == "multi":

            # Subset
            sample_tokens = np.zeros(shape=(N_SAMPLES_EXP * len(classes), x_ts_tokens.shape[1]),dtype=int)
            idx_true = np.zeros(shape=(N_SAMPLES_EXP * len(classes)),dtype=int)
            for i in range(len(classes)):
                idx = (y_ts == i).to_numpy()
                sample_tokens[i * N_SAMPLES_EXP:i * N_SAMPLES_EXP + N_SAMPLES_EXP, :] = x_ts_tokens[idx, :][
                                                                                        0:N_SAMPLES_EXP, :]
                idx_true[i * N_SAMPLES_EXP:i * N_SAMPLES_EXP + N_SAMPLES_EXP] = y_ts.values[idx][0:N_SAMPLES_EXP]

            print(idx_true.shape, sample_tokens.shape)

        top_feat_dict_shap=shap_explanation_dataset1(explainer=explainer, sample_tokens=sample_tokens, id_true=idx_true, classes=classes,
                                  tokenizer=tokenizer, model=model, summary_plot=False, dependence_plot=False,
                                  topk=TOPK_FEATURE)
        # Save top feat dict
        if SAVE_EXP_DICT:
            with open(f"top_feat_dict_shap_{model_name}.json", "w") as outfile:
                json.dump(top_feat_dict_shap, outfile, indent=4)

        # Print most frequents feature
        print("\nSHAP RESULTS")
        print_top_feature_dataset1(top_feat_dict_shap,feature_set_path=feature_set_path)




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
# import pandas as pd
# import json
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm
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

# %% Hash file
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
