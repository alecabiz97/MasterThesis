import warnings
from keras.layers import LSTM, Bidirectional, Dense, Dropout, Input,Embedding,Attention,Conv1D,MaxPooling1D,CuDNNLSTM
import pickle
import keras
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, plot_roc_curve, classification_report, RocCurveDisplay
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from utils import *
from lime import lime_text
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from time import time



# TF_GPU_ALLOCATOR=cuda_malloc_async
warnings.simplefilter(action='ignore', category=FutureWarning)


def get_neurlux(vocab_size,EMBEDDING_DIM,MAXLEN):
    inp=Input(shape=(MAXLEN))
    x=Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM,input_length=MAXLEN)(inp)
    x = Conv1D(filters=100, kernel_size=4, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=4,data_format="channels_first")(x)
    # x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(32))(x)
    # x,att_score=Attention(name='attention_vec')([x,x], return_attention_scores=True)
    # x=AttentionWithContext()(x)
    # x=Addition(x)
    x = Dense(10, activation="relu")(x)
    x = Dropout(0.25)(x)
    x = Dense(1, activation="sigmoid")(x)
    model=keras.models.Model(inputs=inp,outputs=x)

    model.compile(loss="binary_crossentropy", optimizer='adam', metrics='accuracy')
    return model


def import_data(subset_n_samples,type_split,feature_maxlen=None):
    split_date = "2013-08-09"
    classes = ["Benign", "Malign"]
    df = get_label_date_text_dataframe_dataset1("dataset1\\labels_preproc.csv", feature_maxlen=feature_maxlen)
    # df = pd.read_csv("data.csv")
    # df = pd.read_csv("spamdata_v2.csv")

    df = df.sample(frac=1)  # Shuffle dataset
    if subset_n_samples:
        df = df.iloc[0:subset_n_samples, :].reset_index(drop=True)  # Subset
    print(df.head())

    # Create training, validation and test set
    if type_split == 'random':
        x_tr, x_tmp, y_tr, y_tmp = train_test_split(df['text'], df['label'], test_size=0.2, stratify=df['label'])
    elif type_split == 'time':
        x_tr, y_tr = df[df['date'] < split_date]['text'], df[df['date'] < split_date]['label']
        x_tmp, y_tmp = df[df['date'] >= split_date]['text'], df[df['date'] >= split_date]['label']
    x_val, x_ts, y_val, y_ts = train_test_split(x_tmp, y_tmp, test_size=0.6, stratify=y_tmp)

    print(f"Split train-test: {type_split}")
    print(f"Train size: {len(y_tr)} -- n_classes:{len(set(y_tr))}")
    print(f"Validation size: {len(y_val)} -- n_classes:{len(set(y_val))}")
    print(f"Test size: {len(y_ts)} -- n_classes:{len(set(y_ts))}")

    return x_tr,y_tr,x_val,y_val,x_ts,y_ts,classes


def lime_explanation(x,x_tokens,y,model,feature_maxlen,classes,num_features,feature_stats=False):

    def predict_proba(sample):
        x = tokenizer.texts_to_sequences(sample)
        x = pad_sequences(x, maxlen=MAXLEN, padding='post')

        scores_tmp = model.predict(x)
        scores = []
        for val in scores_tmp:
            scores.append([1 - val, val[0]])
        scores = np.array(scores)

        return scores

    def get_stats(meta_path):
        meta = pd.read_csv(meta_path)
        stats = {"apistats": set(),
                 "apistats_opt": set(),
                 "regkey_opened": set(),
                 "regkey_read": set(),
                 "dll_loaded": set(),
                 "mutex": set()}
        for i, (filepath, label) in enumerate(tqdm(meta[['name', 'label']].values)):
            with open(f"{filepath}.json", 'r') as fp:
                data = json.load(fp)

            x = [preprocessing_data(val) for val in data["behavior"]["apistats"]]
            stats["apistats"].update(x)

            x = [preprocessing_data(val) for val in data["behavior"]["apistats_opt"]]
            stats["apistats_opt"].update(x)

            x = [preprocessing_data(val) for val in data["behavior"]["summary"]["regkey_opened"]]
            stats["regkey_opened"].update(x)

            x = [preprocessing_data(val) for val in data["behavior"]["summary"]["regkey_read"]]
            stats["regkey_read"].update(x)

            x = [preprocessing_data(val) for val in data["behavior"]["summary"]["dll_loaded"]]
            stats["dll_loaded"].update(x)

            x = [preprocessing_data(val) for val in data["behavior"]["summary"]["mutex"]]
            stats["mutex"].update(x)
        return stats


    MAXLEN = sum(feature_maxlen.values())



    if feature_stats:
        meta_path = "dataset1\\labels_preproc.csv"
        stats = get_stats(meta_path)
        cnt_api = 0
        cnt_api_opt = 0
        cnt_regop = 0
        cnt_regre = 0
        cnt_dll = 0
        cnt_mutex = 0

    for idx in range(len(y)):
        sample = x.iloc[idx]
        y_sample = y.iloc[idx]
        # print(f"Idx: {idx}")
        y_pred = model.predict(tf.constant([x_tokens[idx]])).squeeze().round().astype(int)
        print(f"Label sample: {classes[y_sample]}")
        print(f"Predicted: {classes[y_pred]}")
        explainer = lime_text.LimeTextExplainer(class_names=classes)
        explanation = explainer.explain_instance(sample, classifier_fn=predict_proba, num_features=num_features, top_labels=1)

        explanation.save_to_file(f'exp_{idx}.html')
        print("Explanation file created")

        if feature_stats:
            for val, importance in explanation.as_list(label=explanation.available_labels()[0]):
                if val in stats["apistats"]:
                    cnt_api += 1
                elif val in stats["apistats_opt"]:
                    cnt_api_opt += 1
                elif val in stats["regkey_opened"]:
                    cnt_regop += 1
                elif val in stats["regkey_read"]:
                    cnt_regre += 1
                elif val in stats["dll_loaded"]:
                    cnt_dll += 1
                elif val in stats["mutex"]:
                    cnt_mutex += 1

    if feature_stats:
        print(f"API: {cnt_api}")
        print(f"API OPT: {cnt_api_opt}")
        print(f"REGOP: {cnt_regop}")
        print(f"REGRE: {cnt_regre}")
        print(f"DLL: {cnt_dll}")
        print(f"MUTEX: {cnt_mutex}")

if __name__ == '__main__':

    # Hyperparameters
    feature_maxlen = {
        # "apistats": 200,
        "apistats_opt": 300,
        "regkey_opened": 500,
        "regkey_read": 500,
        "dll_loaded": 200,
        "mutex": 50
    }
    MAXLEN = sum(feature_maxlen.values())
    EMBEDDING_DIM=256 # 256
    BATCH_SIZE = 50
    EPOCHS = 10 # 10
    LEARNING_RATE = 0.0001
    TYPE_SPLIT='random' # 'time' or 'random'
    SUBSET_N_SAMPLES=1000 # if None takes all data
    LIME_EXPLANATION=False

    # Import data
    x_tr, y_tr, x_val, y_val, x_ts, y_ts, classes = import_data(subset_n_samples=SUBSET_N_SAMPLES,
                                                                type_split=TYPE_SPLIT,feature_maxlen=feature_maxlen)
    n_classes = len(set(y_tr))

    # Tokenize
    x_tr_tokens, x_val_tokens, x_ts_tokens, vocab_size, tokenizer = tokenize_data(x_tr, x_val, x_ts, maxlen=MAXLEN)
    print(f"Vocab size: {vocab_size}")

    # Save tokenizer
    with open(f'tokenizer_detection.pickle', 'wb') as fp:
        pickle.dump(tokenizer, fp)

    # Model definition
    model = get_neurlux(vocab_size, EMBEDDING_DIM, MAXLEN)
    print(model.summary())

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    mc = ModelCheckpoint(f'./model_detection.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
    print("START TRAINING")

    history_embedding = model.fit(tf.constant(x_tr_tokens), tf.constant(y_tr),
                                  epochs=EPOCHS, batch_size=BATCH_SIZE,
                                  validation_data=(tf.constant(x_val_tokens), tf.constant(y_val)),
                                  verbose=1, callbacks=[es, mc])


    #Test
    print("TEST")

    # print(classification_report(y_pred, y_ts))
    print(f"Test accuracy: {model.evaluate(x_ts_tokens, np.array(y_ts),verbose=False)[1]}")
    print(f"Train accuracy: {model.evaluate(x_tr_tokens, np.array(y_tr),verbose=False)[1]}")
    # print(confusion_matrix(y_ts,y_pred))

    scores=model.predict(tf.constant(x_ts_tokens),verbose=False).squeeze()
    y_pred=scores.round().astype(int)

    # ROC curve
    auc=roc_auc_score(y_ts,scores)
    print(f"AUC: {auc}")
    fpr, tpr, thresh=roc_curve(y_ts,scores)
    plt.plot(fpr,tpr,label=f'AUC = {round(auc,2)})')
    plt.legend()
    plt.show()
    # RocCurveDisplay.from_predictions(y_ts,scores)

    # Confusion matrix
    plot_confusion_matrix(y_true=y_ts, y_pred=y_pred, classes=classes)

    # LIME Explanation
    if LIME_EXPLANATION:
        x=x_ts[0:1]
        x_tokens=x_ts_tokens[0:1]
        y=y_ts[0:1]
        lime_explanation(x=x,x_tokens=x_tokens,y=y,model=model,feature_maxlen=feature_maxlen,
                         classes=classes,num_features=10,feature_stats=False)

# %%

# meta = pd.read_csv("dataset1\\labels_preproc.csv")
# x1,x2,x3,x4,x5,x6=[],[],[],[],[],[]
#
# for i, (filepath, label,date) in enumerate(tqdm(meta[['name', 'label','date']].values)):
#     with open(f"{filepath}.json", 'r') as fp:
#         data = json.load(fp)
#     a=data['behavior']['apistats_opt']
#     # tokenizer = Tokenizer(num_words=10000)
#     # tokenizer.fit_on_texts(a)
#     # b = tokenizer.texts_to_sequences(a)
#     x1.append(len(data["behavior"]['apistats']))
#     x2.append(len(data["behavior"]["apistats_opt"]))
#     x3.append(len(data["behavior"]["summary"]["regkey_opened"]))
#     x4.append(len(data["behavior"]["summary"]["regkey_read"]))
#     x5.append(len(data["behavior"]["summary"]["dll_loaded"]))
#     x6.append(len(data["behavior"]["summary"]["mutex"]))
#
#
#
# print("API")
# print(np.min(x1))
# print(np.max(x1))
# print(np.mean(x1))
# plt.hist(x1)
# plt.title('API')
# plt.show()
#
# print("API OPT")
# print(np.min(x2))
# print(np.max(x2))
# print(np.mean(x2))
# plt.hist(x2)
# plt.title('API OPT')
# plt.show()
#
# print("REGOP")
# print(np.min(x3))
# print(np.max(x3))
# print(np.mean(x3))
# plt.hist(x3)
# plt.title('REGOP')
# plt.show()
#
# print("REGRE")
# print(np.min(x4))
# print(np.max(x4))
# print(np.mean(x4))
# plt.hist(x4)
# plt.title('REGRE')
# plt.show()
#
# print("DLL")
# print(np.min(x5))
# print(np.max(x5))
# print(np.mean(x5))
# plt.hist(x5)
# plt.title('DLL')
# plt.show()
#
# print("MUTEX")
# print(np.min(x6))
# print(np.max(x6))
# print(np.mean(x6))
# plt.hist(x6)
# plt.title('MUTEX')
# plt.show()












