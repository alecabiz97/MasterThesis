import warnings
from keras.layers import LSTM, Bidirectional, Dense, Dropout, Input,Embedding,Conv1D,MaxPooling1D,CuDNNLSTM
import pickle
import keras
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, plot_roc_curve,\
    classification_report, RocCurveDisplay, DetCurveDisplay
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from utils import *
from lime import lime_text
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from time import time
from copy import deepcopy
from keras import constraints, initializers, regularizers
import keras.backend as K
from attention import Attention


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


def lime_explanation(x,x_tokens,y,model,feature_maxlen,classes,num_features,feature_stats=False):

    def predict_proba(sample):
        x = tokenizer.texts_to_sequences(sample)
        x = pad_sequences(x, maxlen=MAXLEN, padding='post')

        scores_tmp = model.predict(x)
        scores = []
        for val in scores_tmp:
            scores.append([1 - val[0], val[0]])
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
        meta_path = "data\\dataset1\\labels_preproc.csv"
        stats = get_stats(meta_path)
        cnt_api = 0
        cnt_api_opt = 0
        cnt_regop = 0
        cnt_regre = 0
        cnt_dll = 0
        cnt_mutex = 0
    exps=[]
    for idx in range(len(y)):
        sample = x.iloc[idx]
        y_sample = y.iloc[idx]
        # print(f"Idx: {idx}")
        y_pred = model.predict(tf.constant([x_tokens[idx]])).squeeze().round().astype(int)
        print(f"Label sample: {classes[y_sample]}")
        print(f"Predicted: {classes[y_pred]}")
        explainer = lime_text.LimeTextExplainer(class_names=classes)
        explanation = explainer.explain_instance(sample, classifier_fn=predict_proba, num_features=num_features, top_labels=1)
        exps.append(deepcopy(explanation))

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

    return exps

if __name__ == '__main__':

    # Hyperparameters
    feature_maxlen = {
        # "apistats": 500,
        "apistats_opt": 500,
        # "regkey_opened": 500,
        # "regkey_read": 500,
        # "dll_loaded": 200,
        # "mutex": 50
    }
    MAXLEN = sum(feature_maxlen.values())
    EMBEDDING_DIM=256 # 256
    BATCH_SIZE = 50
    EPOCHS = 15 # 30
    LEARNING_RATE = 0.0001
    TYPE_SPLIT='random' # 'time' or 'random'
    SPLIT_DATE="2013-08-09"
    SUBSET_N_SAMPLES=1000 # if None takes all data
    WITH_ATTENTION=True
    TRAINING=False
    meta_path="..\\data\\dataset1\\labels_preproc.csv"
    model_name="Neurlux_detection"
    classes = ["Benign", "Malign"]


    # Explanation
    LIME_EXPLANATION = False
    TOPK_FEATURE=10
    N_SAMPLES_EXP=10

    # Import data
    df, classes = import_data(meta_path=meta_path,subset_n_samples=SUBSET_N_SAMPLES,feature_maxlen=feature_maxlen,
                              callback=get_label_date_text_dataframe_dataset1,classes=classes)
    n_classes = len(classes)

    # Split Train-Test-Validation
    x_tr, y_tr, x_val, y_val, x_ts, y_ts= split_train_val_test_dataframe(df, type_split=TYPE_SPLIT, split_date=SPLIT_DATE, tr=0.8)

    # Tokenize
    x_tr_tokens, x_val_tokens, x_ts_tokens, vocab_size, tokenizer = tokenize_data(x_tr, x_val, x_ts, maxlen=MAXLEN)
    print(f"Vocab size: {vocab_size}")

    # Save tokenizer
    #with open(f'tokenizer_detection.pickle', 'wb') as fp:
    #    pickle.dump(tokenizer, fp)

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

    #fig, axs = plt.subplots(1, 2, figsize=(15, 5))
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

    # LIME Explanation
    if LIME_EXPLANATION:
        x=x_ts[0:N_SAMPLES_EXP]
        x_tokens=x_ts_tokens[0:N_SAMPLES_EXP]
        y=y_ts[0:N_SAMPLES_EXP]

        explanations = lime_explanation(x=x, x_tokens=x_tokens, y=y, model=model, feature_maxlen=feature_maxlen,
                                        classes=classes, num_features=TOPK_FEATURE, feature_stats=False)

        # top_feat_lime=[val[0] for val in explanation.as_list(label=explanation.available_labels()[0])]
        top_feat_lime = [[val[0] for val in exp.as_list(label=exp.available_labels()[0])] for exp in explanations]


        # Attention
        if WITH_ATTENTION:
            top_feat_att=get_top_feature_attention(attention_model,tokenizer,x_tokens,topk=TOPK_FEATURE)


            for i in range(N_SAMPLES_EXP):
                cnt = 0
                for val in top_feat_att[i]:
                    if val in top_feat_lime[i]:
                        #print(val)
                        cnt += 1
                print(f"[Sample {i}] Common feature Attention/LIME: {cnt}/{TOPK_FEATURE}")





# %%

# meta = pd.read_csv("data\\dataset1\\labels_preproc.csv")
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















