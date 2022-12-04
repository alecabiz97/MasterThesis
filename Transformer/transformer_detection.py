import warnings
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, plot_roc_curve,\
    classification_report, RocCurveDisplay, DetCurveDisplay
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from utils import *

from keras import layers


# TF_GPU_ALLOCATOR=cuda_malloc_async
warnings.simplefilter(action='ignore', category=FutureWarning)


def get_transformer_model(MAXLEN,vocab_size,EMBEDDING_DIM=256,num_heads=2,ff_dim=32):
    # Transformer (funziona)
    inp = layers.Input(shape=(MAXLEN,))
    x = TokenAndPositionEmbedding(MAXLEN, vocab_size, EMBEDDING_DIM)(inp)
    # x = layers.Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=MAXLEN)(inp)
    x= TransformerBlock(EMBEDDING_DIM, num_heads, ff_dim)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs=inp, outputs=outputs)
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


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
    TRAINING=False
    meta_path="..\\data\\dataset1\\labels_preproc.csv"
    model_name="Transformer_detection"
    classes = ["Benign", "Malign"]


    # Explanation
    LIME_EXPLANATION = True
    TOPK_FEATURE=10
    N_SAMPLES_EXP=1

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
    model = get_transformer_model(MAXLEN, vocab_size, EMBEDDING_DIM)
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

    RocCurveDisplay.from_predictions(y_ts, scores,name="Transformer")
    plt.title("Receiver Operating Characteristic (ROC) curves")
    plt.grid(linestyle="--")
    plt.legend(loc='lower right')
    plt.show()

    DetCurveDisplay.from_predictions(y_ts, scores,name="Transformer")
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

        explanations = lime_explanation_dataset1(x=x, x_tokens=x_tokens, y=y, model=model,tokenizer=tokenizer,
                                                 feature_maxlen=feature_maxlen,classes=classes,
                                                 num_features=TOPK_FEATURE, feature_stats=True)

        # top_feat_lime=[val[0] for val in explanation.as_list(label=explanation.available_labels()[0])]
        top_feat_lime = [[val[0] for val in exp.as_list(label=exp.available_labels()[0])] for exp in explanations]

# %% Create a json file with dataset1 feature set
# def get_feature_set(meta_path):
#     meta = pd.read_csv(meta_path)
#
#     feature_set = {"apistats": set(),
#              "apistats_opt": set(),
#              "regkey_opened": set(),
#              "regkey_read": set(),
#              "dll_loaded": set(),
#              "mutex": set()}
#     for i, (filepath, label) in enumerate(tqdm(meta[['name', 'label']].values)):
#         with open(f"data\\{filepath}.json", 'r') as fp:
#             data = json.load(fp)
#
#         x = [preprocessing_data(val) for val in data["behavior"]["apistats"]]
#         feature_set["apistats"].update(x)
#
#         x = [preprocessing_data(val) for val in data["behavior"]["apistats_opt"]]
#         feature_set["apistats_opt"].update(x)
#
#         x = [preprocessing_data(val) for val in data["behavior"]["summary"]["regkey_opened"]]
#         feature_set["regkey_opened"].update(x)
#
#         x = [preprocessing_data(val) for val in data["behavior"]["summary"]["regkey_read"]]
#         feature_set["regkey_read"].update(x)
#
#         x = [preprocessing_data(val) for val in data["behavior"]["summary"]["dll_loaded"]]
#         feature_set["dll_loaded"].update(x)
#
#         x = [preprocessing_data(val) for val in data["behavior"]["summary"]["mutex"]]
#         feature_set["mutex"].update(x)
#     return feature_set
#
# meta_path = "data\\dataset1\\labels_preproc.csv"
# s=get_feature_set(meta_path)
#
# for k in s.keys():
#     s[k]=list(s[k])
#     print(f"{k}: {len(s[k])}")
#
# json_object = json.dumps(s, indent=3)
# with open("dataset1_feature_set.json","w") as fp:
#     fp.write(json_object)
#
# with open("dataset1_feature_set.json","r") as fp:
#     s2=json.load(fp)
#
# for k in s2.keys():
#     print(f"{k}: {len(s2[k])}")














