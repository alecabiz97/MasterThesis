import warnings
from keras.callbacks import EarlyStopping, ModelCheckpoint
from utils import *
from keras import layers


warnings.simplefilter(action='ignore', category=FutureWarning)


def get_transformer_model(vocab_size,EMBEDDING_DIM,MAXLEN,n_classes=10,num_heads=2,ff_dim=32):
    # Transformer (funziona)
    inp = layers.Input(shape=(MAXLEN,))
    x = TokenAndPositionEmbedding(MAXLEN, vocab_size, EMBEDDING_DIM)(inp)
    # x = layers.Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=MAXLEN)(inp)
    x= TransformerBlock(EMBEDDING_DIM, num_heads, ff_dim)(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    model = keras.Model(inputs=inp, outputs=outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


if __name__ == '__main__':

    # Hyperparameters
    feature_maxlen = {
        # 'keys': 500,
        'resolved_apis': 600,
        # 'executed_commands': 20,
        # 'write_keys': 20,
        # 'files': 600,
        # 'read_files': 200,
        # "started_services":50,
        # "created_services":50,
        # 'write_files': 400,
        # 'delete_keys': 100,
        # 'read_keys': 400,
        # 'delete_files':100,
        # 'mutexes': 50
    }
    MAXLEN = sum(feature_maxlen.values())

    # feature_maxlen=None
    # MAXLEN = 1000

    EMBEDDING_DIM=256 # 256
    BATCH_SIZE = 40
    EPOCHS = 2 # 10
    LEARNING_RATE = 0.0001
    TYPE_SPLIT='time' # 'time' or 'random'
    SPLIT_DATE_VAL_TS = "2019-08-01"
    SPLIT_DATE_TR_VAL = "2019-05-01"
    SUBSET_N_SAMPLES=1000
    TRAINING = True
    meta_path = "..\\data\\Avast\\subset_100.csv"
    model_name = "Transformer_Avast"
    classes = ['Adload', 'Emotet', 'HarHar', 'Lokibot', 'njRAT', 'Qakbot', 'Swisyn', 'Trickbot', 'Ursnif', 'Zeus']


    # Explanation
    LIME_EXPLANATION = False
    TOPK_FEATURE = 10
    N_SAMPLES_EXP = 1

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
    model = get_transformer_model(vocab_size,EMBEDDING_DIM,MAXLEN)
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

    # LIME Explanation
    if LIME_EXPLANATION:
        x = x_ts[0:N_SAMPLES_EXP]
        x_tokens = x_ts_tokens[0:N_SAMPLES_EXP]
        y = y_ts[0:N_SAMPLES_EXP]

        explanations = lime_explanation_avast(x=x, x_tokens=x_tokens, y=y, model=model,tokenizer=tokenizer,
                                              feature_maxlen=feature_maxlen,classes=classes,
                                              num_features=TOPK_FEATURE, feature_stats=False)

        # top_feat_lime=[val[0] for val in explanation.as_list(label=explanation.available_labels()[0])]
        top_feat_lime = [[val[0] for val in exp.as_list(label=exp.available_labels()[0])] for exp in explanations]




# %% Create a json file with Avast feature set
# def get_feature_set(meta_path):
#     meta = pd.read_csv(meta_path)
#     root = 'data\\Avast\\public_small_reports'
#
#     feature_set = {
#         'keys': set(),
#         'resolved_apis': set(),
#         'executed_commands': set(),
#         'write_keys': set(),
#         'files': set(),
#         'read_files': set(),
#         'started_services': set(),
#         'created_services': set(),
#         'write_files': set(),
#         'delete_keys': set(),
#         'read_keys': set(),
#         'delete_files': set(),
#         'mutexes': set()
#     }
#     for i, sha in enumerate(tqdm(meta["sha256"].values)):
#         try:
#             filepath = f"{root}\\{sha}.json"
#             with open(filepath, 'r') as fp:
#                 data = json.load(fp)
#
#             for k in feature_set.keys():
#                 x = [preprocessing_data(val) for val in data["behavior"]["summary"][k]]
#                 feature_set[k].update(x)
#         except:
#             pass
#     return feature_set
#
# # meta_path = "data\\Avast\\subset_100.csv"
# meta_path = 'data\\Avast\\public_labels.csv'
# s=get_feature_set(meta_path)
#
# for k in s.keys():
#     s[k]=list(s[k])
#     print(f"{k}: {len(s[k])}")
#
# json_object = json.dumps(s, indent=3)
# with open("Avast_feature_set.json","w") as fp:
#     fp.write(json_object)
#
# with open("Avast_feature_set.json","r") as fp:
#     s2=json.load(fp)
#
# for k in s2.keys():
#     print(f"{k}: {len(s2[k])}")
