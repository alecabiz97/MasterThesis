import warnings
from keras.callbacks import EarlyStopping, ModelCheckpoint
from utils import *
from keras import layers
from tensorflow.keras.preprocessing.sequence import pad_sequences

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
        'keys': 500,
        'resolved_apis': 200,
        'executed_commands': 20,
        'write_keys': 20,
        'files': 100,
        'read_files': 200,
        'write_files': 100,
        'delete_keys': 100,
        'read_keys': 250,
        'delete_files': 50,
        'mutexes': 20,
        # "started_services":5,
        # "created_services":5,
    }

    MAXLEN = sum(feature_maxlen.values())

    EMBEDDING_DIM=256 # 256
    BATCH_SIZE = 10
    EPOCHS = 30 # 10
    LEARNING_RATE = 0.0001
    TYPE_SPLIT='random' # 'time' or 'random'
    SPLIT_DATE_VAL_TS = "2019-07-01"
    SPLIT_DATE_TR_VAL = "2019-05-01"
    SUBSET_N_SAMPLES=None
    TRAINING = False
    meta_path = "..\\data\\Avast\\subset_100.csv"
    # model_name = "Transformer_Avast"
    model_name = f"transformer_avast_all_{EPOCHS}_{TYPE_SPLIT}"
    classes = ['Adload', 'Emotet', 'HarHar', 'Lokibot', 'njRAT', 'Qakbot', 'Swisyn', 'Trickbot', 'Ursnif', 'Zeus']

    # Explanation
    SHAP = False
    LIME = False
    EXP_MODE = 'multi'  # single or multi
    TOPK_FEATURE = 10
    N_SAMPLES_EXP = 10
    SAVE_EXP_DICT=False

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
    print(f"Train accuracy: {model.evaluate(x_tr_tokens, np.array(y_tr),verbose=False,batch_size=BATCH_SIZE)[1]}")
    print(f"Test accuracy: {model.evaluate(x_ts_tokens, np.array(y_ts),verbose=False,batch_size=BATCH_SIZE)[1]}")
    # print(confusion_matrix(y_ts,y_pred))

    y_pred = np.argmax(model.predict(tf.constant(x_ts_tokens),batch_size=BATCH_SIZE), axis=1)

    # Confusion matrix
    plot_confusion_matrix(y_true=y_ts, y_pred=y_pred, classes=classes)

    # %% Explanation

    hash = "b773c3406e289cd100237bec78642bf0cbc95f0c408b20165cc3d02b89d35081"  # Emotet

    #     hash="f7a4a26c10c86ce3c1e9b606ed3e59c4c12758c24de95bd68016200b28e6b06b" # Emotet
    #     hash="6847bd9c431b65456654ce635ce365ca4c66bb056f648eab54e87ad7b7269c60" # Trickbot
    #     hash="2e1640fb12cc66af4428e3e8e2a0de4d768f2e4085144a3f04aafc79fd53c38a" # Trickbot
    #     hash="1b6ca28027f62a4922348c55b891472c9530da1b0ab2af1ab615a491612bea01" # Trickbot
    #     hash="efb793eafd7993152fcb0075887584cd65bab183d0ebea0bbbcf05255c8be8db" # njRAT
    #     hash="32c58040d3d6ec5305a1a0ebb48ba05aebe3ac2f905a7f152f32fc9170e16711" # Trickbot

    y_true = "Emotet"

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
            x = []
            x_tokens = np.zeros(shape=(N_SAMPLES_EXP * len(classes), x_ts_tokens.shape[1]),dtype=int)
            y = np.zeros(shape=(N_SAMPLES_EXP * len(classes)), dtype=int)
            for i in range(len(classes)):
                idx = (y_ts == i).to_numpy()
                x_tokens[i * N_SAMPLES_EXP:i * N_SAMPLES_EXP + N_SAMPLES_EXP, :] = x_ts_tokens[idx, :][
                                                                                   0:N_SAMPLES_EXP, :]
                y[i * N_SAMPLES_EXP:i * N_SAMPLES_EXP + N_SAMPLES_EXP] = y_ts.values[idx][0:N_SAMPLES_EXP]
                x.extend(x_ts.values[idx][0:N_SAMPLES_EXP].tolist())

            x = pd.Series(x)
            y = pd.Series(y.tolist())

            # x = x_ts[0:N_SAMPLES_EXP]
            # x_tokens = x_ts_tokens[0:N_SAMPLES_EXP]
            # y = y_ts[0:N_SAMPLES_EXP]

        top_feat_dict_lime = lime_explanation_avast(x=x, x_tokens=x_tokens, y=y, model=model, tokenizer=tokenizer,
                                               feature_maxlen=feature_maxlen, classes=classes,
                                               num_features=TOPK_FEATURE,batch_size=BATCH_SIZE,save_html=False)
        # Save top feat dict
        if SAVE_EXP_DICT:
            with open(f"top_feat_dict_lime_{model_name}.json", "w") as outfile:
                json.dump(top_feat_dict_lime, outfile, indent=4)

        # Load top feat dict
        with open(f"top_feat_dict_lime_{model_name}.json", "r") as outfile:
            top_feat_dict_lime = json.load(outfile)

        feature_set_path="../data/Avast/Avast_feature_set.json"
        print_top_feature_avast(top_feat_dict_lime,feature_set_path=feature_set_path)

    # %%
    if SHAP:
        def f(x):
            return model.predict(x, batch_size=BATCH_SIZE)

        explainer = shap.KernelExplainer(f, np.zeros((1, x_ts_tokens.shape[1])))
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
            sample_tokens=np.zeros(shape=(N_SAMPLES_EXP*len(classes),x_ts_tokens.shape[1]),dtype=int)
            idx_true=np.zeros(shape=(N_SAMPLES_EXP*len(classes)),dtype=int)
            for i in range(len(classes)):
                idx=(y_ts==i).to_numpy()
                sample_tokens[i*N_SAMPLES_EXP:i*N_SAMPLES_EXP+N_SAMPLES_EXP,:]=x_ts_tokens[idx,:][0:N_SAMPLES_EXP,:]
                idx_true[i*N_SAMPLES_EXP:i*N_SAMPLES_EXP+N_SAMPLES_EXP]=y_ts.values[idx][0:N_SAMPLES_EXP]

            # sample = x_ts.iloc[0:0 + N_SAMPLES_EXP]
            # sample_tokens = x_ts_tokens[0:0 + N_SAMPLES_EXP]
            # idx_true = y_ts.iloc[0:0 + N_SAMPLES_EXP].values

            print(idx_true.shape, sample_tokens.shape)

        top_feat_dict_shap = shap_explanation_avast(explainer=explainer, sample_tokens=sample_tokens, classes=classes,
                                               tokenizer=tokenizer, model=model, idx_true=idx_true,
                                               summary_plot_feat=False,batch_size=BATCH_SIZE,
                                               summary_plot=False, dependence_plot=False)
        # Save top feat dict
        if SAVE_EXP_DICT:
            with open(f"top_feat_dict_shap_{model_name}.json", "w") as outfile:
                json.dump(top_feat_dict_shap, outfile, indent=4)

        # Load top feat dict
        with open(f"top_feat_dict_shap_{model_name}.json", "r") as outfile:
            top_feat_dict_shap=json.load(outfile)

        print_top_feature_avast(top_feat_dict_shap)

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
