import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from utils import *
from neurlux_classification import get_neurlux


if __name__ == '__main__':
    # Hyperparameters
    EMBEDDING_DIM = 256  # 256
    BATCH_SIZE = 50
    EPOCHS = 15  # 10
    LEARNING_RATE = 0.0001
    TYPE_SPLIT = 'random'  # 'time' or 'random'
    SPLIT_DATE_VAL_TS = "2019-08-01"
    SPLIT_DATE_TR_VAL = "2019-05-01"
    SUBSET_N_SAMPLES = 1000
    WITH_ATTENTION = True
    TRAINING = True  # If True training the models, if False load the trained model
    meta_path = "..\\data\\Avast\\subset_100.csv"
    classes = ['Adload', 'Emotet', 'HarHar', 'Lokibot', 'njRAT', 'Qakbot', 'Swisyn', 'Trickbot', 'Ursnif', 'Zeus']


    feature_maxlen = [
        {'keys': 500},
        {'resolved_apis': 500},
        {'executed_commands': 500},
        {'write_keys': 500},
        {'files': 500},
        {'read_files': 500},
        {'write_files': 500},
        {'delete_keys': 500},
        {'read_keys': 500},
        {'delete_files': 500},
        {'mutexes': 500},
        # {'keys': 500,'resolved_apis': 500,'read_keys': 500,'files': 500},
    ]
    test_acc = []
    train_acc = []

    names=[list(x.keys())[0] for x in feature_maxlen]
    model_names = [f"neurlux_avast_{n}_{EPOCHS}_{TYPE_SPLIT}.h5" for n in names]

    for feat, name,model_name in zip(feature_maxlen, names,model_names):
        MAXLEN=sum(feat.values())

        # Import data
        df = import_data(meta_path=meta_path,subset_n_samples=SUBSET_N_SAMPLES, feature_maxlen=feat,
                                  callback = get_label_date_text_dataframe_avast)
        n_classes = len(classes)

        # Split Train-Test-Validation
        x_tr, y_tr, x_val, y_val, x_ts, y_ts = split_train_val_test_dataframe(df, type_split=TYPE_SPLIT,
                                                                              split_dates=[SPLIT_DATE_VAL_TS,SPLIT_DATE_TR_VAL], tr=0.8)
        # Tokenize
        x_tr_tokens, x_val_tokens, x_ts_tokens, vocab_size, tokenizer = tokenize_data(x_tr, x_val, x_ts, maxlen=MAXLEN)
        print(f"Vocab size: {vocab_size}")


        # Model definition
        model,_ = get_neurlux(vocab_size, EMBEDDING_DIM, MAXLEN,n_classes=n_classes,with_attention=WITH_ATTENTION)
        # print(model.summary())

        if TRAINING:
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
            mc = ModelCheckpoint(f"./trained_models/avast/{model_name}", monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
            print("START TRAINING")
            history_embedding = model.fit(tf.constant(x_tr_tokens), tf.constant(y_tr),
                                          epochs=EPOCHS, batch_size=BATCH_SIZE,
                                          validation_data=(tf.constant(x_val_tokens), tf.constant(y_val)),
                                          verbose=1, callbacks=[es, mc])
        else:
            model.load_weights(f"./trained_models/avast/{model_name}")

        # Test
        print("TEST")

        # print(classification_report(y_pred, y_ts))
        test_acc.append(model.evaluate(x_ts_tokens, np.array(y_ts),verbose=False)[1])
        train_acc.append(model.evaluate(x_tr_tokens, np.array(y_tr),verbose=False)[1])
        # print(confusion_matrix(y_ts,y_pred))

        # Confusion matrix
        # plot_confusion_matrix(y_true=y_ts,y_pred=y_pred,classes=classes)

        scores = model.predict(tf.constant(x_ts_tokens)).squeeze()
        y_pred = scores.round().astype(int)


# %%
    for i in range(len(feature_maxlen)):
        print(" ".join(feature_maxlen[i].keys()))
        print(f"    Train acc: {round(train_acc[i],2)}")
        print(f"    Test acc: {round(test_acc[i],2)}")