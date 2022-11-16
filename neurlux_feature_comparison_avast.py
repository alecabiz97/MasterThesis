import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from utils import *
from neurlux_classification import get_neurlux, tokenize_data, import_data


if __name__ == '__main__':
    # Hyperparameters
    EMBEDDING_DIM = 256  # 256
    BATCH_SIZE = 100
    EPOCHS = 10  # 10
    LEARNING_RATE = 0.0001
    TYPE_SPLIT = 'random'  # 'time' or 'random'
    SUBSET_N_SAMPLES = 1000 # if None takes all data


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
        {'mutexes': 500}
    ]
    test_acc = []
    train_acc = []

    for feat in feature_maxlen:
        MAXLEN=sum(feat.values())
        # Import data
        x_tr, y_tr, x_val, y_val, x_ts, y_ts,classes = import_data(subset_n_samples=SUBSET_N_SAMPLES,
                                                           type_split=TYPE_SPLIT,feature_maxlen=feat)
        n_classes = len(set(y_tr))

        # Tokenize
        x_tr_tokens, x_val_tokens, x_ts_tokens, vocab_size, tokenizer = tokenize_data(x_tr, x_val, x_ts, maxlen=MAXLEN)
        print(f"Vocab size: {vocab_size}")


        # Model definition
        model = get_neurlux(vocab_size, EMBEDDING_DIM, MAXLEN,n_classes=n_classes)
        # print(model.summary())

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        mc = ModelCheckpoint(f'./model.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
        print("START TRAINING")
        history_embedding = model.fit(tf.constant(x_tr_tokens), tf.constant(y_tr),
                                      epochs=EPOCHS, batch_size=BATCH_SIZE,
                                      validation_data=(tf.constant(x_val_tokens), tf.constant(y_val)),
                                      verbose=1, callbacks=[es, mc])

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