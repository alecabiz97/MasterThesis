import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, det_curve, plot_roc_curve, \
    classification_report, RocCurveDisplay,DetCurveDisplay
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from utils import *
from neurlux_detection import get_neurlux, tokenize_data, import_data

if __name__ == '__main__':
    # Hyperparameters
    EMBEDDING_DIM = 256  # 256
    BATCH_SIZE = 40
    EPOCHS = 30  # 30
    LEARNING_RATE = 0.0001
    TYPE_SPLIT = 'random'  # 'time' or 'random'
    SPLIT_DATE = "2013-08-09"
    SUBSET_N_SAMPLES = None  # if None takes all data
    WITH_ATTENTION = True

    feature_maxlen = [
        {"apistats": 500,"apistats_opt": 500,"regkey_opened": 500,"regkey_read": 500,"dll_loaded": 500,"mutex": 500},
        {"apistats": 500},
        {"apistats_opt": 500},
        {"regkey_opened": 500},
        {"regkey_read": 500},
        {"dll_loaded": 500},
        {"mutex": 500},
    ]

    # names = ['API', 'API OPT','DLL']
    names=['All','API','API OPT','Regkey Opened','Regkey Read','DLL Loaded','Mutex']
    test_acc = []
    train_acc = []

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    for feat, name in zip(feature_maxlen, names):
        MAXLEN = sum(feat.values())

        # Import data
        df, classes = import_data(subset_n_samples=SUBSET_N_SAMPLES, feature_maxlen=feat)
        n_classes = len(classes)

        # Split Train-Test-Validation
        x_tr, y_tr, x_val, y_val, x_ts, y_ts = split_train_val_test_dataframe(df, type_split=TYPE_SPLIT,
                                                                              split_date=SPLIT_DATE, tr=0.8)

        # Tokenize
        x_tr_tokens, x_val_tokens, x_ts_tokens, vocab_size, tokenizer = tokenize_data(x_tr, x_val, x_ts, maxlen=MAXLEN)
        print(f"Vocab size: {vocab_size}")

        # Model definition
        model, _ = get_neurlux(vocab_size, EMBEDDING_DIM, MAXLEN, with_attention=WITH_ATTENTION)
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
        test_acc.append(model.evaluate(x_ts_tokens, y_ts)[1])
        train_acc.append(model.evaluate(x_tr_tokens, y_tr)[1])
        # print(confusion_matrix(y_ts,y_pred))

        # Confusion matrix
        # plot_confusion_matrix(y_true=y_ts,y_pred=y_pred,classes=classes)

        scores = model.predict(tf.constant(x_ts_tokens)).squeeze()
        y_pred = scores.round().astype(int)

        RocCurveDisplay.from_predictions(y_ts,scores,ax=axs[0],name=f'{name}')
        DetCurveDisplay.from_predictions(y_ts,scores,ax=axs[1],name=f'{name}')


    axs[0].set_title("Receiver Operating Characteristic (ROC) curves")
    axs[1].set_title("Detection Error Tradeoff (DET) curves")

    axs[0].grid(linestyle="--")
    axs[1].grid(linestyle="--")
    axs[0].legend(loc='lower right')
    axs[1].legend(loc='upper right')
    plt.savefig(f"figure/Neurlux_Roc_Det_Epochs_{EPOCHS}_Split_{TYPE_SPLIT}.pdf")
    plt.legend()
    plt.show()

    # %%
    for i in range(len(feature_maxlen)):
        print(" ".join(feature_maxlen[i].keys()))
        print(f"    Train acc: {round(train_acc[i], 2)}")
        print(f"    Test acc: {round(test_acc[i], 2)}")


