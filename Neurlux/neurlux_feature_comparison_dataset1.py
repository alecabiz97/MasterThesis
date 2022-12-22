import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, det_curve, plot_roc_curve, \
    classification_report, RocCurveDisplay,DetCurveDisplay
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from utils import *
from neurlux_detection import get_neurlux

if __name__ == '__main__':
    # Hyperparameters
    EMBEDDING_DIM = 256  # 256
    BATCH_SIZE = 40
    EPOCHS = 30  # 30
    LEARNING_RATE = 0.0001
    TYPE_SPLIT = 'time'  # 'time' or 'random'
    SPLIT_DATE_VAL_TS = "2013-08-09"
    SPLIT_DATE_TR_VAL = "2012-12-09"
    SUBSET_N_SAMPLES = None  # if None takes all data
    WITH_ATTENTION = True
    TRAINING=False # If True training the models, if False load the trained model
    TYPE_FIGURE="roc" # "roc" - "det" - "roc_det"
    SAVE_FIGURE=True
    meta_path="..\\data\\dataset1\\labels_preproc.csv"
    classes = ["Benign", "Malign"]

    feature_maxlen = [
        # {"apistats": 200,"apistats_opt": 200,"regkey_opened": 500,"regkey_read": 500,"dll_loaded": 120,"mutex": 100},
        {"apistats": 200},
        {"apistats_opt": 200},
        {"regkey_opened": 500},
        {"regkey_read": 500},
        {"dll_loaded": 120},
        {"mutex": 100},
        {"regkey_deleted": 100},
        {"regkey_written": 50},
        {"file_deleted": 100},
        {"file_failed": 50},
        {"file_read": 50},
        {"file_opened": 50},
        {"file_exists": 50},
        {"file_written": 50},
        {"file_created": 50}
    ]

    names=[
           'API',
           'API_OPT',
           'Regkey_Opened',
           'Regkey_Read',
           'DLL_Loaded',
           'Mutex',
           'Regkey_Deleted',
           'Regkey_Written',
           'File_Deleted',
           'File_Failed',
           'File_Read',
           'File_Opened',
           'File_Exists',
           'File_Written',
           'File_Created'
           ]
    # names = [list(x.keys())[0] for x in feature_maxlen]

    model_names = [f"neurlux_detection_{n}_{EPOCHS}_{TYPE_SPLIT}.h5" for n in names]
    test_acc = []
    train_acc = []

    if TYPE_FIGURE == 'roc_det':
        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    elif TYPE_FIGURE == 'roc':
        fig_roc, ax_roc = plt.subplots(1, figsize=(15, 7))
    elif TYPE_FIGURE == 'det':
        fig_det, ax_det = plt.subplots(1, figsize=(10, 7))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
              "tab:brown", "gold", "tab:gray", "tab:olive", "tab:cyan",
              "black", "magenta", "navy", "lime", "darkgreen"]
    for feat, name,model_name,color in zip(feature_maxlen, names,model_names,colors):
        MAXLEN = sum(feat.values())

        # Import data
        df = import_data(meta_path=meta_path,subset_n_samples=SUBSET_N_SAMPLES, feature_maxlen=feat,
                                  callback=get_label_date_text_dataframe_dataset1)
        n_classes = len(classes)

        # Split Train-Test-Validation
        x_tr, y_tr, x_val, y_val, x_ts, y_ts = split_train_val_test_dataframe(df, type_split=TYPE_SPLIT,
                                                                              split_dates=[SPLIT_DATE_VAL_TS,SPLIT_DATE_TR_VAL])
        # Tokenize
        x_tr_tokens, x_val_tokens, x_ts_tokens, vocab_size, tokenizer = tokenize_data(x_tr, x_val, x_ts, maxlen=MAXLEN)
        print(f"Vocab size: {vocab_size}")

        # Model definition
        model, _ = get_neurlux(vocab_size, EMBEDDING_DIM, MAXLEN, with_attention=WITH_ATTENTION)
        # print(model.summary())

        if TRAINING:
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
            mc = ModelCheckpoint(f'./trained_models/dataset1/{model_name}', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
            print("START TRAINING")
            history_embedding = model.fit(tf.constant(x_tr_tokens), tf.constant(y_tr),
                                          epochs=EPOCHS, batch_size=BATCH_SIZE,
                                          validation_data=(tf.constant(x_val_tokens), tf.constant(y_val)),
                                          verbose=1, callbacks=[es, mc])
        else:
            model.load_weights(f"./trained_models/dataset1/{model_name}")

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

        # Plot ROC and DET curves
        if TYPE_FIGURE == 'roc_det':
            RocCurveDisplay.from_predictions(y_ts, scores, ax=axs[0], name=f'{name}',color=color)
            DetCurveDisplay.from_predictions(y_ts, scores, ax=axs[1], name=f'{name}',color=color)
        elif TYPE_FIGURE == 'roc':
            RocCurveDisplay.from_predictions(y_ts, scores, name=f'{name}', ax=ax_roc,color=color)
        elif TYPE_FIGURE == 'det':
            DetCurveDisplay.from_predictions(y_ts, scores, name=f'{name}', ax=ax_det,color=color)

    if TYPE_FIGURE == 'roc_det':
        axs[0].set_title("Receiver Operating Characteristic (ROC) curves")
        axs[0].grid(linestyle="--")
        axs[0].legend(loc='lower right')

        axs[1].set_title("Detection Error Tradeoff (DET) curves")
        axs[1].grid(linestyle="--")
        axs[1].legend(loc='upper right')

        plt.suptitle(f"Neurlux Epochs: {EPOCHS} Split: {TYPE_SPLIT}")
        plt.legend()
        if SAVE_FIGURE:
            plt.savefig(f"../figure/Neurlux_Roc_Det_Epochs_{EPOCHS}_Split_{TYPE_SPLIT}.pdf")
    elif TYPE_FIGURE == 'roc':
        ax_roc.set_title("Receiver Operating Characteristic (ROC) curves")
        ax_roc.grid(linestyle="--")
        ax_roc.legend(loc='lower right')
        ax_roc.set_xscale('log')
        if SAVE_FIGURE:
            plt.savefig(f"../figure/Neurlux_Roc_Epochs_{EPOCHS}_Split_{TYPE_SPLIT}.pdf")
    elif TYPE_FIGURE == 'det':
        ax_det.set_title("Detection Error Tradeoff (DET) curves")
        ax_det.grid(linestyle="--")
        ax_det.legend(loc='upper right')
        if SAVE_FIGURE:
            plt.savefig(f"../figure/Neurlux_Det_Epochs_{EPOCHS}_Split_{TYPE_SPLIT}.pdf")
    plt.show()




    # %%
    for i in range(len(feature_maxlen)):
        print(" ".join(feature_maxlen[i].keys()))
        print(f"    Train acc: {round(train_acc[i], 2)}")
        print(f"    Test acc: {round(test_acc[i], 2)}")


