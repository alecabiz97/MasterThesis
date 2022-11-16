from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, plot_roc_curve, classification_report, RocCurveDisplay
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from utils import *
from neurlux_detection import get_neurlux, tokenize_data, import_data


if __name__ == '__main__':
    # Hyperparameters
    EMBEDDING_DIM = 256  # 256
    BATCH_SIZE = 100
    EPOCHS = 10  # 10
    LEARNING_RATE = 0.0001
    TYPE_SPLIT = 'random'  # 'time' or 'random'
    SUBSET_N_SAMPLES = 1000 # if None takes all data

    feature_maxlen = [
        {"apistats": 500},
        {"apistats_opt": 500},
        {"regkey_opened": 500},
        {"regkey_read": 500},
        {"dll_loaded": 500},
        {"mutex": 500},
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
        model = get_neurlux(vocab_size, EMBEDDING_DIM, MAXLEN)
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

        # ROC curve
        auc = roc_auc_score(y_ts, scores)
        print(f"AUC: {auc}")
        fpr, tpr, thresh = roc_curve(y_ts, scores)
        plt.plot(fpr, tpr, label=f'{feat} (AUC = {round(auc, 2)})')
        # RocCurveDisplay.from_predictions(y_ts,scores)

    plt.title(f"Epochs: {EPOCHS} - Split: {TYPE_SPLIT}")
    plt.legend()
    plt.show()

# %%
    for i in range(len(feature_maxlen)):
        print(" ".join(feature_maxlen[i].keys()))
        print(f"    Train acc: {round(train_acc[i],2)}")
        print(f"    Test acc: {round(test_acc[i],2)}")