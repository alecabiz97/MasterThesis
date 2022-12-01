import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, det_curve, plot_roc_curve, \
    classification_report, RocCurveDisplay,DetCurveDisplay
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from utils import *
from Neurlux.neurlux_detection import tokenize_data, import_data

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.callbacks import EarlyStopping, ModelCheckpoint
import keras
import tensorflow_hub as hub
import tensorflow_text as text


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim), ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def get_config(self):
        config = super().get_config()
        config.update({
            "att": self.att,
            "ffn": self.ffn,
            "layernorm1": self.layernorm1,
            "layernorm2": self.layernorm2,
            "dropout1": self.dropout1,
            "dropout2": self.dropout2,
        })
        return config

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def get_config(self):
        config = super().get_config()
        config.update({
            "token_emb": self.token_emb,
            "pos_emb": self.pos_emb,
        })
        return config

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions


def get_model(MAXLEN,vocab_size,EMBEDDING_DIM=256,num_heads=2,ff_dim=32):
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
    EMBEDDING_DIM = 256  # 256
    BATCH_SIZE = 10 #
    EPOCHS = 30 # 30
    LEARNING_RATE = 0.0001
    TYPE_SPLIT = 'random'  # 'time' or 'random'
    SPLIT_DATE = "2013-08-09"
    SUBSET_N_SAMPLES = None  # if None takes all data
    TRAINING=False # If True training the models, if False load the trained model
    meta_path="..\\data\\dataset1\\labels_preproc.csv"

    feature_maxlen = [
        # {"apistats": 500, "apistats_opt": 500, "regkey_opened": 500, "regkey_read": 500, "dll_loaded": 500,"mutex": 500},
        {"apistats": 500},
        {"apistats_opt": 500},
        {"regkey_opened": 500},
        {"regkey_read": 500},
        {"dll_loaded": 500},
        {"mutex": 500},
    ]

    # names = ['Mutex']
    # names = ['All', 'API', 'API_OPT', 'Regkey_Opened', 'Regkey_Read', 'DLL_Loaded', 'Mutex']
    names = ['API', 'API_OPT', 'Regkey_Opened', 'Regkey_Read', 'DLL_Loaded', 'Mutex']

    model_names = [f"transformer_detection_{n}_{EPOCHS}_{TYPE_SPLIT}.h5" for n in names]

    test_acc = []
    train_acc = []

    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    for feat, name,model_name in zip(feature_maxlen, names,model_names):
        MAXLEN = sum(feat.values())

        # Import data
        df, classes = import_data(meta_path=meta_path,subset_n_samples=SUBSET_N_SAMPLES, feature_maxlen=feat)
        n_classes = len(classes)

        # Split Train-Test-Validation
        x_tr, y_tr, x_val, y_val, x_ts, y_ts = split_train_val_test_dataframe(df, type_split=TYPE_SPLIT, split_date=SPLIT_DATE,tr=0.8)

        # Tokenize
        x_tr_tokens, x_val_tokens, x_ts_tokens, vocab_size, tokenizer = tokenize_data(x_tr, x_val, x_ts, maxlen=MAXLEN)
        print(f"Vocab size: {vocab_size}")

        #num_heads = 2  # Number of attention heads
        #ff_dim = 32  # Hidden layer size in feed forward network inside transformer

        model=get_model(MAXLEN,vocab_size,EMBEDDING_DIM)
        # print(model.summary())

        if TRAINING:
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
            mc = ModelCheckpoint(f'./trained_models/dataset1/{model_name}', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)

            print("START TRAINING")
            history = model.fit(x_tr_tokens, y_tr, batch_size=BATCH_SIZE, epochs=EPOCHS,
                                validation_data=(x_val_tokens, y_val),callbacks=[es,mc])
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

        scores = model.predict(tf.constant(x_ts_tokens), verbose=False).squeeze()
        y_pred = scores.round().astype(int)

        RocCurveDisplay.from_predictions(y_ts, scores, ax=axs[0], name=f'{name}')
        DetCurveDisplay.from_predictions(y_ts, scores, ax=axs[1], name=f'{name}')

    axs[0].set_title("Receiver Operating Characteristic (ROC) curves")
    axs[1].set_title("Detection Error Tradeoff (DET) curves")

    axs[0].grid(linestyle="--")
    axs[1].grid(linestyle="--")
    axs[0].legend(loc='lower right')
    axs[1].legend(loc='upper right')
    plt.suptitle(f"Transformer Epochs: {EPOCHS} Split: {TYPE_SPLIT}")
    plt.savefig(f"../figure/Transformer_Roc_Det_Epochs_{EPOCHS}_Split_{TYPE_SPLIT}.pdf")
    plt.legend()
    plt.show()

    # %%
    for i in range(len(feature_maxlen)):
        print(" ".join(feature_maxlen[i].keys()))
        print(f"    Train acc: {round(train_acc[i], 2)}")
        print(f"    Test acc: {round(test_acc[i], 2)}")







