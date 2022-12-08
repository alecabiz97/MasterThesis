import warnings
from keras.layers import Activation, LSTM, Bidirectional, Dense, Dropout, Input,Embedding, Conv1D,MaxPooling1D,CuDNNLSTM
import keras.backend as K
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import keras
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import initializers, regularizers
from keras import constraints
from utils import *

warnings.simplefilter(action='ignore', category=FutureWarning)


def get_neurlux(vocab_size,EMBEDDING_DIM,MAXLEN,n_classes=None,with_attention=False):
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
    x = Dense(n_classes, activation="softmax")(x)

    model=keras.models.Model(inputs=inp,outputs=x)
    model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics='accuracy')

    if with_attention:
        attention_model = keras.models.Model(inputs=inp, outputs=attention_out)
        return model, attention_model
    else:
        return model, None

if __name__ == '__main__':

    # Hyperparameters
    feature_maxlen = {
        'keys': 500,
        'resolved_apis': 600,
        'executed_commands': 20,
        'write_keys': 20,
        'files': 600,
        'read_files': 200,
        "started_services":50,
        "created_services":50,
        'write_files': 400,
        'delete_keys': 100,
        'read_keys': 400,
        'delete_files':100,
        'mutexes': 50
    }
    MAXLEN = sum(feature_maxlen.values())

    # feature_maxlen=None
    # MAXLEN = 1000

    EMBEDDING_DIM=256 # 256
    BATCH_SIZE = 40
    EPOCHS = 15 # 10
    LEARNING_RATE = 0.0001
    TYPE_SPLIT='random' # 'time' or 'random'
    SPLIT_DATE_VAL_TS = "2019-08-01"
    SPLIT_DATE_TR_VAL = "2019-05-01"
    SUBSET_N_SAMPLES=1000 # if None takes all data
    WITH_ATTENTION = True
    TRAINING = True
    meta_path = "..\\data\\Avast\\subset_100.csv"
    model_name = "Neurlux_Avast"
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
    model,attention_model = get_neurlux(vocab_size, EMBEDDING_DIM, MAXLEN, n_classes=n_classes,with_attention=WITH_ATTENTION)
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
                                              num_features=TOPK_FEATURE, feature_stats=True)

        # top_feat_lime=[val[0] for val in explanation.as_list(label=explanation.available_labels()[0])]
        top_feat_lime = [[val[0] for val in exp.as_list(label=exp.available_labels()[0])] for exp in explanations]

        # Attention
        if WITH_ATTENTION:
            top_feat_att = get_top_feature_attention(attention_model, tokenizer, x_tokens, topk=TOPK_FEATURE)

            for i in range(N_SAMPLES_EXP):
                cnt = 0
                for val in top_feat_att[i]:
                    if val in top_feat_lime[i]:
                        # print(val)
                        cnt += 1
                print(f"[Sample {i}] Common feature Attention/LIME: {cnt}/{TOPK_FEATURE}")







