import warnings
from keras.layers import Activation, LSTM, Bidirectional, Dense, Dropout, Input,Embedding,Attention, Conv1D,MaxPooling1D,CuDNNLSTM
import keras.backend as K
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import keras
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import initializers, regularizers
from keras import constraints
from utils import *
from lime import lime_text


warnings.simplefilter(action='ignore', category=FutureWarning)


def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)


class AttentionWithContext(keras.layers.Layer):
    """
    Attention operation, with a context/query vector, for temporal data.
    Supports Masking.
    Follows the work of Yang et al. [https://www.cs.cmu.edu/~diyiy/docs/naacl16.pdf]
    "Hierarchical Attention Networks for Document Classification"
    by using a context vector to assist the attention
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    Note: The layer has been tested with Keras 2.0.6
    Example:
        model.add(LSTM(64, return_sequences=True))
        model.add(AttentionWithContext())
        # next add a Dense layer (for classification/regression) or whatever...
    """

    def __init__(self,
                 W_regularizer=None, u_regularizer=None, b_regularizer=None,
                 W_constraint=None, u_constraint=None, b_constraint=None,
                 bias=True, **kwargs):

        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(AttentionWithContext, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight(shape=(input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight(shape=(input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight(shape=(input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(AttentionWithContext, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

    def get_config(self):
        config = {
            "W_regularizer": self.W_regularizer,
            "u_regularizer": self.u_regularizer,
            "b_regularizer": self.b_regularizer,
            "W_constraint": self.W_constraint,
            "u_constraint": self.u_constraint,
            "b_constraint": self.b_constraint,
            "bias": self.bias,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))


class Addition(keras.layers.Layer):
    """
    This layer is supposed to add of all activation weight.
    We split this from AttentionWithContext to help us getting the activation weights
    follows this equation:
    (1) v = \sum_t(\alpha_t * h_t)

    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        2D tensor with shape: `(samples, features)`.
    """

    def __init__(self, **kwargs):
        super(Addition, self).__init__(**kwargs)

    def build(self, input_shape):
        self.output_dim = input_shape[-1]
        super(Addition, self).build(input_shape)

    def call(self, x):
        return K.sum(x, axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)


def get_neurlux(vocab_size,EMBEDDING_DIM,MAXLEN,n_classes=None):
    inp=Input(shape=(MAXLEN))
    x=Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM,input_length=MAXLEN)(inp)
    x = Conv1D(filters=100, kernel_size=4, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=4,data_format="channels_first")(x)
    # x = Bidirectional(CuDNNLSTM(64, return_sequences=True))(x)
    x = Bidirectional(CuDNNLSTM(32))(x)
    # x,att_score=Attention(name='attention_vec')([x,x], return_attention_scores=True)
    # x=AttentionWithContext()(x)
    # x=Addition(x)
    x = Dense(10, activation="relu")(x)
    x = Dropout(0.25)(x)
    x = Dense(n_classes, activation="softmax")(x)

    model=keras.models.Model(inputs=inp,outputs=x)

    model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics='accuracy')
    return model


def import_data(subset_n_samples,type_split,feature_maxlen=None):
    split_date = '2019-08-01'
    classes = ['Adload', 'Emotet', 'HarHar', 'Lokibot', 'njRAT', 'Qakbot', 'Swisyn', 'Trickbot', 'Ursnif', 'Zeus']
    # df = pd.read_csv("data_avast_100.csv")
    df = get_label_date_text_dataframe_avast("Avast\\subset_100.csv",feature_maxlen=feature_maxlen)
    # df = pd.read_csv("data_avast.csv")

    df = df.sample(frac=1)  # Shuffle dataset
    if subset_n_samples:
        df = df.iloc[0:subset_n_samples, :].reset_index(drop=True)  # Subset
    print(df.head())

    # Create training, validation and test set
    if type_split == 'random':
        x_tr, x_tmp, y_tr, y_tmp = train_test_split(df['text'], df['label'], test_size=0.2, stratify=df['label'])
    elif type_split == 'time':
        x_tr, y_tr = df[df['date'] < split_date]['text'], df[df['date'] < split_date]['label']
        x_tmp, y_tmp = df[df['date'] >= split_date]['text'], df[df['date'] >= split_date]['label']
    x_val, x_ts, y_val, y_ts = train_test_split(x_tmp, y_tmp, test_size=0.6, stratify=y_tmp)

    print(f"Split train-test: {type_split}")
    print(f"Train size: {len(y_tr)} -- n_classes:{len(set(y_tr))}")
    print(f"Validation size: {len(y_val)} -- n_classes:{len(set(y_val))}")
    print(f"Test size: {len(y_ts)} -- n_classes:{len(set(y_ts))}")

    return x_tr,y_tr,x_val,y_val,x_ts,y_ts,classes

def lime_explanation(x,x_tokens,y,model,feature_maxlen,classes,num_features,feature_stats=False):

    def predict_proba(sample):
        x = tokenizer.texts_to_sequences(sample)
        x = pad_sequences(x, maxlen=MAXLEN, padding='post')
        scores = model.predict(x)

        return scores

    def get_stats(meta_path):
        meta = pd.read_csv(meta_path)
        root = 'Avast\\public_small_reports'

        stats = {
            'keys': set(),
            'resolved_apis': set(),
            'executed_commands': set(),
            'write_keys': set(),
            'files': set(),
            'read_files': set(),
            'write_files': set(),
            'delete_keys': set(),
            'read_keys': set(),
            'delete_files': set(),
            'mutexes': set()
        }
        for i, sha in enumerate(tqdm(meta["sha256"].values)):
            try:
                filepath = f"{root}\\{sha}.json"
                with open(filepath, 'r') as fp:
                    data = json.load(fp)

                for k in stats.keys():
                    x = [preprocessing_data(val) for val in data["behavior"]["summary"][k]]
                    stats[k].update(x)

            except:
                pass
        return stats


    MAXLEN = sum(feature_maxlen.values())



    if feature_stats:
        meta_path = "Avast\\subset_100.csv"
        stats = get_stats(meta_path)
        feature_importance={k:0 for k in stats.keys()}

    for idx in range(len(y)):
        sample = x.iloc[idx]
        y_sample = y.iloc[idx]
        # print(f"Idx: {idx}")
        y_pred = np.argmax(model.predict(tf.constant([x_tokens[idx]])).squeeze())
        print(f"Label sample: {classes[y_sample]}")
        print(f"Predicted: {classes[y_pred]}")
        explainer = lime_text.LimeTextExplainer(class_names=classes)
        explanation = explainer.explain_instance(sample, classifier_fn=predict_proba, num_features=num_features, top_labels=1)

        explanation.save_to_file(f'exp_{idx}.html')
        print("Explanation file created")

        if feature_stats:
            for val, importance in explanation.as_list(label=explanation.available_labels()[0]):
                for k in stats.keys():
                    if val in stats[k]:
                        feature_importance[k] += 1

    if feature_stats:
        for k, val in feature_importance.items():
            print(f'{k}: {val}')



if __name__ == '__main__':

    # Hyperparameters
    feature_maxlen = {
        'keys': 600,
        'resolved_apis': 600,
        'executed_commands': 20,
        'write_keys': 20,
        'files': 1000,
        'read_files': 200,
        "started_services":50,
        "created_services":50,
        'write_files': 200,
        'delete_keys': 100,
        'read_keys': 100,
        'delete_files':100,
        'mutexes': 50
    }
    MAXLEN = sum(feature_maxlen.values())
    # MAXLEN = 500
    EMBEDDING_DIM=256 # 256
    BATCH_SIZE = 50
    EPOCHS = 10 # 10
    LEARNING_RATE = 0.0001
    TYPE_SPLIT='time' # 'time' or 'random'
    SUBSET_N_SAMPLES=1000 # if None takes all data
    LIME_EXPLANATION=True

    # Import data
    x_tr, y_tr, x_val, y_val, x_ts, y_ts, classes = import_data(subset_n_samples=SUBSET_N_SAMPLES,
                                                                type_split=TYPE_SPLIT,feature_maxlen=feature_maxlen)
    n_classes = len(set(y_tr))

    # Tokenize
    x_tr_tokens, x_val_tokens, x_ts_tokens, vocab_size, tokenizer = tokenize_data(x_tr, x_val, x_ts, maxlen=MAXLEN)
    print(f"Vocab size: {vocab_size}")

    # Save tokenizer
    with open(f'tokenizer_classification.pickle', 'wb') as fp:
        pickle.dump(tokenizer, fp)

    # Model definition
    model = get_neurlux(vocab_size, EMBEDDING_DIM, MAXLEN, n_classes=n_classes)
    print(model.summary())

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    mc = ModelCheckpoint(f'./model_classification.h5', monitor='val_accuracy', mode='max', verbose=1, save_best_only=True)
    print("START TRAINING")
    history_embedding = model.fit(tf.constant(x_tr_tokens), tf.constant(y_tr),
                                  epochs=EPOCHS, batch_size=BATCH_SIZE,
                                  validation_data=(tf.constant(x_val_tokens), tf.constant(y_val)),
                                  verbose=1, callbacks=[es, mc])

    #Test
    print("TEST")

    # print(classification_report(y_pred, y_ts))
    print(f"Test accuracy: {model.evaluate(x_ts_tokens, np.array(y_ts),verbose=False)[1]}")
    print(f"Train accuracy: {model.evaluate(x_tr_tokens, np.array(y_tr),verbose=False)[1]}")
    # print(confusion_matrix(y_ts,y_pred))


    y_pred = np.argmax(model.predict(tf.constant(x_ts_tokens)), axis=1)

    # Confusion matrix
    plot_confusion_matrix(y_true=y_ts, y_pred=y_pred, classes=classes)

    # LIME Explanation
    if LIME_EXPLANATION:
        x = x_ts[0:1]
        x_tokens = x_ts_tokens[0:1]
        y = y_ts[0:1]
        lime_explanation(x=x, x_tokens=x_tokens, y=y, model=model, feature_maxlen=feature_maxlen,
                         classes=classes, num_features=10, feature_stats=False)







