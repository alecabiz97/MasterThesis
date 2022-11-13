from keras.models import load_model
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from tqdm import tqdm
from utils import *
import json
import tensorflow as tf
from lime import lime_text


def predict_proba(sample):
    x = tokenizer.texts_to_sequences(sample)
    x = pad_sequences(x, maxlen=MAXLEN, padding='post')
    if MODE == "classification":
        scores = model.predict(x)
    elif MODE == 'detection':
        scores_tmp = model.predict(x)
        scores = []
        for val in scores_tmp:
            scores.append([1 - val, val[0]])
        scores = np.array(scores)

    return scores


def get_stats(meta_path):
    meta = pd.read_csv(meta_path)
    stats = {"apistats": set(),
             "apistats_opt": set(),
             "regkey_opened": set(),
             "regkey_read": set(),
             "dll_loaded": set(),
             "mutex": set()}
    for i, (filepath, label) in enumerate(tqdm(meta[['name', 'label']].values)):
        with open(f"{filepath}.json", 'r') as fp:
            data = json.load(fp)

        x = [preprocessing_data(val) for val in data["behavior"]["apistats"]]
        stats["apistats"].update(x)

        x = [preprocessing_data(val) for val in data["behavior"]["apistats_opt"]]
        stats["apistats_opt"].update(x)

        x = [preprocessing_data(val) for val in data["behavior"]["summary"]["regkey_opened"]]
        stats["regkey_opened"].update(x)

        x = [preprocessing_data(val) for val in data["behavior"]["summary"]["regkey_read"]]
        stats["regkey_read"].update(x)

        x = [preprocessing_data(val) for val in data["behavior"]["summary"]["dll_loaded"]]
        stats["dll_loaded"].update(x)

        x = [preprocessing_data(val) for val in data["behavior"]["summary"]["mutex"]]
        stats["mutex"].update(x)
    return stats


if __name__ == '__main__':

    # Hyperparameters
    MAXLEN = 500  # 500
    EMBEDDING_DIM = 256  # 256
    MODE = 'detection'  # 'classification' or 'detection'

    # Load tokenizer
    with open(f'tokenizer_{MODE}.pickle', 'rb') as fp:
        tokenizer = pickle.load(fp)
    print("Tokenizer loaded")

    # Load model
    model = load_model(f'model_{MODE}.h5')
    print("Model loaded")


    if MODE == 'classification':
        classes = ['Adload', 'Emotet', 'HarHar', 'Lokibot', 'njRAT', 'Qakbot', 'Swisyn', 'Trickbot', 'Ursnif', 'Zeus']
        # df = pd.read_csv("data_avast_100.csv")
        df = get_label_date_text_dataframe_avast("Avast\\subset_100.csv")
    elif MODE == 'detection':
        classes = ["Benign", "Malign"]
        df = get_label_date_text_dataframe_dataset1("dataset1\\labels_preproc.csv")
        # df = pd.read_csv("data.csv")

    df = df.sample(frac=1)  # Shuffle dataset
    df = df.iloc[0:100, :].reset_index(drop=True)  # Subset
    print(df.head())

    x,y=df['text'],df['label']

    x_tokens = tokenizer.texts_to_sequences(x.values)
    x_tokens = pad_sequences(x_tokens, maxlen=MAXLEN, padding='post')

    meta_path = "dataset1\\labels_preproc.csv"
    stats = get_stats(meta_path)

    cnt_api = 0
    cnt_api_opt = 0
    cnt_regop = 0
    cnt_regre = 0
    cnt_dll = 0
    cnt_mutex = 0

    for idx in range(10):
        # idx=1
        sample = x.iloc[idx]
        y_sample = y.iloc[idx]
        print(f"Idx: {idx}")
        y_pred = model.predict(tf.constant([x_tokens[idx]])).squeeze().round().astype(int)
        print(f"Label sample: {classes[y_sample]}")
        print(f"Predicted: {classes[y_pred]}")
        explainer = lime_text.LimeTextExplainer(class_names=classes)
        explanation = explainer.explain_instance(sample, classifier_fn=predict_proba, num_features=10, top_labels=1)

        explanation.save_to_file(f'exp.html')
        print("Explanation file created")

        for a, b in explanation.as_list(label=explanation.available_labels()[0]):
            if a in stats["apistats"]:
                cnt_api += 1
            elif a in stats["apistats_opt"]:
                cnt_api_opt += 1
            elif a in stats["regkey_opened"]:
                cnt_regop += 1
            elif a in stats["regkey_read"]:
                cnt_regre += 1
            elif a in stats["dll_loaded"]:
                cnt_dll += 1
            elif a in stats["mutex"]:
                cnt_mutex += 1

    print(f"API: {cnt_api}")
    print(f"API OPT: {cnt_api_opt}")
    print(f"REGOP: {cnt_regop}")
    print(f"REGRE: {cnt_regre}")
    print(f"DLL: {cnt_dll}")
    print(f"MUTEX: {cnt_mutex}")






