import json
from sklearn.metrics import RocCurveDisplay,DetCurveDisplay
import matplotlib.pyplot as plt

# %% Feature comparison

features_names = ["API","API_OPT","Regkey_Opened","Regkey_Read","DLL_Loaded","Mutex","Regkey_Deleted","Regkey_Written","File_Deleted","File_Failed","File_Read","File_Opened","File_Exists","File_Written","File_Created"]
colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple",
          "tab:brown", "gold", "tab:gray", "tab:olive", "tab:cyan",
          "black", "magenta", "navy", "lime", "darkgreen"]

roc=True
det=False


#ROC curves, feature comparison
if roc:
    for split in ['time','random']:
        for model in ['Neurlux', 'Transformer', 'JsonGrinder']:
            fig_roc, ax_roc = plt.subplots(1, figsize=(12, 7))
            for name,color in zip(features_names,colors):
                with open(f"./detection_curves_data/{model}_scores_y_{name}_{split}.json","r") as fp:
                    data=json.load(fp)

                scores=data['scores']
                y=data['y']

                RocCurveDisplay.from_predictions(y, scores, name=f'{name}', ax=ax_roc,color=color)
            ax_roc.set_title("Receiver Operating Characteristic (ROC) curves",fontdict={'fontsize':20})
            ax_roc.grid(linestyle="--")
            ax_roc.legend(loc='lower right')
            ax_roc.set_xscale('log')
            ax_roc.tick_params(axis='both', which='major', labelsize=15)
            plt.xlabel('False Positive Rate',fontsize=18)
            plt.ylabel('True Positive Rate',fontsize=18)
            plt.savefig(f"./figure/{model}_Roc_Feature_{split}.pdf")
            plt.show()

# DET curves, feature comparison
if det:
    for split in ['time', 'random']:
        for model in ['Neurlux', 'Transformer', 'JsonGrinder']:
            fig_det, ax_det = plt.subplots(1, figsize=(12, 7))
            for name, color in zip(features_names, colors):
                with open(f"./detection_curves_data/{model}_scores_y_{name}_{split}.json", "r") as fp:
                    data = json.load(fp)

                scores = data['scores']
                y = data['y']

                DetCurveDisplay.from_predictions(y, scores, name=f'{name}', ax=ax_det,color=color)
            ax_det.set_title("Detection Error Tradeoff (DET) curves",fontdict={'fontsize': 20})
            ax_det.grid(linestyle="--")
            ax_det.legend(loc='upper right')
            plt.xlabel('False Positive Rate', fontsize=18)
            plt.ylabel('False Negative Rate', fontsize=18)
            plt.savefig(f"./figure/{model}_Det_Feature_{split}.pdf")
            plt.show()

# %% All feature

# ROC curves
if roc:
    for split in ['time', 'random']:
        fig_roc, ax_roc = plt.subplots(1, figsize=(12, 7))
        for model in ['Neurlux', 'Transformer', 'JsonGrinder']:
            with open(f"./detection_curves_data/{model}_scores_y_All_{split}.json","r") as fp:
                data=json.load(fp)

            scores=data['scores']
            y=data['y']

            RocCurveDisplay.from_predictions(y, scores, name=f'{model}', ax=ax_roc)
        ax_roc.set_title("Receiver Operating Characteristic (ROC) curves",fontdict={'fontsize':27})
        ax_roc.grid(linestyle="--")
        ax_roc.legend(loc='lower right',prop={'size': 20})
        ax_roc.set_xscale('log')
        ax_roc.tick_params(axis='both', which='major', labelsize=20)
        plt.xlabel('False Positive Rate',fontsize=25)
        plt.ylabel('True Positive Rate',fontsize=25)
        plt.savefig(f"./figure/Roc_All_{split}.pdf")
        plt.show()

# DET curve
if det:
    for split in ['time', 'random']:
        fig_det, ax_det = plt.subplots(1, figsize=(12, 7))
        for model in ['Neurlux', 'Transformer', 'JsonGrinder']:
            with open(f"./detection_curves_data/{model}_scores_y_All_{split}.json", "r") as fp:
                data = json.load(fp)

            scores = data['scores']
            y = data['y']

            DetCurveDisplay.from_predictions(y, scores, name=f'{model}', ax=ax_det)
        ax_det.set_title("Detection Error Tradeoff (DET) curves", fontdict={'fontsize': 20})
        ax_det.grid(linestyle="--")
        ax_det.legend(loc='upper right',prop={'size': 20})
        plt.xlabel('False Positive Rate', fontsize=15)
        plt.ylabel('False Negative Rate', fontsize=15)
        plt.savefig(f"./figure/Det_All_{split}.pdf")
        plt.show()