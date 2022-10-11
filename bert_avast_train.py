import torch
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel,\
    BertTokenizer, BertModel,DistilBertModel, DistilBertConfig, DistilBertTokenizer,DistilBertForSequenceClassification,DistilBertTokenizerFast
import time
from transformers import AutoTokenizer, AutoModel
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from DatasetClasses import AvastDataset
from utils import *
import copy
from torch.nn.functional import softmax
import pickle

def train_model(model,dataloaders, optimizer, num_epochs=50):
    since = time.time()

    history = {'Train': [], 'Val': []}

    best_model_wts = None
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch + 1, num_epochs))
        for phase in ['Train', 'Val']:
            if phase == 'Train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0  # Initialized loss
            running_corrects = 0  # Initialized correct checks

            for batch_idx, batch in enumerate(dataloaders[phase]):

                # Prepare data
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                # Forward
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss, logits = outputs['loss'], outputs['logits']

                # Backward
                if phase == 'Train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                # Statistics
                y_pred=torch.argmax(softmax(logits,dim=1),1)
                running_loss += float(loss)
                running_corrects += torch.sum(y_pred == labels)

            # Loss and Accuracy Evaluation
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(' ---- {} [Loss: {:.4f} Acc: {:.4f}]'.format(phase, epoch_loss, epoch_acc))
            history[phase].append((epoch_loss, epoch_acc.cpu().item()))
            # deep copy the model
            if phase == 'Val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                # model.to('cpu')
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best Validation Acc: {:.2f}%'.format(best_acc * 100))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model,history


if __name__ == '__main__':
    # Hyperparameters
    # MAX_LEN = 128
    BATCH_SIZE = 64
    EPOCHS = 3
    LEARNING_RATE = 1e-05

    with open('vocab_avast.txt', 'r') as fp:
        vocab=fp.read()
        vocab=vocab.split('\n')

    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
    tokenizer = tokenizer.train_new_from_iterator(vocab, vocab_size=10000)
    # tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")

    df_tr = pd.read_pickle('Avast/avast_dataframe_train.pkl')

    # Subset
    df_train=create_subset(df_tr,n_sample_per_class=10)


    # Split train, validation and test
    df_tr, df_val = split_train_val(df_train,val_frac=0.1)

    print(f'Train samples: {len(df_tr)}')
    print(f'Val samples: {len(df_val)}')

    print(f"Number of malware families (Train): {len(set(df_train['classification_family']))}")
    print(f"Number of malware families (Validation): {len(set(df_val['classification_family']))}")

    for MAX_LEN in [128,256,512]:
        print(f'MAX LEN: {MAX_LEN}')
        train_dataset = AvastDataset(df_tr, tokenizer, MAX_LEN)
        valid_dataset = AvastDataset(df_val, tokenizer, MAX_LEN)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False)

        # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        device = torch.device('cpu')
        # print(device)

        # model = BERTClass()
        model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased',num_labels=10)
        model.to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=LEARNING_RATE)

        dataloaders = {'Train': train_loader, 'Val': val_loader}
        print('START TRAINING')
        model,history=train_model(model,dataloaders,optimizer,num_epochs=EPOCHS)

        # Save model and history
        # torch.save(model.state_dict(), f'Avast/best_model_{MAX_LEN}.pt')
        # f = open(f'Avast/history_{MAX_LEN}.pkl', 'wb')
        # pickle.dump(history, f)
        # f.close()

        # Plot the train
        #plot_history(history)



