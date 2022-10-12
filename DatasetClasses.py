import torch

class AvastDataset(torch.utils.data.Dataset):
    classes = ['Adload','Emotet', 'HarHar', 'Lokibot','njRAT','Qakbot','Swisyn','Trickbot','Ursnif','Zeus']

    def __init__(self, df, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = df['words'].to_numpy()
        self.labels = [AvastDataset.classes.index(x) for x in df['classification_family'].values]
        self.max_len = max_len

    def __getitem__(self, idx):
        data = self.data[idx]
        # print(len(data))
        item = self.tokenizer(data,truncation=True, padding='max_length',max_length=self.max_len,return_tensors='pt')
        item['input_ids']=item['input_ids'].squeeze()
        item['attention_mask']=item['attention_mask'].squeeze()
        # item = {key: torch.tensor(val[idx]) for key, val in encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

if __name__ == '__main__':
    pass