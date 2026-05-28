# 自定义数据集
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import config


class TranslationDataset(Dataset):
    def __init__(self, data_path):
        self.data = pd.read_json(data_path, lines=True).to_dict(orient='records')

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            input_tensor = torch.tensor(self.data[index]['zh'],dtype=torch.long)
            target_tensor = torch.tensor(self.data[index]['en'],dtype=torch.long)

            return input_tensor, target_tensor



def get_dataloader(train=True):
    data_path = config.PROCESSED_DATA_DIR / ('indexed_train.jsonl' if train else 'indexed_test.jsonl')
    dataset = TranslationDataset(data_path)
    return DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)

if __name__ == '__main__':
  train_loader = get_dataloader(train=True)
  for inputs, targets in train_loader:
    print(inputs.shape)
    print(targets.shape)
    break