import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, config, data):
        super().__init__()
        self.config = config
        self.data = data

    def __len__(self):
        return len(self.data.data)

    def __getitem__(self, idx):
        feature = torch.Tensor(self.data.data)[idx]
        label = torch.LongTensor(self.data.target)[idx]
        return feature, label