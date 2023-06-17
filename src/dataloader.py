from pathlib import Path

import torch
from sklearn import datasets
from torch.utils.data import DataLoader, random_split

from src.dataset import MyDataset


class Mydataloader():
    '''load data
    '''
    def __init__(self, config):
        self.config = config
        self.dataset_dir = Path(config.dataset_dir)
        if not self.dataset_dir.exists():
            self.dataset_dir.mkdir()

    def load(self):
        '''load dataset according to config.dataset
        '''
        if self.config.dataset.lower() == 'dataset-1':
            # do custom dataset-1 process
            self.data = Mydataloader.read_dataset_1()
            # or
            # self.data_train = Mydataloader.read_dataset_1('train')
            # self.data_valid = Mydataloader.read_dataset_1('valid')
            # self.data_test = Mydataloader.read_dataset_1('test')

        elif self.config.dataset.lower() == 'dataset-2':
            # do custom dataset-2 process
            self.data = Mydataloader.read_dataset_2()

    def create_dataset(self):
        '''create dataset according to customize dataset
        '''
        dataset_train_path = self.dataset_dir / f'{self.config.dataset}_train.pkl'
        dataset_valid_path = self.dataset_dir / f'{self.config.dataset}_valid.pkl'
        dataset_test_path = self.dataset_dir / f'{self.config.dataset}_test.pkl'
        if dataset_test_path.exists():
            if self.config.mode == 'train':
                self.dataset_train = torch.load(dataset_train_path)
                self.dataset_valid = torch.load(dataset_valid_path)
            elif self.config.mode == 'test':
                self.dataset_test = torch.load(dataset_test_path)
            return

        # create dataset
        dataset = MyDataset(self.config, self.data)

        # split dataset
        self.dataset_train, dataset_test = self.split_dataset(dataset, rate=0.8)
        self.dataset_valid, self.dataset_test = self.split_dataset(dataset_test, rate=0.5)
        torch.save(self.dataset_train, dataset_train_path)
        torch.save(self.dataset_valid, dataset_valid_path)
        torch.save(self.dataset_test, dataset_test_path)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.dataset_train,
            batch_size=self.config.batch_size,
            pin_memory=True,
            shuffle=True,
        )

    def valid_dataloader(self):
        return DataLoader(
            dataset=self.dataset_valid,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.dataset_test,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

    def split_dataset(self, data, rate):
        '''custom split function
        '''
        train_size = int(rate * len(data))
        test_size = len(data) - train_size
        data_train, data_test = random_split(data, [train_size, test_size])
        return data_train, data_test

    @staticmethod
    def read_dataset_1():
        '''load the dataset here or process the dataset
        '''
        data = datasets.load_iris()
        return data

    @staticmethod
    def read_dataset_2():
        '''load the dataset here or process the dataset
        '''
        pass