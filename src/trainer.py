from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.utils import init_logger


class Trainer:
    def __init__(self, config, logger=None, gpu=-1):
        self.config = config
        self.logger = logger if logger is not None else init_logger()

        if gpu == -1:
            self.device = torch.device("cpu")
        else:
            self.device = torch.device(
                f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

        self.model_dir = Path(self.config.model_dir)
        if not self.model_dir.exists():
            self.model_dir.mkdir()

    def train(self, model, dataloader):
        # prepare optimizer and criterion
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        criterion = torch.nn.CrossEntropyLoss()

        # prepare dataloader
        train_dl = dataloader.train_dataloader()
        valid_dl = dataloader.valid_dataloader()

        # prepare model
        model = model.to(self.device)

        self.logger.info('start training...')
        self.logger.info(f'device: {self.device}')
        for epoch in range(self.config.epochs):
            # train epoch
            self.train_epoch(epoch, model, train_dl, optimizer, scheduler, criterion)

            # valid epoch
            self.valid_epoch(epoch, model, valid_dl, criterion)

            # save model
            torch.save(model.state_dict(), self.model_dir / f"model_{epoch+1}.pkl")
        self.logger.info('training done!')

    def train_epoch(self, epoch, model, train_dl, optimizer, scheduler, criterion):
        model.train()
        train_loss = []
        tbar = tqdm(train_dl, total=len(train_dl), desc='Training')
        for idx, dl in enumerate(tbar):
            feature, label = dl
            feature = feature.to(self.device)
            label = label.to(self.device)

            optimizer.zero_grad()
            pred = model(feature)
            loss = criterion(pred, label)
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            # update bar
            tbar.set_description(f'Epoch [{epoch + 1}/{self.config.epochs}]')
            tbar.set_postfix(loss=f'{np.mean(train_loss):.4f}',
                             lr=scheduler.get_last_lr()[0])
        scheduler.step()

    @torch.no_grad()
    def valid_epoch(self, epoch, model, valid_dl, criterion):
        if (epoch + 1) % self.config.valid_freq != 0:
            return

        model.eval()
        valid_loss, valid_acc = [], []
        vbar = tqdm(valid_dl, desc='valid', total=len(valid_dl))
        for idx, dl in enumerate(vbar):
            feature, label = dl
            feature = feature.to(self.device)
            label = label.to(self.device)

            pred = model(feature)
            loss = criterion(pred, label)
            acc = (pred.argmax(dim=1) == label).float().mean()

            valid_loss.append(loss.item())
            valid_acc.append(acc.item())

            # update bar
            vbar.set_postfix(loss=f'{np.mean(valid_loss):.4f}',
                             acc=f'{np.mean(valid_acc):.4f}')

    @torch.no_grad()
    def test(self, model, dataloader, model_path):
        # prepare dataloader
        test_dl = dataloader.test_dataloader()

        # prepare model
        model = model.to(self.device)
        model.load_state_dict(torch.load(model_path))
        model.eval()

        test_acc = []
        tbar = tqdm(test_dl, desc='test', total=len(test_dl))
        self.logger.info('start testing...')
        for idx, dl in enumerate(tbar):
            feature, label = dl
            feature = feature.to(self.device)
            label = label.to(self.device)

            pred = model(feature)
            acc = (pred.argmax(dim=1) == label).float().mean()
            test_acc.append(acc.item())

            # update bar
            tbar.set_postfix(acc=f'{np.mean(test_acc):.4f}')
        self.logger.info('testing done.')
        self.logger.info('-------------------------------------')
        self.logger.info('test result:')
        self.logger.info(f'Accuracy: {np.mean(test_acc):.4f}')