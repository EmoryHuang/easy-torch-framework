# 一个自己写的 PyTorch 简单框架范例

使用 Pytorch 实现神经网络模型的一般流程包括：

1. 准备数据
2. 定义模型
3. 训练模型
4. 评估模型
5. 使用模型
6. 保存模型

**对新手来说，其中最困难的部分实际上是准备数据过程。**

另外一方面，Pytorch 通常需要用户编写自定义训练循环，训练循环的代码风格因人而异。

现在市面上不乏对 PyTorch 进行二次包装的框架，例如：[Lightning](https://www.pytorchlightning.ai/)，huggingface。诚然这些 Trainer 功能足够强大，为了吸引更多人使用，加入尽可能多的功能，比如基本的日志、tensorboard、断点重训、训练时验证等。

在最开始的时候我也尝试过去使用它来写模型，然而大多数时候我们并不需要这么多的功能，我们不需要各种各样的接口，有时候甚至不如自己写一个 Loop，毕竟我能确保我清楚所有的流程，也能够在必要的时候进行适当的“魔改”。

但是一个简单清晰的 Trainer 确实有不少好处，起码代码看着优美多了，下面是我自己写模型时候一个简单框架范例，并没有添加什么乱七八糟的外部库，只是简单定义了一个 Trainer 类，同时对整体的训练流程进行了总结。

当然，这只是我自己习惯的一些写法（叠甲叠甲）🤣

你可以在 []() 找到完整的代码。

## Overview

```text
.
├── Dataset/                    # 数据文件夹
├── Model/                      # 模型文件夹
├── main.py                     # main 函数
├── readme.md
├── requirements.txt            # 环境要求
└── src
    ├── args.py                 # 参数配置
    ├── dataloader.py           # 数据加载文件，控制所有数据
    ├── dataset.py              # 数据结构文件，包含所有的自定义数据集
    ├── model.py                # 模型文件，包含所有的模型
    ├── trainer.py              # trainer 类
    └── utils.py                # 工具函数文件
```

下面我简单说一下每个文件都干了什么。

## main 函数

```py
import src.utils as utils
from src.args import get_args
from src.dataloader import Mydataloader
from src.model import MyModel
from src.trainer import Trainer

# load config
config = get_args()

# init logger
logger = utils.init_logger()

# init seed
utils.seed_everything(3407)


def main():
    logger.info(f'start loading data: {config.dataset} ...')
    my_loader = Mydataloader(config)
    my_loader.load()
    logger.info('load data done!')

    logger.info('start creating dataset ...')
    my_loader.create_dataset()
    logger.info('creating dataset done!')

    logger.info(f'start loading model ...')
    logger.info(f'mode: {config.mode}')
    model = MyModel(config)
    logger.info('load model done!')
    logger.info(f'epochs={config.epochs}, '
                f'batch_size={config.batch_size}, '
                f'hidden_size={config.hidden_size}')

    trainer = Trainer(config=config, logger=logger, gpu=config.gpu)
    if config.mode == 'train':
        trainer.train(model=model, dataloader=my_loader)
    else:
        trainer.test(model=model, dataloader=my_loader, model_path=config.model_path)


if __name__ == '__main__':
    main()
```

main 函数其实非常简单，大致上也不需要进行修改，基本都是通用的。

1. 首先`7-14`行，分别创建了 **参数**（`config`），**日志** (`logger`) 对象，并且指定了随机种子。
2. 示例化了`Mydataloader`类，同时调用了`load()`方法和`create_dataset()`方法。

> `Mydataloader` 类主要用于控制所有和输入数据相关的东西，包括数据的加载，数据集 `Dataset` 的创建。

3. 接着初始化模型 `MyModel`，模型文件存放在 `./src/model.py` 内。
4. 最后创建 `Trainer` 类对象，里面包含了训练的具体循环。

基本上写完之后就不需要关心 main 函数了。

## src.args

参数配置这里其实都是一样的，我只是把它单独拿出来而已。

当然，有些人也可能习惯用 `yaml` 来配置参数，道理是一样的。

```py
import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Training arguments.")
    # commom
    parser.add_argument('--gpu', default=-1, type=int, help='the gpu to use.')
    parser.add_argument('--mode', default='train', type=str, help='train/test')
    ...

    parser.add_argument('--dataset',
                        default='dataset-1',
                        type=str,
                        help='Choose your dataset')

    args = parser.parse_args()

    # Adjustment according to the dataset
    if args.dataset == 'dataset-1':
        args.alpha = 0.5
        args.beta = 0.5
    elif args.dataset == 'dataset-2':
        args.alpha = 0.3
        args.beta = 0.3

    return args
```

## src.dataloader

`dataloader.py` 文件里面只定义了 `Mydataloader` 类，用于控制所有和数据相关的东西。

```py
class Mydataloader():
    def __init__(self, config): ...
    def load(self): ...                 # 读取数据
    def create_dataset(self): ...       # 创建数据集，使用自定义的 Dataset 进行包装
    def train_dataloader(self): ...     # 训练集 DataLoader
    def valid_dataloader(self): ...     # 验证集 DataLoader
    def test_dataloader(self): ...      # 测试集 DataLoader
    def split_dataset(self, data, rate): ...    # 自定义数据集分割函数
    @staticmethod
    def read_dataset_1():...            # 针对数据集 1 的自定义数据处理函数
    @staticmethod
    def read_dataset_2():...            # 针对数据集 2 的自定义数据处理函数
```

### load() & create_dataset()

`load()` 函数并没有实现什么具体的功能，只是一个加载数据的入口。

`create_dataset()` 函数的主要作用就是将读到的数据包装成自定义的`Dataset`，对数据集进行划分，同时对数据进行保存。

```py
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
```

### dataloader

```py
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
```

为了后面方便，直接把三个 `DataLoader` 写成函数，这样在后面调用的时候只需要这样写就行了：

```py
train_dl = dataloader.train_dataloader()
valid_dl = dataloader.valid_dataloader()
test_dl = dataloader.test_dataloader()
```

### custom

下面这三个函数我这里都是举了个例子，基本都是针对数据集的加载具体处理方法：

```py
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
```

## dataset

`dataset.py` 文件里面存放了自定义的数据结构，这个文件还是要修改为你自己的，也没什么好说的。

```py
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
```

## model

`model.py` 文件同理，存放模型文件，这个也应该都是这么做的。简单贴个代码：

```py
class MyModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer1 = nn.Linear(4, 50)
        self.layer2 = nn.Linear(50, 50)
        self.layer3 = nn.Linear(50, 3)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.softmax(self.layer3(x), dim=1)
        return x
```

## trainer

`trainer.py` 文件里定义了 `Trainer` 类，包含下面几个函数：

```py
class Trainer:
    def __init__(self, config, logger=None, gpu=-1): ...
    def train(self, model, dataloader): ...
    def train_epoch(self, epoch, model, train_dl, optimizer, scheduler, criterion): ...
    @torch.no_grad()
    def valid_epoch(self, epoch, model, valid_dl, criterion): ...
    @torch.no_grad()
    def test(self, model, dataloader, model_path): ...
```

### train

在 `main` 函数里调用了 `train()` 后，就会进行下面的步骤：

1. 定义优化器和损失函数
2. 创建 DataLoader
3. 准备模型
4. 进入循环
   1. `train_epoch()`
   2. `valid_epoch()`
   3. 保存模型

```py
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
```

### train_epoch

这里以 `train_epoch` 为例，熟悉训练过程的话你就会发现这里其实就是正常的训练流程，基本什么也没变。

```py
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
```

## utils

工具类就不说了，不知道往哪放就放这里。

## 其他

另外，个人一般还会有一个 `preprocess.py` 文件用来做数据处理，因为这也是很大的一部分，这样在 `dataloader` 里面加载数据的时候就会干净很多。

## 总结

看完之后你会发现，其实就是原生的 PyTorch，也完全没有变化，我只是按我自己比较喜欢、习惯的方式对功能进行分门别类，让整理的流程变得更加简单和清晰。这样我下次写模型的时候基本只需要关注数据部分以及模型部分，针对模型训练流程上的东西大部分只需要复用这个模式就可以了。
