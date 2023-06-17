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

#! command example
# train
# python main.py --mode=train --gpu=1
# nohup python main.py --mode=train --gpu=2 > ./main.log 2>&1 &

# test
# python main.py --mode=test --model_path=./Model/model_50.pkl