import logging
import os
import random

import numpy as np
import torch


def seed_everything(seed=3407):
    '''Sets the seed for generating random numbers in :pytorch:`PyTorch`,
    :obj:`numpy` and Python.

    Args:
        seed (int): The desired seed.
    '''
    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_logger():
    '''Create logger for the experiment.
    '''
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # create consol handler
    consol_handler = logging.StreamHandler()
    consol_handler.setLevel(logging.INFO)

    # set format
    formatter = logging.Formatter(
        '[%(asctime)s %(filename)s line:%(lineno)d process:%(process)d] %(levelname)s: %(message)s'
    )
    consol_handler.setFormatter(formatter)
    logger.addHandler(consol_handler)
    return logger