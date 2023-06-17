import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Training arguments.")
    # commom
    parser.add_argument('--gpu', default=-1, type=int, help='the gpu to use.')
    parser.add_argument('--mode', default='train', type=str, help='train/test')
    parser.add_argument('--dataset_dir',
                        default='./Dataset',
                        type=str,
                        help='Database dir.')
    parser.add_argument('--model_dir', default='./Model', type=str, help='Model dir.')

    parser.add_argument('--epochs', default=50, type=int, help='Train epochs.')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size.')
    parser.add_argument('--hidden_size', default=32, type=int, help='Hidden size.')
    parser.add_argument('--learning_rate',
                        default=0.001,
                        type=float,
                        help='Learning rate.')
    parser.add_argument('--valid_freq', default=5, type=int, help='Valid frequence.')
    parser.add_argument('--model_path',
                        default='./Model/model_50.pkl',
                        type=str,
                        help='Test model path')

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