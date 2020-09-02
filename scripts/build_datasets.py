import argparse
from nmt.config.dataset_config import DatasetConfig

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build and save train, validation, and test datasets.')
    parser.add_argument('--src_lang', action='store', type=str, default=DatasetConfig.SRC_LANG,
                        help=f'The source language. Default: {DatasetConfig.SRC_LANG}.')
    parser.add_argument('--dest_lang', action='store', type=str, default=DatasetConfig.DEST_LANG,
                        help=f'The destination language. Default: {DatasetConfig.DEST_LANG}.')
    parser.add_argument('--n_samples', action='store', type=int, default=DatasetConfig.N_SAMPLES,
                        help=f'The number of samples. Default: {DatasetConfig.N_SAMPLES}.')
    parser.add_argument('--min_len', action='store', type=int, default=DatasetConfig.MIN_LEN,
                        help=f'The min length of an example. Default: {DatasetConfig.MIN_LEN}.')
    parser.add_argument('--max_len', action='store', type=int, default=DatasetConfig.MAX_LEN,
                        help=f'The max length of an example. Default: {DatasetConfig.MAX_LEN}.')
    parser.add_argument('--save', action='store', type=bool, default=True,
                        help='To whether or not save datasets and fields.')
    args = parser.parse_args()
    pass
