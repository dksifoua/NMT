import argparse
from nmt.config.dataset_config import DatasetConfig
from nmt.processing.processing import build_datasets, build_examples, build_vocab, load_and_clean, save_datasets
from nmt.processing.processing import save_field
from nmt.utils.logger import Logger

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
    parser.add_argument('--min_freq', action='store', type=int, default=DatasetConfig.MIN_FREQ,
                        help=f'The min freq of an words in vocabulary. Default: {DatasetConfig.MIN_FREQ}.')
    parser.add_argument('--save', action='store', type=bool, default=True,
                        help='To whether or not save datasets and fields.')
    args = parser.parse_args()
    logger = Logger(name='BuildDatasets')
    pairs = load_and_clean(src_lang=args.src_lang, dest_lang=args.dest_lang, n_samples=args.n_samples,
                           min_len=args.min_len, max_len=args.max_len, logger=logger)
    examples, src_field, dest_field = build_examples(data=pairs, src_lang=args.src_lang, dest_lang=args.dest_lang,
                                                     logger=logger)
    train_dataset, valid_dataset, test_dataset = build_datasets(examples=examples, src_field=src_field,
                                                                dest_field=dest_field, logger=logger)
    build_vocab(field=src_field, lang=args.src_lang, train_data=train_dataset, min_freq=args.min_freq,
                special_tokens=['<unk>', '<pad>'], logger=logger)
    build_vocab(field=dest_field, lang=args.dest_lang, train_data=train_dataset, min_freq=args.min_freq,
                special_tokens=['<sos>', '<eos>', '<unk>', '<pad>'], logger=logger)
    save_field(field=src_field, filename=args.src_lang, logger=logger)
    save_field(field=dest_field, filename=args.dest_lang, logger=logger)
    save_datasets(datasets=[train_dataset, valid_dataset, test_dataset], filenames=['train', 'valid', 'test'],
                  logger=logger)
