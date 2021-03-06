import os
import re
import dill
import tqdm
import random
from unicodedata import normalize
import numpy as np
import torch
from torchtext.data import Dataset, Example, Field
from nmt.config.global_config import GlobalConfig
from nmt.utils.logger import Logger
from typing import Dict, List, Tuple, ByteString

__all__ = ['build_datasets', 'build_examples', 'build_vocab', 'load_and_clean', 'load_dataset', 'load_field',
           'save_datasets', 'save_field']


def read_file(filepath: str) -> List[ByteString]:
    try:
        with open(filepath, mode='rb') as file:
            content = file.readlines()
        return content
    except Exception:
        raise Exception(f'File {filepath} does not exist')


def clean_lines(lines: List[ByteString]) -> List[str]:
    cleaned = []
    for line in tqdm.tqdm(lines):
        line = line.decode('utf-8', 'ignore')
        line = line.strip()
        line = normalize('NFKD', line)
        line = re.sub(' +', ' ', line)
        cleaned.append(line)
    return cleaned


def load_and_clean(src_lang: str, dest_lang: str, n_samples: int, min_len: int, max_len: int,
                   logger: Logger) -> List[Dict[str, str]]:
    np.random.seed(GlobalConfig.SEED)
    logger.info('LOAD AND CLEAN DATA')
    data_fr = read_file(os.path.join(GlobalConfig.DATA_PATH, f'europarl-v7.fr-en.{src_lang}'))
    data_en = read_file(os.path.join(GlobalConfig.DATA_PATH, f'europarl-v7.fr-en.{dest_lang}'))
    assert len(data_fr) == len(data_en)
    logger.info(f'Number of examples: {len(data_fr):,}')
    indexes = np.random.choice(range(len(data_fr)), size=n_samples, replace=False)
    pairs = [*zip(clean_lines([data_fr[index] for index in indexes]),
                  clean_lines([data_en[index] for index in indexes]))]
    pairs = [*map(lambda x: {f'{src_lang}': x[0], f'{dest_lang}': x[1]}, pairs)]
    logger.info(f'Number of examples after sampling: {len(pairs):,}')
    pairs = [*filter(lambda pair: min_len <= len(pair['fr'].split()) <= max_len and min_len <= len(
        pair['fr'].split()) <= max_len, pairs)]
    logger.info(f'Number of examples after filtering: {len(pairs):,}')
    return pairs


def build_examples(data: List[Dict[str, str]], src_lang: str, dest_lang: str,
                   logger: Logger) -> Tuple[List[Example], Field, Field]:
    logger.info('BUILD EXAMPLES')
    src_field = Field(lower=True, tokenize='spacy', tokenizer_language=src_lang, include_lengths=True)
    dest_field = Field(init_token='<sos>', eos_token='<eos>', lower=True, tokenize='spacy',
                       tokenizer_language=dest_lang, include_lengths=True)
    examples = [Example.fromdict(
        data=pair,
        fields={
            f'{src_lang}': ('src', src_field),
            f'{dest_lang}': ('dest', dest_field)
        }
    ) for pair in tqdm.tqdm(data)]
    logger.info(f'Number of examples: {len(examples):,}')
    return examples, src_field, dest_field


def build_datasets(examples: List[Example], src_field: Field, dest_field: Field,
                   logger: Logger) -> Tuple[Dataset, Dataset, Dataset]:
    random.seed(GlobalConfig.SEED)
    logger.info('BUILD DATASETS')
    data = Dataset(examples, fields={'src': src_field, 'dest': dest_field})
    train_data, valid_data, test_data = data.split(split_ratio=[0.9, 0.05, 0.05])
    logger.info(f'train set size: {len(train_data.examples):,}')
    logger.info(f'valid set size: {len(valid_data.examples):,}')
    logger.info(f'test set size: {len(test_data.examples):,}')
    return train_data, valid_data, test_data


def build_vocab(field: Field, lang: str, train_data: Dataset, min_freq: int, special_tokens: List[str],
                logger: Logger) -> None:
    field.build_vocab(train_data, min_freq=min_freq, specials=special_tokens)
    logger.info(f'Length of {lang} built vocabulary: {len(field.vocab):,}')


def save_datasets(datasets: List[Dataset], filenames: List[str], logger: Logger) -> None:
    assert len(datasets) == len(filenames) == 3
    logger.info('SAVE DATASETS')
    for filename, dataset in zip(filenames, datasets):
        save_examples(examples=dataset.examples, filename=filename, logger=logger)


def load_dataset(filename: str, src_field: Field, dest_field: Field, logger: Logger) -> Dataset:
    logger.info(f'Load {filename} dataset')
    examples = load_examples(filename=filename)
    dataset = Dataset(examples, fields={'src': src_field, 'dest': dest_field})
    return dataset


def save_field(field: Field, filename: str, logger: Logger) -> None:
    with open(os.path.join(GlobalConfig.DATA_PATH, f'{filename}.field'), mode='wb') as file:
        torch.save(field, file, pickle_module=dill)
    logger.info(f'Saved {filename} field.')


def load_field(filename: str) -> Field:
    with open(os.path.join(GlobalConfig.DATA_PATH, f'{filename}.field'), mode='rb') as file:
        return torch.load(file, pickle_module=dill)


def save_examples(examples: List[Example], filename: str, logger: Logger) -> None:
    with open(os.path.join(GlobalConfig.DATA_PATH, f'{filename}.examples'), mode='wb') as file:
        torch.save(examples, file, pickle_module=dill)
    logger.info(f'Saved {filename} examples.')


def load_examples(filename: str) -> Field:
    with open(os.path.join(GlobalConfig.DATA_PATH, f'{filename}.examples'), mode='rb') as file:
        return torch.load(file, pickle_module=dill)
