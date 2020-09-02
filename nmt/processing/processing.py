import os
import re
import dill
import tqdm
import pickle
from unicodedata import normalize
import numpy as np
from torchtext.data import Example, Field
from nmt.config.global_config import GlobalConfig
from typing import Dict, List, Tuple, ByteString


def load_and_clean(src_lang: str, dest_lang: str, n_samples: int, min_len: int, max_len: int) -> List[Dict[str, str]]:
    data_fr = read_file(os.path.join(GlobalConfig.DATA_PATH, f'europarl-v7.fr-en.{src_lang}'))
    data_en = read_file(os.path.join(GlobalConfig.DATA_PATH, f'europarl-v7.fr-en.{dest_lang}'))
    assert len(data_fr) == len(data_en)
    print(f'Number of examples: {len(data_fr):,}')
    indexes = np.random.choice(range(len(data_fr)), size=n_samples, replace=False)
    pairs = [*zip(clean_lines([data_fr[index] for index in indexes]),
                  clean_lines([data_en[index] for index in indexes]))]
    pairs = [*map(lambda x: {f'{src_lang}': x[0], f'{dest_lang}': x[1]}, pairs)]
    print(f'Number of examples after sampling: {len(pairs):,}')
    print(f'Example:\n\tFR => {pairs[0][f"{src_lang}"]}\n\tEN => {pairs[0][f"{dest_lang}"]}')
    pairs = [*filter(lambda pair: min_len <= len(pair['fr'].split()) <= max_len and min_len <= len(
        pair['fr'].split()) <= max_len, pairs)]
    print(f'Number of examples after filtering: {len(pairs):,}')
    return pairs


def build_examples(data: List[Dict[str, str]], src_lang: str, dest_lang: str) -> Tuple[List[Example], Field, Field]:
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
    print(f'Number of examples: {len(examples):,}')
    return examples, src_field, dest_field


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


def save_data(data: List[Dict[str, str]], filename: str) -> None:
    pickle.dump(data, open(os.path.join(GlobalConfig.DATA_PATH, f'{filename}.pkl'), mode='wb'))
    print(f'Saved {filename} data.')


def load_data(filename: str) -> List[Dict[str, str]]:
    return pickle.load(open(os.path.join(GlobalConfig.DATA_PATH, f'{filename}.pkl'), mode='rb'))


def save_field(field: Field, filename: str) -> None:
    dill.dump(field, open(os.path.join(GlobalConfig.DATA_PATH, f'{filename}.field'), mode='wb'))
    print(f'Saved {filename} field.')


def load_field(filename: str) -> Field:
    return dill.load(os.path.join(GlobalConfig.DATA_PATH, f'{filename}.field'))


def save_examples(example: Example, filename: str) -> None:
    dill.dump(example, open(os.path.join(GlobalConfig.DATA_PATH, f'{filename}.examples'), mode='wb'))
    print(f'Saved {filename} field.')


def load_examples(filename: str) -> Field:
    return dill.load(os.path.join(GlobalConfig.DATA_PATH, f'{filename}.examples'))
