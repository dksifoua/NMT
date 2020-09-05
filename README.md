[![Build Status](https://travis-ci.com/dksifoua/NMT.svg?branch=master)](https://travis-ci.com/dksifoua/NMT.svg?branch=master)
[![master](https://codecov.io/gh/dksifoua/NMT/branch/master/graph/badge.svg)](https://codecov.io/gh/dksifoua/NMT)
[![contributions welcome](https://img.shields.io/badge/contributions-welcome-brightgreen.svg?style=flat)](https://github.com/dksifoua/nmt/issues)
[![License: MIT](https://img.shields.io/badge/License-MIT-brightgreen.svg)](https://opensource.org/licenses/MIT)

# Neural Machine Translation

State of the art of Neural Machine Translation with PyTorch and TorchText.

## Quick start
I used the `europarl-v7` parallel corpora to build models. The data is downloadable [here](http://www.statmt.org/europarl/v7/fr-en.tgz)!

- **Install requirements and Download data**

Run commands below:

```shell
$ pip install -r requirements.txt

$ python -m spacy download fr
$ python -m spacy download en
$ python -m spacy download fr_core_news_lg
$ python -m spacy download en_core_web_lg

$ mkdir -p ./checkpoints
$ mkdir -p ./data
$ mkdir -p ./images
$ mkdir -p ./logs

$ wget --no-check-certificate \
    http://www.statmt.org/europarl/v7/fr-en.tgz \
    -O ./data/fr-en.tgz

$ tar -xzvf ./data/fr-en.tgz -C ./data

$ rm -rf ./data/fr-en.tg
```

or simply run:

```shell
$ ./init.sh
```

- **Build datasets**
```shell script
$ python -m scripts.build_datasets --help
usage: build_datasets.py [-h] [--src_lang SRC_LANG] [--dest_lang DEST_LANG]
                         [--n_samples N_SAMPLES] [--min_len MIN_LEN]
                         [--max_len MAX_LEN] [--min_freq MIN_FREQ]
                         [--save SAVE]

Build and save train, validation, and test datasets.

optional arguments:
  -h, --help            show this help message and exit
  --src_lang SRC_LANG   The source language. Default: fr.
  --dest_lang DEST_LANG
                        The destination language. Default: en.
  --n_samples N_SAMPLES
                        The number of samples. Default: 200000.
  --min_len MIN_LEN     The min length of an example. Default: 10.
  --max_len MAX_LEN     The max length of an example. Default: 25.
  --min_freq MIN_FREQ   The min freq of an words in vocabulary. Default: 5.
  --save SAVE           To whether or not save datasets and fields.

```

## Modeling

### Encoder-Decoder architecture
The encoder-decoder architecture is a neural network design pattern. As shown in the figure below, the architecture is 
partitioned into two parts, the encoder and the decoder. The encoder's role is to encode the inputs into state, which 
often contains several tensors. Then the state is passed into the decoder to generate the outputs. In machine 
translation, the encoder transforms a source sentence, e.g., `Hello world!.`, into state vector, that captures its 
semantic information. The decoder then uses this state to generate the translated target sentence, e.g., 
`Bonjour le monde !`

### Sequence-to-Sequence model
The sequence-to-sequence model is based on the encoder-decoder architecture to generate a sequence output for a sequence
 input, as demonstrated below. Both the encoder and the decoder commonly use recurrent neural networks (RNNs) to handle 
 sequence inputs of variable length. The hidden state of the encoder is used directly to initialize the decoder hidden 
 state to pass information from the encoder to the decoder. In this project, I tried several sequence-to-sequence models
  with LSTMs, Attention mechanisms, CNNs and Transformers.

## Results

### Training

```shell
$ python -m scripts.train --help
```

### Evaluation

```shell
$ python -m scripts.evaluate --help
```

## References
