import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from nmt.config.dataset_config import DatasetConfig
from nmt.config.global_config import GlobalConfig
from nmt.config.train_config import TrainConfig
from nmt.config.model_config import EncoderLSTMConfig, DecoderLSTMConfig
from nmt.processing.processing import load_dataset, load_field
from nmt.train.trainer import Trainer
from nmt.train.train_utils import count_parameters
from nmt.utils.logger import Logger
from nmt.utils.utils import seed_everything
from typing import Any


def init_seq_to_seq_lstm_model(_module: Any, _src_vocab_size: int, _dest_vocab_size: int,
                               _device: torch.device) -> nn.Module:
    encoder = getattr(_module, 'EncoderLayerLSTM')(embedding_size=EncoderLSTMConfig.EMBEDDING_SIZE,
                                                   hidden_size=EncoderLSTMConfig.HIDDEN_SIZE,
                                                   vocab_size=_src_vocab_size, n_layers=EncoderLSTMConfig.N_LAYERS,
                                                   dropout=EncoderLSTMConfig.EMBEDDING_DROPOUT,
                                                   recurrent_dropout=EncoderLSTMConfig.REC_DROPOUT)
    decoder = getattr(_module, 'DecoderLayerLSTM')(embedding_size=DecoderLSTMConfig.EMBEDDING_SIZE,
                                                   hidden_size=DecoderLSTMConfig.HIDDEN_SIZE,
                                                   vocab_size=_dest_vocab_size, n_layers=DecoderLSTMConfig.N_LAYERS,
                                                   embedding_dropout=DecoderLSTMConfig.EMBEDDING_DROPOUT,
                                                   recurrent_dropout=DecoderLSTMConfig.REC_DROPOUT)
    return getattr(_module, 'SeqToSeqLSTM')(encoder=encoder, decoder=decoder, device=_device)


def init_seq_to_seq_bi_lstm_model(_module: Any, _src_vocab_size: int, _dest_vocab_size: int,
                                  _device: torch.device) -> nn.Module:
    encoder = getattr(_module, 'EncoderLayerBiLSTM')(embedding_size=EncoderLSTMConfig.EMBEDDING_SIZE,
                                                     hidden_size=EncoderLSTMConfig.HIDDEN_SIZE,
                                                     vocab_size=_src_vocab_size, n_layers=EncoderLSTMConfig.N_LAYERS,
                                                     dropout=EncoderLSTMConfig.EMBEDDING_DROPOUT,
                                                     recurrent_dropout=EncoderLSTMConfig.REC_DROPOUT)
    decoder = getattr(_module, 'DecoderLayerLSTM')(embedding_size=DecoderLSTMConfig.EMBEDDING_SIZE,
                                                   hidden_size=DecoderLSTMConfig.HIDDEN_SIZE,
                                                   vocab_size=_dest_vocab_size, n_layers=DecoderLSTMConfig.N_LAYERS,
                                                   embedding_dropout=DecoderLSTMConfig.EMBEDDING_DROPOUT,
                                                   recurrent_dropout=DecoderLSTMConfig.REC_DROPOUT)
    return getattr(_module, 'SeqToSeqBiLSTM')(encoder=encoder, decoder=decoder, device=_device)


def init_seq_to_seq_luong_attn_model(_module: Any, _src_vocab_size: int, _dest_vocab_size: int, _device: torch.device,
                                     _pad_index: int) -> nn.Module:
    encoder = getattr(_module, 'EncoderLayerBiLSTM')(embedding_size=EncoderLSTMConfig.EMBEDDING_SIZE,
                                                     hidden_size=EncoderLSTMConfig.HIDDEN_SIZE,
                                                     vocab_size=_src_vocab_size, n_layers=EncoderLSTMConfig.N_LAYERS,
                                                     dropout=EncoderLSTMConfig.EMBEDDING_DROPOUT,
                                                     recurrent_dropout=EncoderLSTMConfig.REC_DROPOUT)
    attention = getattr(_module, 'LuongAttentionLayer')(hidden_size=EncoderLSTMConfig.HIDDEN_SIZE)
    decoder = getattr(_module, 'LuongDecoderLayerLSTM')(embedding_size=DecoderLSTMConfig.EMBEDDING_SIZE,
                                                        hidden_size=DecoderLSTMConfig.HIDDEN_SIZE,
                                                        vocab_size=_dest_vocab_size,
                                                        n_layers=DecoderLSTMConfig.N_LAYERS,
                                                        dropout=DecoderLSTMConfig.DROPOUT,
                                                        embedding_dropout=DecoderLSTMConfig.EMBEDDING_DROPOUT,
                                                        recurrent_dropout=DecoderLSTMConfig.REC_DROPOUT,
                                                        attention_layer=attention)
    return getattr(_module, 'SeqToSeqLuongAttentionLSTM')(encoder=encoder, decoder=decoder, device=_device,
                                                          pad_index=_pad_index)


def init_seq_to_seq_badhanau_attn_model(_module: Any, _src_vocab_size: int, _dest_vocab_size: int,
                                        _device: torch.device, _pad_index: int) -> nn.Module:
    encoder = getattr(_module, 'EncoderLayerBiLSTM')(embedding_size=EncoderLSTMConfig.EMBEDDING_SIZE,
                                                     hidden_size=EncoderLSTMConfig.HIDDEN_SIZE,
                                                     vocab_size=_src_vocab_size, n_layers=EncoderLSTMConfig.N_LAYERS,
                                                     dropout=EncoderLSTMConfig.EMBEDDING_DROPOUT,
                                                     recurrent_dropout=EncoderLSTMConfig.REC_DROPOUT)
    attention = getattr(_module, 'BadhanauAttentionLayer')(hidden_size=EncoderLSTMConfig.HIDDEN_SIZE)
    decoder = getattr(_module, 'BadhanauDecoderLayerLSTM')(embedding_size=DecoderLSTMConfig.EMBEDDING_SIZE,
                                                           hidden_size=DecoderLSTMConfig.HIDDEN_SIZE,
                                                           vocab_size=_dest_vocab_size,
                                                           n_layers=DecoderLSTMConfig.N_LAYERS,
                                                           dropout=DecoderLSTMConfig.DROPOUT,
                                                           embedding_dropout=DecoderLSTMConfig.EMBEDDING_DROPOUT,
                                                           recurrent_dropout=DecoderLSTMConfig.REC_DROPOUT,
                                                           attention_layer=attention)
    return getattr(_module, 'SeqToSeqBadhanauAttentionLSTM')(encoder=encoder, decoder=decoder, device=_device,
                                                             pad_index=_pad_index)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--model', action='store', type=str, required=True,
                        help='The model name (SeqToSeqLSTM, SeqToSeqBiLSTM, SeqToSeqLuongAttentionLSTM, '
                             'SeqToSeqBadhanauAttentionLSTM).')
    parser.add_argument('--src_lang', action='store', type=str, default=DatasetConfig.SRC_LANG,
                        help=f'The source language. Default: {DatasetConfig.SRC_LANG}.')
    parser.add_argument('--dest_lang', action='store', type=str, default=DatasetConfig.DEST_LANG,
                        help=f'The destination language. Default: {DatasetConfig.DEST_LANG}.')
    parser.add_argument('--batch_size', action='store', type=int,
                        help=f'The batch size. Default: {TrainConfig.BATCH_SIZE}.')
    parser.add_argument('--init_lr', action='store', type=float,
                        help=f'The learning rate. Default: {TrainConfig.INIT_LR}.')
    parser.add_argument('--n_epochs', action='store', type=str,
                        help=f'The number of epochs. Default: {TrainConfig.N_EPOCHS}.')
    parser.add_argument('--grad_clip', action='store', type=str,
                        help=f'The value of gradient clipping. Default: {TrainConfig.GRAD_CLIP}.')
    parser.add_argument('--tf_ratio', action='store', type=str,
                        help=f'The teacher forcing ratio. Default: {TrainConfig.TF_RATIO}.')
    args = parser.parse_args()
    seed_everything(GlobalConfig.SEED)
    logger = Logger(name=f'Train{args.model}')
    criterion = nn.CrossEntropyLoss()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')

    module = getattr(__import__('nmt.model'), 'model')

    logger.info('Load Fields')
    src_field = load_field(filename=f'{args.src_lang}')
    dest_field = load_field(filename=f'{args.dest_lang}')

    logger.info('Init the model')
    if args.model == 'SeqToSeqLSTM':
        model = init_seq_to_seq_lstm_model(_module=module, _src_vocab_size=len(src_field.vocab),
                                           _dest_vocab_size=len(dest_field.vocab), _device=device)
    elif args.model == 'SeqToSeqBiLSTM':
        model = init_seq_to_seq_bi_lstm_model(_module=module, _src_vocab_size=len(src_field.vocab),
                                              _dest_vocab_size=len(dest_field.vocab), _device=device)
    elif args.model == 'SeqToSeqLuongAttentionLSTM':
        model = init_seq_to_seq_luong_attn_model(_module=module, _src_vocab_size=len(src_field.vocab),
                                                 _dest_vocab_size=len(dest_field.vocab), _device=device,
                                                 _pad_index=dest_field.vocab.stoi[dest_field.pad_token])
    elif args.model == 'SeqToSeqBadhanauAttentionLSTM':
        model = init_seq_to_seq_badhanau_attn_model(_module=module, _src_vocab_size=len(src_field.vocab),
                                                    _dest_vocab_size=len(dest_field.vocab), _device=device,
                                                    _pad_index=dest_field.vocab.stoi[dest_field.pad_token])
    else:
        raise NotImplementedError(f'The {args.model} has not been implemented!')
    model.to(device)
    logger.info(str(model))
    logger.info(f'Number of parameters of the model: {count_parameters(model):,}')

    logger.info('Init the optimizer')
    optimizer = optim.RMSprop(params=model.parameters(), lr=TrainConfig.INIT_LR)

    logger.info('Load datasets')
    train_dataset = load_dataset(filename='train', src_field=src_field, dest_field=dest_field, logger=logger)
    valid_dataset = load_dataset(filename='valid', src_field=src_field, dest_field=dest_field, logger=logger)
    test_dataset = load_dataset(filename='test', src_field=src_field, dest_field=dest_field, logger=logger)

    logger.info('Init trainer')
    trainer = Trainer(model=model, optimizer=optimizer, criterion=criterion, src_field=src_field, dest_field=dest_field,
                      train_data=train_dataset, valid_data=valid_dataset, test_data=test_dataset, logger=logger)

    logger.info('Build data iterators')
    trainer.build_data_iterator(batch_size=TrainConfig.BATCH_SIZE, device=device)

    logger.info('Suggest a good learning rate')
    trainer.lr_finder(model_name=args.model)

    logger.info('Start training...')
    history = trainer.train(n_epochs=TrainConfig.N_EPOCHS, grad_clip=TrainConfig.GRAD_CLIP,
                            tf_ratio=TrainConfig.TF_RATIO)
    logger.info('Training finished')

    _, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(history['loss'], label='train')
    axes[0].plot(history['val_loss'], label='valid')
    axes[0].set_title('Loss history')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)
    axes[0].legend()
    axes[1].plot(history['acc'], label='train')
    axes[1].plot(history['val_acc'], label='valid')
    axes[1].plot(np.array(history['bleu4']) * 100., label='BLEU-4')
    axes[1].set_title('Top-5 Accuracy & BLEU-4 history')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy & BLEU-4 (%)')
    axes[1].grid(True)
    axes[1].legend()
    plt.savefig(os.path.join(GlobalConfig.IMG_PATH, f'History_{args.model}.png'))
    plt.show()
