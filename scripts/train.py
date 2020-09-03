import os
import argparse
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from nmt.config.dataset_config import DatasetConfig
from nmt.config.global_config import GlobalConfig
from nmt.config.train_config import TrainConfig
from nmt.config.model_config import EncoderLSTMConfig, DecoderLSTMConfig
from nmt.processing.processing import load_dataset, load_field
from nmt.train.trainer import Trainer
from nmt.train.optim_utils import LRFinder
from nmt.utils.logger import Logger
from typing import Any


def init_seq_to_seq_lstm_model(_module: Any, _src_vocab_size: int,
                               _dest_vocab_size: int, _device: torch.device) -> nn.Module:
    encoder = getattr(_module, 'EncoderLayerLSTM')(embedding_size=EncoderLSTMConfig.EMBEDDING_SIZE,
                                                   hidden_size=EncoderLSTMConfig.HIDDEN_SIZE,
                                                   vocab_size=_src_vocab_size,
                                                   n_layers=EncoderLSTMConfig.N_LAYERS,
                                                   dropout=EncoderLSTMConfig.EMBEDDING_DROPOUT,
                                                   recurrent_dropout=EncoderLSTMConfig.REC_DROPOUT)
    decoder = getattr(_module, 'DecoderLayerLSTM')(embedding_size=DecoderLSTMConfig.EMBEDDING_SIZE,
                                                   hidden_size=DecoderLSTMConfig.HIDDEN_SIZE,
                                                   vocab_size=_dest_vocab_size,
                                                   n_layers=DecoderLSTMConfig.N_LAYERS,
                                                   embedding_dropout=DecoderLSTMConfig.EMBEDDING_DROPOUT,
                                                   recurrent_dropout=DecoderLSTMConfig.REC_DROPOUT)
    return getattr(_module, 'SeqToSeqLSTM')(encoder=encoder, decoder=decoder, device=_device)


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
    else:
        raise ValueError(f'The {args.model} has not been implemented!')
    # TODO
    #   Add other model initialization
    #   Add word vector embeddings
    #   Add xavier init weights
    model.to(device)
    logger.debug(str(model))

    logger.info('Init the optimizer')
    optimizer = optim.RMSprop(params=model.parameters(), lr=TrainConfig.INIT_LR)

    logger.info('Load datasets')
    train_dataset = load_dataset(filename='train', src_field=src_field, dest_field=dest_field, logger=logger)
    valid_dataset = load_dataset(filename='valid', src_field=src_field, dest_field=dest_field, logger=logger)
    test_dataset = load_dataset(filename='test', src_field=src_field, dest_field=dest_field, logger=logger)

    logger.info('Suggest a good learning rate')
    lr_finder = LRFinder(model=model, optimizer=optimizer, criterion=criterion, logger=logger,
                         grad_clip=TrainConfig.GRAD_CLIP)
    lr_finder.range_test(data_loader=train_dataset, end_lr=TrainConfig.END_LR, n_iters=TrainConfig.N_ITERS)
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(1, 1, 1)
    ax, lr = lr_finder.plot(ax=ax)
    plt.savefig(os.path.join(GlobalConfig.IMG_PATH, f'SuggestedLR_{args.model}.png'))

    logger.info('Init trainer')
    trainer = Trainer(model=model, optimizer=optimizer, criterion=criterion, dest_field=dest_field,
                      train_data=train_dataset, valid_data=valid_dataset, test_data=test_dataset, logger=logger)

    logger.info('Build data iterators')
    trainer.build_data_iterator(batch_size=TrainConfig.BATCH_SIZE, device=device)

    logger.info('Start training...')
    history = trainer.train(n_epochs=1, grad_clip=TrainConfig.GRAD_CLIP, tf_ratio=TrainConfig.TF_RATIO)
