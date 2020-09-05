import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch
from nmt.config.global_config import GlobalConfig
from nmt.config.train_config import TrainConfig
from nmt.config.dataset_config import DatasetConfig
from nmt.processing.processing import load_dataset, load_field
from nmt.train.trainer import Trainer
from nmt.train.train_utils import count_parameters
from nmt.utils.logger import Logger
from nmt.utils.utils import seed_everything
from scripts import init_seq_to_seq_lstm_model, init_seq_to_seq_bi_lstm_model, init_seq_to_seq_luong_attn_model, \
    init_seq_to_seq_badhanau_attn_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model')
    parser.add_argument('--model', action='store', type=str, required=True,
                        help='The model name (SeqToSeqLSTM, SeqToSeqBiLSTM, SeqToSeqLuongAttentionLSTM, '
                             'SeqToSeqBadhanauAttentionLSTM).')
    parser.add_argument('--src_lang', action='store', type=str, default=DatasetConfig.SRC_LANG,
                        help=f'The source language. Default: {DatasetConfig.SRC_LANG}.')
    parser.add_argument('--dest_lang', action='store', type=str, default=DatasetConfig.DEST_LANG,
                        help=f'The destination language. Default: {DatasetConfig.DEST_LANG}.')
    args = parser.parse_args()
    seed_everything(GlobalConfig.SEED)
    logger = Logger(name=f'Evaluate{args.model}')

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

    logger.info('Load datasets')
    train_dataset = load_dataset(filename='train', src_field=src_field, dest_field=dest_field, logger=logger)
    valid_dataset = load_dataset(filename='valid', src_field=src_field, dest_field=dest_field, logger=logger)
    test_dataset = load_dataset(filename='test', src_field=src_field, dest_field=dest_field, logger=logger)

    logger.info('Init trainer')
    trainer = Trainer(model=model, optimizer=None, criterion=None, src_field=src_field, dest_field=dest_field,
                      train_data=train_dataset, valid_data=valid_dataset, test_data=test_dataset, logger=logger)

    logger.info('Build data iterators')
    trainer.build_data_iterator(batch_size=TrainConfig.BATCH_SIZE, device=device)

    logger.info('Start the model evaluation...')
    attention = trainer.model.__class__.__name__.__contains__('Attention')
    for dataset_name in ['valid', 'test']:
        indexes = np.random.choice(len(getattr(trainer, f'{dataset_name}_data').examples), size=20, replace=False)
        for beam_size in [1, 5]:
            hypotheses, references, sources, bleu4, pred_logps, attention_weights = trainer.evaluate(
                dataset_name='valid', beam_size=beam_size, max_len=DatasetConfig.MAX_LEN, device=device
            )
            logger.info(f'BLEU-4: {bleu4*100:.3f}% on {dataset_name} dataset with beam_size={beam_size}')
            for index in indexes:
                logger.info(f'Source: {" ".join(sources[index])}')
                logger.info(f'Ground truth translation: {" ".join(references[index])}')
                logger.info(f'Predicted translation: {" ".join(hypotheses[index])}')
                logger.info('='*100)
                if attention:
                    fig = plt.figure(figsize=(10, 10))
                    ax = fig.add_subplot(111)
                    cax = ax.matshow(attention_weights[index])
                    fig.colorbar(cax)
                    ax.tick_params(labelsize=15)
                    ax.set_xticklabels(sources[index], rotation=45)
                    ax.set_yticklabels(hypotheses[index])
                    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
                    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
                    plt.savefig(os.path.join(GlobalConfig.IMG_PATH,
                                             f'{args.model}_{dataset_name}_index_{index}_beam_size_{beam_size}.png'))
                    plt.show()
    # TODO
    #   Error analysis
