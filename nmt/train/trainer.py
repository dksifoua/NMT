import os
import tqdm
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Dataset, Field
from torchtext.data.metrics import bleu_score
from torchtext.data.iterator import BucketIterator
from nmt.train.train_utils import accuracy, adjust_lr, adjust_tf, AverageMeter, clip_gradient, load, save
from nmt.config.global_config import GlobalConfig
from nmt.utils.logger import Logger


class Trainer:
    """
    Training routines.

    Args:
        model (nn.Module): the wrapped model.
        optimizer (optim.Optimizer): the wrapped optimizer.
        criterion (nn.Module): the wrapped loss function.
        train_data (Dataset): train dataset.
        valid_data (Dataset): valid dataset.
        test_data (Dataset): test dataset.
    """

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, criterion: nn.Module, dest_field: Field,
                 train_data: Dataset, valid_data: Dataset, test_data: Dataset, logger: Logger):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.dest_field = dest_field
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.logger = logger
        self.train_iterator = None
        self.valid_iterator = None
        self.test_iterator = None

    def build_data_iterator(self, batch_size: int, device: torch.device):
        """
        Build data iterators for the training.

        Args:
            batch_size (int): the batch size.
            device (torch.device): the device on which the training will process.
        """
        self.train_iterator, self.valid_iterator, self.test_iterator = BucketIterator.splits(
            (self.train_data, self.valid_data, self.test_data),
            batch_size=batch_size,
            sort_key=lambda x: len(x.src),
            sort_within_batch=True,
            device=device
        )

    def train_step(self, epoch: int, grad_clip: float, tf_ratio: float):
        """
        Train the model on a batch.

        Args:
            epoch (int): the epoch number.
            grad_clip (float): the value beyond which we clip gradients in order avoid exploding gradients.
            tf_ratio (float): the teacher forcing ratio. Must be in [0, 1.0]

        Returns:
            loss (float): the validation loss.
            acc (float): the validation top-5 accuracy.
        """
        loss_tracker, acc_tracker = AverageMeter(), AverageMeter()
        self.model.train()
        progress_bar = tqdm.tqdm(enumerate(self.train_iterator), total=len(self.train_iterator))
        for i, data in progress_bar:
            # Forward prop.
            logits, sorted_dest_sequences, sorted_decode_lengths, sorted_indices = self.model(*data.src, *data.dest,
                                                                                              tf_ratio=tf_ratio)
            # Since we decoded starting with <sos>, the targets are all words after <sos>, up to <eos>
            sorted_dest_sequences = sorted_dest_sequences[1:, :]
            # Remove paddings
            logits = nn.utils.rnn.pack_padded_sequence(logits, sorted_decode_lengths).data
            sorted_dest_sequences = nn.utils.rnn.pack_padded_sequence(sorted_dest_sequences, sorted_decode_lengths).data
            # Calculate loss
            loss = self.criterion(logits, sorted_dest_sequences)
            # Back prop.
            self.optimizer.zero_grad()
            loss.backward()
            # Clip gradients
            if grad_clip is not None:
                clip_gradient(self.optimizer, grad_clip)
            # Update weights
            self.optimizer.step()
            # Track metrics
            loss_tracker.update(loss.item(), sum(sorted_decode_lengths))
            acc_tracker.update(accuracy(logits, sorted_dest_sequences, top_k=5), sum(sorted_decode_lengths))
            # Update progressbar description
            progress_bar.set_description(
                f'Epoch: {epoch + 1:03d} - loss: {loss_tracker.average:.3f} - acc: {acc_tracker.average:.3f}%')
            self.logger.debug(
                f'Epoch: {epoch + 1:03d} - loss: {loss_tracker.average:.3f} - acc: {acc_tracker.average:.3f}%')
        return loss_tracker.average, acc_tracker.average

    def validate(self, epoch: int):
        """

        Args:
            epoch: int
                The epoch number.

        Returns:
            loss: float
                The validation loss.
            acc: float
                The validation top-5 accuracy.
            bleu-4: float
                The validation BLEU score.
        """
        references, hypotheses = [], []
        loss_tracker, acc_tracker = AverageMeter(), AverageMeter()
        self.model.eval()
        with torch.no_grad():
            progress_bar = tqdm.tqdm(enumerate(self.valid_iterator), total=len(self.valid_iterator))
            for i, data in progress_bar:
                # Forward prop.
                logits, sorted_dest_sequences, sorted_decode_lengths, sorted_indices = self.model(*data.src, *data.dest,
                                                                                                  tf_ratio=0.)
                # Since we decoded starting with <sos>, the targets are all words after <sos>, up to <eos>
                sorted_dest_sequences = sorted_dest_sequences[1:, :]
                # Remove paddings
                logits_copy = logits.clone()
                logits = nn.utils.rnn.pack_padded_sequence(logits, sorted_decode_lengths).data
                sorted_dest_sequences = nn.utils.rnn.pack_padded_sequence(sorted_dest_sequences,
                                                                          sorted_decode_lengths).data
                # Calculate loss
                loss = self.criterion(logits, sorted_dest_sequences)
                # Track metrics
                loss_tracker.update(loss.item(), sum(sorted_decode_lengths))
                acc_tracker.update(accuracy(logits, sorted_dest_sequences, top_k=5), sum(sorted_decode_lengths))
                # Update references
                target_sequences = data.dest[0].t()[sorted_indices]
                for j in range(target_sequences.size(0)):
                    target_sequence = target_sequences[j].tolist()
                    reference = [self.dest_field.vocab.itos[indice] for indice in target_sequence if indice not in (
                        self.dest_field.vocab.stoi[self.dest_field.init_token],
                        self.dest_field.vocab.stoi[self.dest_field.pad_token]
                    )]
                    references.append([reference])
                # Update hypotheses
                _, predictions = torch.max(logits_copy, dim=2)
                predictions = predictions.t().tolist()
                for j, p in enumerate(predictions):
                    hypotheses.append([self.dest_field.vocab.itos[indice]
                                       for indice in predictions[j][:sorted_decode_lengths[j]]  # Remove padding
                                       if indice not in (
                                           self.dest_field.vocab.stoi[self.dest_field.init_token],
                                           self.dest_field.vocab.stoi[self.dest_field.pad_token]
                                       )])
                assert len(references) == len(hypotheses)
                # Update progressbar description
                progress_bar.set_description(
                    f'Epoch: {epoch + 1:03d} - val_loss: {loss_tracker.average:.3f}'
                    f' - val_acc: {acc_tracker.average:.3f}%')
                self.logger.debug(f'Epoch: {epoch + 1:03d} - val_loss: {loss_tracker.average:.3f}'
                                  f' - val_acc: {acc_tracker.average:.3f}%')
            # Calculate BLEU-4 score
            bleu4 = bleu_score(hypotheses, references, max_n=4, weights=[0.25, 0.25, 0.25, 0.25])
            # Display some examples
            for i in np.random.choice(len(self.valid_iterator), size=3, replace=False):
                src, dest = ' '.join(references[i][0]), ' '.join(hypotheses[i])
                self.logger.debug(f'Ground truth translation: {src}')
                self.logger.info(f'Predicted translation: {dest}')
                self.logger.info('=' * 100)
        return loss_tracker.average, acc_tracker.average, bleu4

    def train(self, n_epochs: int, grad_clip: float, tf_ratio: float):
        if f'Best_{self.model.__class__.__name__}.pth' in os.listdir(GlobalConfig.CHECKPOINT_PATH):
            model_state_dict, optim_state_dict, last_improvement = load(self.model.__class__.__name__)
            self.model.load_state_dict(model_state_dict)
            self.optimizer.load_state_dict(optim_state_dict)
            self.logger.info('The model is loaded!')
        else:
            last_improvement = 0
        history, best_bleu = {'acc': [], 'loss': [], 'val_acc': [], 'val_loss': [], 'bleu4': []}, 0.
        for epoch in range(n_epochs):
            if last_improvement == 4:  # Stop training if no improvement since last 4 epochs
                self.logger.info('Training Finished - The model has stopped improving since last 4 epochs')
                break
            if last_improvement > 0:  # Decay LR if no improvement
                adjust_lr(optimizer=self.optimizer, shrink_factor=0.9, verbose=True, logger=self.logger)
            loss, acc = self.train_step(epoch=epoch, grad_clip=grad_clip, tf_ratio=tf_ratio)  # Train step
            val_loss, val_acc, bleu4 = self.validate(epoch=epoch)  # Validation step
            # Update history dict
            history['acc'].append(acc)
            history['loss'].append(loss)
            history['val_acc'].append(val_acc)
            history['val_loss'].append(val_loss)
            history['bleu4'].append(bleu4)
            # Print BLEU score
            text = f'BLEU-4: {bleu4 * 100:.3f}%'
            if bleu4 > best_bleu:
                best_bleu, last_improvement = bleu4, 0
            else:
                last_improvement += 1
                text += f' - Last improvement since {last_improvement} epoch(s)'
            self.logger.info(text)
            # Decrease teacher forcing rate
            tf_ratio = adjust_tf(tf_ratio=tf_ratio, shrink_factor=0.8, verbose=False)
            # Checkpoint
            save(model=self.model, optimizer=self.optimizer, last_improvement=last_improvement, bleu4=bleu4)
        return history
