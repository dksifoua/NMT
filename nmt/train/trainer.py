import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Dataset, Field
from torchtext.data.metrics import bleu_score
from torchtext.data.iterator import BucketIterator
from nmt.train.utils import accuracy, adjust_lr, AverageMeter, clip_gradient
from nmt.utils.logger import Logger


class Trainer:
    """
    Training routines.

    Args:
        model (nn.Module): the wrapped model.
        optimizer (optim.Optimizer): the wrapped optimizer.
        criterion (nn.Module): the wrapped loss function.
        field (Field): Language manager.
        train_data (Dataset): train dataset.
        valid_data (Dataset): valid dataset.
        test_data (Dataset): test dataset.
    """

    def __init__(self, model: nn.Module, optimizer: optim.Optimizer, criterion: nn.Module,
                 field: Field, train_data: Dataset, valid_data: Dataset, test_data: Dataset):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.field = field
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.train_iterator = None
        self.valid_iterator = None
        self.test_iterator = None
        self.logger = Logger(name=f'{model.__class__.__name__}Trainer')

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
        return loss_tracker.average, acc_tracker.average

    def validate(self, epoch: int):
        """

        Args:
            epoch (int): the epoch number.

        Returns:
            loss (float): the validation loss.
            acc (float): the validation top-5 accuracy.
            bleu-4 (float): the validation BLEU score.
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
                    reference = [self.field.vocab.itos[indice] for indice in target_sequence if indice not in (
                        self.field.vocab.stoi[self.field.init_token],
                        self.field.vocab.stoi[self.field.pad_token]
                    )]
                    references.append([reference])
                # Update hypotheses
                _, predictions = torch.max(logits_copy, dim=2)
                predictions = predictions.t().tolist()
                for j, p in enumerate(predictions):
                    hypotheses.append([self.field.vocab.itos[indice]
                                       for indice in predictions[j][:sorted_decode_lengths[j]]  # Remove padding
                                       if indice not in (
                                           self.field.vocab.stoi[self.field.init_token],
                                           self.field.vocab.stoi[self.field.pad_token]
                                       )])
                assert len(references) == len(hypotheses)
                # Update progressbar description
                progress_bar.set_description(
                    f'Epoch: {epoch + 1:03d} - val_loss: {loss_tracker.average:.3f}'
                    f' - val_acc: {acc_tracker.average:.3f}%')
            # Calculate BLEU-4 score
            bleu4 = bleu_score(hypotheses, references, max_n=4, weights=[0.25, 0.25, 0.25, 0.25])
            # TODO
            #   Display some examples
        return loss_tracker.average, acc_tracker.average, bleu4

    def train(self, n_epochs: int, grad_clip: float, tf_ratio: float):
        # TODO
        #   Load the saved model if exits to continue the training.
        last_improvement = 0
        history, best_bleu = {'acc': [], 'loss': [], 'val_acc': [], 'val_loss': [], 'bleu4': []}, 0.
        for epoch in range(n_epochs):
            # Stop training if no improvement since last 4 epochs
            if last_improvement == 4:
                self.logger.info('Training Finished - The model has stopped improving since last 4 epochs')
                break
            # Decay LR if no improvement
            if last_improvement > 0:
                adjust_lr(optimizer=self.optimizer, shrink_factor=0.9, verbose=True)
            # Train step
            loss, acc = self.train_step(epoch=epoch, grad_clip=grad_clip, tf_ratio=tf_ratio)
            # Validation step
            val_loss, val_acc, bleu4 = self.validate(epoch=epoch)
            # Update history dict
            history['acc'].append(acc)
            history['loss'].append(loss)
            history['val_acc'].append(val_acc)
            history['val_loss'].append(val_loss)
            history['bleu4'].append(bleu4)
            # TODO
            #   Print BLEU score
            #   Adjust the teacher forcing rate
            #   Save model
        return history
