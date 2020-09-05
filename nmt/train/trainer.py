import os
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext.data import Dataset, Field
from torchtext.data.metrics import bleu_score
from torchtext.data.iterator import BucketIterator
from nmt.config.global_config import GlobalConfig
from nmt.config.train_config import TrainConfig
from nmt.train.train_utils import accuracy, adjust_lr, adjust_tf, AverageMeter, clip_gradient, load, save
from nmt.train.optim_utils import LRFinder
from nmt.train.beam_utils import find_best_path, Node
from nmt.utils.logger import Logger
from typing import Optional


class Trainer:
    """
    Training routines.

    Args:
        model: nn.Module
            The wrapped model.
        optimizer: Optional[optim.Optimizer]
            The wrapped optimizer. Can be None for evaluation and inference phases.
        criterion: Optional[nn.Module]
            The wrapped loss function. Can be None for evaluation and inference phases.
        train_data: Dataset
            Train dataset.
        valid_data: Dataset
            Valid dataset.
        test_data: Dataset
            Test dataset.
    """

    def __init__(self, model: nn.Module, optimizer: Optional[optim.Optimizer], criterion: Optional[nn.Module], src_field: Field,
                 dest_field: Field, train_data: Dataset, valid_data: Dataset, test_data: Dataset, logger: Logger):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.src_field = src_field
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

    def lr_finder(self, model_name: str):
        """
        Find the best learning rate for training process.

        Args:
            model_name:
                The class name of the model.
        """
        lr_finder = LRFinder(model=self.model, optimizer=self.optimizer, criterion=self.criterion, logger=self.logger,
                             grad_clip=TrainConfig.GRAD_CLIP)
        lr_finder.range_test(data_loader=self.train_iterator, end_lr=TrainConfig.END_LR, n_iters=TrainConfig.N_ITERS)
        fig = plt.figure(figsize=(15, 5))
        ax = fig.add_subplot(1, 1, 1)
        ax, lr = lr_finder.plot(ax=ax)
        plt.savefig(os.path.join(GlobalConfig.IMG_PATH, f'SuggestedLR_{model_name}.png'))
        plt.show()
        if lr is not None:  # Create an optimizer with the suggested LR
            self.optimizer = optim.RMSprop(params=self.model.parameters(), lr=lr)

    def load_model_optimizer_weights(self):
        last_improvement = 0
        if f'Best_{self.model.__class__.__name__}.pth' in os.listdir(GlobalConfig.CHECKPOINT_PATH):
            model_state_dict, optim_state_dict, last_improvement = load(self.model.__class__.__name__)
            self.model.load_state_dict(model_state_dict)
            if self.optimizer is not None:
                self.optimizer.load_state_dict(optim_state_dict)
            self.logger.info('The model is loaded!')
        return last_improvement

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
        Validate the model on a batch.

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
        """
        Train the model.

        Args:
            n_epochs:  int
            grad_clip: float
            tf_ratio: float

        Returns:
            history: Dict[str, List[float]]
        """
        last_improvement = self.load_model_optimizer_weights()
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
            save(model=self.model, optimizer=self.optimizer, last_improvement=last_improvement, bleu4=bleu4,
                 is_best=bleu4 >= best_bleu)
        return history

    def evaluate(self, dataset_name: str, beam_size: int, max_len: int, device: torch.device):
        """
        Evaluate the model on the test data

        Args:
            beam_size: int
            dataset_name: str
                The dataset on which we evaluate the model. Can be valid or test.
            max_len: int
            device: torch.device

        Returns:
            hypotheses: List[str]
            references: List[str]
            sources: List[str]
            bleu4: float
            pred_logps: List[float]
            attention_weights: List[np.array]
        """
        if dataset_name not in ['valid', 'test']:
            raise ValueError
        _ = self.load_model_optimizer_weights()
        # TODO
        #   Use dataset instead of iterator
        attention = self.model.__class__.__name__.__contains__('Attention')
        references, hypotheses, sources, pred_logps, attention_weights = [], [], [], [], []
        self.model.eval()
        with torch.no_grad():
            iterator = getattr(self, f'{dataset_name}_iterator')
            progress_bar = tqdm.tqdm(enumerate(iterator), total=len(iterator))
            for i, data in progress_bar:
                src_sequences, src_lengths = data.src[0], data.src[1]
                dest_sequences, dest_lengths = data.dest[0], data.dest[1]
                batch_size = src_sequences.shape[1]
                for j in range(batch_size):  # We evaluate sentence by sentence
                    src_sequence = src_sequences[:, j].unsqueeze(1)  # [seq_len, 1]
                    dest_sequence = dest_sequences[:, j].unsqueeze(1)  # [seq_len, 1]
                    src_length, dest_length = src_lengths[j, None], dest_lengths[j, None]  # [1,]

                    # Encoding
                    enc_outputs, (h_state, c_state) = self.model.encoder(input_sequences=src_sequence,
                                                                         sequence_lengths=src_length)
                    # Decoding
                    if attention:
                        mask = self.model.create_mask(src_sequence)  # [seq_len, 1]
                    tree = [[Node(
                        token=torch.LongTensor([self.dest_field.vocab.stoi[self.dest_field.init_token]]).to(device),
                        states=(h_state, c_state, None)
                    )]]
                    for _ in range(max_len):
                        next_nodes = []
                        for node in tree[-1]:
                            if node.eos:  # Skip eos token
                                continue
                            # Decode
                            if attention:
                                logit, (h_state, c_state, attention_weights) = self.model.decoder(
                                    input_word_index=node.token,
                                    h_state=node.states[0].contiguous(),
                                    c_state=node.states[1].contiguous(),
                                    enc_outputs=enc_outputs,
                                    mask=mask
                                )
                            else:
                                logit, (h_state, c_state) = self.model.decoder(input_word_index=node.token,
                                                                               h_state=node.states[0].contiguous(),
                                                                               c_state=node.states[1].contiguous())
                            # logit: [1, vocab_size]
                            # h_state: [n_layers, 1, hidden_size]
                            # c_state: [n_layers, 1, hidden_size]

                            # Get scores
                            logp = F.log_softmax(logit, dim=1).squeeze(dim=0)  # [vocab_size]

                            # Get top k tokens & logps
                            topk_logps, topk_tokens = torch.topk(logp, beam_size)

                            for k in range(beam_size):
                                next_nodes.append(Node(
                                    token=topk_tokens[k, None], states=(
                                        h_state, c_state, attention_weights if attention else None),
                                    logp=topk_logps[k, None].cpu().item(), parent=node,
                                    eos=topk_tokens[k].cpu().item() == self.dest_field.vocab[self.dest_field.eos_token]
                                ))

                        if len(next_nodes) == 0:
                            break

                        # Sort next_nodes to get the best
                        next_nodes = sorted(next_nodes, key=lambda _node: _node.logps, reverse=True)

                        # Update the tree
                        tree.append(next_nodes[:beam_size])

                    # Find the best path of the tree
                    best_path = find_best_path(tree)

                    # Get the translation
                    pred_translated = [*map(lambda _node: self.dest_field.vocab.itos[_node.token], best_path)]
                    pred_translated = [*filter(lambda word: word not in [
                        self.dest_field.init_token, self.dest_field.eos_token
                    ], pred_translated[::-1])]

                    # Update hypotheses
                    hypotheses.append(pred_translated)

                    # Update pred logps
                    pred_logps.append(sum([*map(lambda _node: _node.logps, best_path)]))

                    # Update attention weights
                    if attention:
                        attention_weights.append(
                            torch.cat([*map(lambda _node: _node.states[-1], best_path)], dim=1).cpu().detach().numpy()
                        )

                    # Update references
                    references.append([[
                        self.dest_field.vocab.itos[indice]
                        for indice in dest_sequence
                        if indice not in (
                            self.dest_field.vocab.stoi[self.dest_field.init_token],
                            self.dest_field.vocab.stoi[self.dest_field.eos_token],
                            self.dest_field.vocab.stoi[self.dest_field.pad_token]
                        )
                    ]])

                    # Update sources
                    sources.append([
                        self.src_field.vocab.itos[indice]
                        for indice in src_sequence
                        if indice not in (
                            self.src_field.vocab.stoi[self.src_field.init_token],
                            self.src_field.vocab.stoi[self.src_field.eos_token],
                            self.src_field.vocab.stoi[self.src_field.pad_token]
                        )
                    ])

            # Calculate BLEU-4 score
            bleu4 = bleu_score(hypotheses, references, max_n=4, weights=[0.25, 0.25, 0.25, 0.25])

        return hypotheses, references, sources, bleu4, pred_logps, attention_weights

    def translate(self):
        # TODO
        raise NotImplementedError
