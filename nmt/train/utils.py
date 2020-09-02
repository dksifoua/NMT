import os
import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from nmt.config.global_config import GlobalConfig
from nmt.utils.logger import Logger


def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_embeddings(embeddings: torch.FloatTensor) -> None:
    bias = np.sqrt(3.0 / embeddings.shape[1])
    nn.init.uniform_(embeddings, -bias, bias)


def load_embeddings(nlp, field, dim=300) -> torch.FloatTensor:
    embeddings = torch.FloatTensor(len(field.vocab), dim)
    init_embeddings(embeddings)
    for token, index in tqdm.tqdm(field.vocab.stoi.items()):
        token = nlp(token)
        if token.has_vector:
            embeddings[index] = torch.tensor(token.vector, dtype=torch.float32)
    return embeddings


def adjust_lr(optimizer: optim.Optimizer, shrink_factor: float, verbose: bool = False, logger: Logger = None):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    if verbose and logger is not None:
        logger.debug("\nDecaying learning rate.")
        logger.debug("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def adjust_tf(tf_ratio: float, shrink_factor: float, verbose: bool = False, logger: Logger = None):
    tf_ratio = tf_ratio * shrink_factor
    if verbose and logger is not None:
        logger.debug("The teacher forcing rate is %f\n" % (tf_ratio,))
    return tf_ratio


def clip_gradient(optimizer: optim.Optimizer, grad_clip: float):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def accuracy(logits: torch.FloatTensor, labels: torch.IntTensor, top_k: int = 5):
    """
    Compute the top-k accuracy.

    Args:
        logits: torch.FloatTensor[seq_len, batch_size, vocab_size]
        labels: torch.IntTensor[seq_len, batch_size]
        top_k: int

    Returns:
        float: the top-k accuracy
    """
    batch_size = logits.shape[1]
    _, indices = logits.topk(top_k, dim=2, largest=True, sorted=True)  # [seq_len, batch_size, top_k]
    correct = indices.eq(labels.view(-1, 1).expand_as(indices))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)


def f1_score(logits: torch.FloatTensor, labels: torch.FloatTensor):
    """
    Calculate the F1 score.
    precision = correct / length_output
    recall = correct / length_reference
    f1 = 2 * (precision * recall) / (precision + recall)

    Args:
        logits: torch.FloatTensor[seq_len, batch_size, vocab_size]
        labels: torch.FloatTensor[seq_len, batch_size]

    Returns:
        float: the F1 score.
    """
    raise NotImplementedError


def save(model: nn.Module, optimizer: nn.Module, last_improvement: int, bleu4: float, is_best: bool):
    state = {
        'bleu-4': bleu4,
        'last_improvement': last_improvement,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, os.path.join(GlobalConfig.CHECKPOINT_PATH,  f'{model.__class__.__name__}.pth'))
    if is_best:
        torch.save(state, os.path.join(GlobalConfig.CHECKPOINT_PATH, f'Best_{model.__class__.__name__}.pth'))


def load(model_name: str):
    state = torch.load(os.path.join(GlobalConfig.CHECKPOINT_PATH, f'Best_{model_name}.pt'))
    return state.get('model'), state.get('optimizer'), state.get('last_improvement')


class AverageMeter:

    def __init__(self):
        self.value = 0.
        self.sum = 0.
        self.count = 0
        self.average = 0.

    def reset(self):
        self.value = 0.
        self.sum = 0.
        self.count = 0
        self.average = 0.

    def update(self, value, n=1):
        self.value = value
        self.sum += value * n
        self.count += n
        self.average = self.sum / self.count
