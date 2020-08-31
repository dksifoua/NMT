import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


def count_parameters(model: nn.Module):
    """
    Count the number of parameters of the model.

    Args:
        model (nn.Module): the wrapped model.

    Returns:
        int: the number of parameters of the model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_embeddings(embeddings: torch.FloatTensor) -> None:
    """
    Initialize embeddings.

    Args:
        embeddings (torch.FloatTensor): the embeddings.
    """
    bias = np.sqrt(3.0 / embeddings.shape[1])
    nn.init.uniform_(embeddings, -bias, bias)


def load_embeddings(nlp, field, dim=300) -> torch.FloatTensor:
    # TODO
    raise NotImplementedError


def xavier_init_weights(model: nn.Module):
    # TODO
    raise NotImplementedError


def adjust_lr(optimizer: optim.Optimizer, shrink_factor: float, verbose: bool):
    # TODO
    raise NotImplementedError


def clip_gradient(optimizer: optim.Optimizer, grad_clip: float):
    """

    Args:
        optimizer (optim.Optimizer)
        grad_clip (float)
    """
    # TODO
    raise NotImplementedError


def accuracy(logits: torch.FloatTensor, labels: torch.FloatTensor, top_k: int = 5):
    """
    Calculate the top-k accuracy.

    Args:
        logits (torch.FloatTensor[seq_len, batch_size, vocab_size]):
        labels (torch.FloatTensor[seq_len, batch_size]):
        top_k (int):

    Returns:
        float: the top-k accuracy.
    """
    # TODO
    #   Review this function.
    batch_size = logits.shape[1]
    _, indices = logits.topk(top_k, dim=2, largest=True, sorted=True)
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
            logits (torch.FloatTensor[seq_len, batch_size, vocab_size]):
            labels (torch.FloatTensor[seq_len, batch_size]):

        Returns:
            float: the F1 score.
        """
    # TODO Get the F1 score
    raise NotImplementedError


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
