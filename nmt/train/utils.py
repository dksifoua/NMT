import torch.optim as optim


def adjust_lr(optimizer: optim.Optimizer, shrink_factor, verbose: bool):
    # TODO
    pass


def accuracy(y_hat, y_true, top_k: int = 5):
    # TODO
    return 0.


def clip_gradient(optimizer: optim.Optimizer, grad_clip: float):
    """

    Args:
        optimizer (optim.Optimizer)
        grad_clip (float)
    """
    # TODO
    pass


class AverageMeter:

    def __init__(self):
        # TODO
        self.value = 0.
        self.average = 0.

    def update(self, value, n):
        # TODO
        return self.value
