import torch
import torch.nn as nn
import torch.nn.functional as F


class LuongAttentionLayer(nn.Module):
    """
    Luong Attention Layer

    Args:
        hidden_size (int)
    """

    def __init__(self, hidden_size: int):
        super(LuongAttentionLayer, self).__init__()
        self.hidden_size = hidden_size

    def forward(self, h_state: torch.FloatTensor, enc_outputs: torch.FloatTensor, mask: torch.BoolTensor = None):
        """
        Args:
            h_state (torch.FloatTensor[n_layers, batch_size, hidden_size]): current hidden state decoder.
            enc_outputs (torch.FloatTensor[seq_len, batch_size, hidden_size]): encoder outputs.
            mask (torch.BootTensor[seq_len, batch_size, 1], optional): mask to ignore pad tokens. Default: None.

        Returns:
            attn_weights (torch.FloatTensor[seq_len, batch_size, 1])
        """
        if not (h_state.shape[-1] == enc_outputs.shape[-1] == self.hidden_size):
            raise ValueError('Hidden size does not match.')

        if h_state.shape[0] > 1:
            h_state = torch.sum(h_state, dim=0)  # [batch_size, hidden_size]
            h_state = h_state.unsqueeze(dim=0)  # [1, batch_size, hidden_size]

        # Calculating alignment scores
        scores = torch.sum(enc_outputs * h_state, dim=2)  # [seq_len, batch_size]
        scores = scores.unsqueeze(dim=-1)  # [seq_len, batch_size, 1]

        # Apply mask to ignore <pad> tokens
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e18)

        # Calculating the attention weights by applying softmax to the alignments scores
        attention_weights = F.softmax(scores, dim=0)  # [seq_len, batch_size, 1]

        return attention_weights


class BadhanauAttentionLayer(nn.Module):
    """
        Badhanau Attention Layer

        Args:
            hidden_size (int)
        """

    def __init__(self, hidden_size: int):
        super(BadhanauAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.W1 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.W2 = nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.V = nn.Linear(in_features=hidden_size, out_features=1)

    def forward(self, h_state: torch.FloatTensor, enc_outputs: torch.FloatTensor, mask: torch.BoolTensor = None):
        """

        Args:
            h_state (torch.FloatTensor[n_layers, batch_size, hidden_size])
            enc_outputs (torch.FloatTensor[seq_len, batch_size, hidden_size])
            mask (torch.BoolTensor[seq_len, batch_size, 1])

        Returns:
            attn_weights (torch.Tensor[seq_len, batch_size, 1], optional)
        """
        if not (h_state.shape[-1] == enc_outputs.shape[-1] == self.hidden_size):
            raise ValueError('Hidden size does not match.')

        if h_state.shape[0] > 1:
            h_state = torch.sum(h_state, dim=0)  # [batch_size, hidden_size]
            h_state = h_state.unsqueeze(0)  # [1, batch_size, hidden_size]

        # Calculating alignment scores
        scores = self.V(
            torch.tanh(
                self.W1(enc_outputs) + self.W2(h_state)  # [seq_len, batch_size, hidden_size]
            )
        )  # [seq_len, batch_size, 1]

        # Apply mask to ignore <pad> tokens
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e18)

        # Calculating the attention weights by applying softmax to the alignments scores
        attention_weights = F.softmax(scores, dim=0)  # [seq_len, batch_size, 1]

        return attention_weights
