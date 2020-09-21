import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class SeqToSeqLSTM(nn.Module):
    """
    LSTM sequence-to-sequence model.

    Args:
        encoder: nn.Module
            The wrapped encoder model
        decoder: nn.Module
            The wrapped decoder model
        device: torch.device
            The device on which the model is going.
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module, device: torch.device):
        if encoder.hidden_size != decoder.hidden_size or encoder.n_layers != decoder.n_layers:
            raise ValueError('Encoder and Decoder must have the same number of recurrent hidden units and layers.')
        super(SeqToSeqLSTM, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src_sequences: torch.IntTensor, src_lengths: torch.IntTensor, dest_sequences: torch.IntTensor,
                dest_lengths: torch.IntTensor, tf_ratio: float):
        """
        Pass inputs through the model.

        Args:
            src_sequences: torch.IntTensor[seq_len, batch_size]
            src_lengths: torch.IntTensor[batch_size,]
            dest_sequences: torch.IntTensor[seq_len, batch_size]
            dest_lengths: torch.IntTensor[batch_size,]
            tf_ratio: float

        Returns:
            logits: torch.FloatTensor[max(decode_lengths), batch_size, vocab_size]
            sorted_dest_sequences: torch.IntTensor[seq_len, batch_size]
            sorted_decode_lengths: list
            sorted_indices: torch.IntTensor[batch_size,]
        """
        # Encoding
        _, (h_state, c_state) = self.encoder(input_sequences=src_sequences, sequence_lengths=src_lengths)
        # h_state: [n_layers, batch_size, hidden_size]
        # c_state: [n_layers, batch_size, hidden_size]

        # Sort the batch (dest) by decreasing lengths
        sorted_dest_lengths, sorted_indices = torch.sort(dest_lengths, dim=0, descending=True)
        sorted_dest_sequences = dest_sequences[:, sorted_indices]
        h_state = h_state[:, sorted_indices, :]
        c_state = c_state[:, sorted_indices, :]

        # We won't decode at the <eos> position, since we've finished generating as soon as we generate <eos>
        # So, decoding lengths are actual lengths - 1
        sorted_decode_lengths = (sorted_dest_lengths - 1).tolist()

        # Decoding
        batch_size, last = dest_sequences.shape[1], None
        logits = torch.zeros(max(sorted_decode_lengths), batch_size, self.decoder.vocab_size).to(self.device)
        for t in range(max(sorted_decode_lengths)):
            batch_size_t = sum([length > t for length in sorted_decode_lengths])
            if last is not None:
                if np.random.rand() < tf_ratio:
                    in_ = last[:batch_size_t]
                else:
                    in_ = sorted_dest_sequences[t, :batch_size_t]
            else:
                in_ = sorted_dest_sequences[t, :batch_size_t]
            # in_ [batch_size,]
            logit, (h_state, c_state) = self.decoder(
                in_,
                h_state[:, :batch_size_t, :].contiguous(),
                c_state[:, :batch_size_t, :].contiguous()
            )
            # logit: [batch_size, vocab_size]
            # h_state: [num_layers, batch_size, hidden_size]
            # c_state: [num_layers, batch_size, hidden_size]
            logits[t, :batch_size_t, :] = logit
            last = torch.argmax(F.softmax(logit, dim=1), dim=1)  # [batch_size,]

        return logits, sorted_dest_sequences, sorted_decode_lengths, sorted_indices


class SeqToSeqBiLSTM(SeqToSeqLSTM):
    """
    LSTM sequence-to-sequence model.

    Args:
        encoder: nn.Module
            The wrapped encoder model
        decoder: nn.Module
            The wrapped decoder model
        device: torch.device
            The device on which the model is going.
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module, device: torch.device):
        if encoder.hidden_size != decoder.hidden_size or encoder.n_layers != decoder.n_layers:
            raise ValueError('Encoder and Decoder must have the same number of recurrent hidden units and layers.')
        super(SeqToSeqBiLSTM, self).__init__(encoder=encoder, decoder=decoder, device=device)
