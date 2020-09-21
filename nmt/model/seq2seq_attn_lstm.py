import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class SeqToSeqLuongAttentionLSTM(nn.Module):
    """
    BiLSTM sequence-to-sequence model with Luong attention.

    Args:
        encoder: nn.Module
            The wrapped encoder model
        decoder: nn.Module
            The wrapped decoder model
        device: torch.device
            The device on which the model is going.
        pad_index: int
    """

    def __init__(self, encoder: nn.Module, decoder: nn.Module, device: torch.device, pad_index: int):
        if encoder.hidden_size != decoder.hidden_size or encoder.n_layers != decoder.n_layers:
            raise ValueError('Encoder and Decoder must have the same number of recurrent hidden units and layers.')
        super(SeqToSeqLuongAttentionLSTM, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.pad_index = pad_index

    def create_mask(self, src_sequences: torch.IntTensor):
        """
        Create mask in order to ignore pad tokens.

        Args:
            src_sequences: torch.IntTensor[seq_len, batch_size]

        Returns:
            mask: torch.BoolTensor[seq_len, batch_size, 1]
        """
        mask = src_sequences != self.pad_index
        return mask.unsqueeze(2)

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
        mask = self.create_mask(src_sequences)

        # Encoding
        enc_outputs, (h_state, c_state) = self.encoder(input_sequences=src_sequences, sequence_lengths=src_lengths)
        # enc_outputs: [seq_len, batch_size, hidden_size]
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
            logit, (h_state, c_state, _) = self.decoder(input_word_index=in_,
                                                        h_state=h_state[:, :batch_size_t, :].contiguous(),
                                                        c_state=c_state[:, :batch_size_t, :].contiguous(),
                                                        enc_outputs=enc_outputs[:, :batch_size_t, :],
                                                        mask=mask[:, :batch_size_t, :])
            # logit: [batch_size, vocab_size]
            # h_state: [num_layers, batch_size, hidden_size]
            # c_state: [num_layers, batch_size, hidden_size]
            logits[t, :batch_size_t, :] = logit
            last = torch.argmax(F.softmax(logit, dim=1), dim=1)  # [batch_size,]

        return logits, sorted_dest_sequences, sorted_decode_lengths, sorted_indices


class SeqToSeqBadhanauAttentionLSTM(nn.Module):
    """
        BiLSTM sequence-to-sequence model with Badhanau attention.

        Args:
            encoder: nn.Module
                The wrapped encoder model
            decoder: nn.Module
                The wrapped decoder model
            device: torch.device
                The device on which the model is going.
            pad_index: int
        """

    def __init__(self, encoder: nn.Module, decoder: nn.Module, device: torch.device, pad_index: int):
        if encoder.hidden_size != decoder.hidden_size or encoder.n_layers != decoder.n_layers:
            raise ValueError('Encoder and Decoder must have the same number of recurrent hidden units and layers.')
        super(SeqToSeqBadhanauAttentionLSTM, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.pad_index = pad_index

    def create_mask(self, src_sequences: torch.IntTensor):
        """
        Create mask in order to ignore pad tokens.

        Args:
            src_sequences: torch.IntTensor[seq_len, batch_size]

        Returns:
            mask: torch.BoolTensor[seq_len, batch_size, 1]
        """
        mask = src_sequences != self.pad_index
        return mask.unsqueeze(2)

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
        mask = self.create_mask(src_sequences)

        # Encoding
        enc_outputs, (h_state, c_state) = self.encoder(input_sequences=src_sequences, sequence_lengths=src_lengths)
        assert enc_outputs.shape[0] == src_sequences.shape[0], f'{src_sequences.shape}, {enc_outputs.shape}'
        # enc_outputs: [seq_len, batch_size, hidden_size]
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
            logit, (h_state, c_state, _) = self.decoder(input_word_index=in_,
                                                        h_state=h_state[:, :batch_size_t, :].contiguous(),
                                                        c_state=c_state[:, :batch_size_t, :].contiguous(),
                                                        enc_outputs=enc_outputs[:, :batch_size_t, :],
                                                        mask=mask[:, :batch_size_t, :])
            # logit: [batch_size, vocab_size]
            # h_state: [num_layers, batch_size, hidden_size]
            # c_state: [num_layers, batch_size, hidden_size]
            logits[t, :batch_size_t, :] = logit
            last = torch.argmax(F.softmax(logit, dim=1), dim=1)  # [batch_size,]

        return logits, sorted_dest_sequences, sorted_decode_lengths, sorted_indices
