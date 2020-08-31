import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderLayerLSTM(nn.Module):
    """
    Encoder layer with LSTM.

    Arg(s):
        embedding_size (int)
        hidden_size (int)
        vocab_size (int)
        n_layers (int)
        dropout (float): must be less than 1.
        recurrent_dropout (float): must be less than 1.
        bidirectional (bool, optional): either directional or bidirectional. Default: False.

    Raises
        ValueError: if dropout or recurrent_dropout is less than 1.
    """

    def __init__(self, embedding_size: int, hidden_size: int, vocab_size: int,
                 n_layers: int, dropout: int, recurrent_dropout: int):
        if not (0 <= dropout < 1.0 and 0 <= recurrent_dropout < 1.0):
            raise ValueError('dropout and recurrent_dropout must be between 0 and 1.O.')
        super(EncoderLayerLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=n_layers,
                            dropout=(recurrent_dropout if n_layers > 1 else 0))

    def load_embeddings(self, embeddings: torch.FloatTensor):
        """
        Load embeddings.

        Args:
            embeddings (torch.FloatTensor[vocab_size, embedding_size])

        Raises:
            ValueError: If the embeddings' dimensions don't match.
        """
        if embeddings.size() != torch.Size([self.vocab_size, self.embedding_size]):
            raise ValueError('The dimensions of embeddings don\'t match.')
        self.embeddings.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune: bool = False):
        """
        Disable the training of embeddings.

        Args:
            fine_tune (bool, optional)
        """
        for param in self.embeddings.parameters():
            param.requires_grad = fine_tune

    def forward(self, input_sequences: torch.IntTensor, sequence_lengths: torch.IntTensor):
        """
        Pass inputs through the model

        Args:
            input_sequences (torch.IntTensor[seq_len, batch_size]):
            sequence_lengths (torch.IntTensor[batch_size,]):

        Raises:
            ValueError: if the batch_size of inputs doesn't match.

        Returns:
            outputs (torch.FloatTensor[seq_len, batch_size, hidden_size])
            h_state (torch.FloatTensor[n_layers, batch_size, hidden_size])
            c_state (torch.FloatTensor[n_layers, batch_size, hidden_size])
        """
        if input_sequences.shape[1] != sequence_lengths.shape[0]:
            raise ValueError('The batch_size of inputs does not match')
        embedded = self.embeddings(input_sequences)
        embedded = F.dropout(embedded, p=self.dropout)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, sequence_lengths)
        outputs, (h_state, c_state) = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        return outputs, (h_state, c_state)


class EncoderLayerBiLSTM(nn.Module):
    """
    Encoder layer with BiLSTM.

    Args:
        embedding_size (int)
        hidden_size (int)
        vocab_size (int)
        n_layers (int)
        dropout (float): must be less than 1.
        recurrent_dropout (float): must be less than 1.

    Raises
        ValueError: if dropout or recurrent_dropout is less than 1.
    """

    def __init__(self, embedding_size: int, hidden_size: int, vocab_size: int,
                 n_layers: int, dropout: int, recurrent_dropout: int):
        if not (0 <= dropout < 1.0 and 0 <= recurrent_dropout < 1.0):
            raise ValueError('dropout and recurrent_dropout must be between 0 and 1.O.')
        super(EncoderLayerBiLSTM, self).__init__()
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=n_layers,
                            bidirectional=True, dropout=(recurrent_dropout if n_layers > 1 else 0))

    def load_embeddings(self, embeddings: torch.FloatTensor):
        """
        Load embeddings.

        Args:
            embeddings (torch.FloatTensor[vocab_size, embedding_size])

        Raises:
            ValueError: If the embeddings' dimensions don't match.
        """
        if embeddings.size() != torch.Size([self.vocab_size, self.embedding_size]):
            raise ValueError('The dimensions of embeddings don\'t match.')
        self.embeddings.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune: bool = False):
        """
        Disable the training of embeddings.

        Args:
            fine_tune (bool, optional)
        """
        for param in self.embeddings.parameters():
            param.requires_grad = fine_tune

    def forward(self, input_sequences: torch.IntTensor, sequence_lengths: torch.IntTensor):
        """
        Pass inputs through the model

        Args:
            input_sequences (torch.IntTensor[seq_len, batch_size]):
            sequence_lengths (torch.IntTensor[batch_size,]):

        Raises:
            ValueError: if the batch_size of inputs doesn't match.

        Returns:
            outputs (torch.FloatTensor[seq_len, batch_size, hidden_size])
            h_state (torch.FloatTensor[n_layers, batch_size, hidden_size])
            c_state (torch.FloatTensor[n_layers, batch_size, hidden_size])
        """
        if input_sequences.shape[1] != sequence_lengths.shape[0]:
            raise ValueError('The batch_size of inputs does not match')
        embedded = self.embeddings(input_sequences)
        embedded = F.dropout(embedded, p=self.dropout)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, sequence_lengths)
        outputs, (h_state, c_state) = self.lstm(packed)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]
        h_state = h_state[:self.n_layers, :, :] + h_state[self.n_layers:, :, :]
        c_state = c_state[:self.n_layers, :, :] + c_state[self.n_layers:, :, :]
        return outputs, (h_state, c_state)
