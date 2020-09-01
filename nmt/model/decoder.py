import torch
import torch.nn as nn
import torch.nn.functional as F


class DecoderLayerLSTM(nn.Module):
    """
    Decoder layer with LSTM.

    Args:
        embedding_size: int
        hidden_size: int
        vocab_size: int
        n_layers: int
        embedding_dropout: float
            Must be in [0, 1.0]
        recurrent_dropout: float
            Must be in [0, 1.0]

    Raises
        ValueError: if dropout or recurrent_dropout is less than 1.
    """
    
    def __init__(self, embedding_size: int, hidden_size: int, vocab_size: int, n_layers: int,
                 embedding_dropout: float, recurrent_dropout: float):
        if not (0 <= embedding_dropout < 1.0 and 0 <= recurrent_dropout < 1.0):
            raise ValueError('embedding_dropout and recurrent_dropout must be between 0 and 1.O.')
        super(DecoderLayerLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.embedding_dropout = embedding_dropout
        self.recurrent_dropout = recurrent_dropout
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=n_layers,
                            dropout=(recurrent_dropout if n_layers > 1 else 0))
        self.fc = nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def load_embeddings(self, embeddings: torch.FloatTensor):
        """
        Load embeddings.

        Args:
            embeddings: torch.FloatTensor[vocab_size, embedding_size]

        Raises:
            ValueError: If the embeddings' dimensions don't match.
        """
        if embeddings.size() != torch.Size([self.vocab_size, self.embedding_size]):
            raise ValueError('The dimensions of embeddings don\'t match.')
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune: bool = False):
        """
        Disable the training of embeddings.

        Args:
            fine_tune: bool, optional
                Default: False.
        """
        for param in self.embedding.parameters():
            param.requires_grad = fine_tune

    def forward(self, input_word_index: torch.IntTensor, h_state: torch.FloatTensor, c_state: torch.FloatTensor):
        """
        Pass inputs through the model.

        Args:
            input_word_index: torch.IntTensor[batch_size,]
            h_state: torch.FloatTensor[n_layer, batch_size, hidden_size]
            c_state: torch.FloatTensor[n_layer, batch_size, hidden_size]

        Returns:
            logit: torch.FloatTensor[batch_size, vocab_size]
            h_state: torch.FloatTensor[n_layer, batch_size, hidden_size]
            c_state: torch.FloatTensor[n_layer, batch_size, hidden_size]
        """
        embedded = self.embedding(input_word_index.unsqueeze(0))
        embedded = F.dropout(embedded, p=self.embedding_dropout)
        output, (h_state, c_state) = self.lstm(embedded, (h_state, c_state))
        logit = self.fc(output.squeeze(0))
        return logit, (h_state, c_state)


class LuongDecoderLayerLSTM(nn.Module):
    """
    Luong Decoder layer with LSTM.

    Args:
        embedding_size: int
        hidden_size: int
        vocab_size: int
        n_layers: int
        dropout: float
            Must be in [0, 1.0]
        embedding_dropout: float
            Must be in [0, 1.0]
        recurrent_dropout: float
            Must be in [0, 1.0]
        attention_layer: nn.Module
            The wrapped attention layer.

    Raises
        ValueError: if dropout or recurrent_dropout is less than 1.
    """

    def __init__(self, embedding_size: int, hidden_size: int, vocab_size: int, n_layers: int,
                 dropout: float, embedding_dropout: float, recurrent_dropout: float, attention_layer: nn.Module):
        if not (0 <= dropout < 1.0 and 0 <= embedding_dropout < 1.0 and 0 <= recurrent_dropout < 1.0):
            raise ValueError('dropout, embedding_dropout and recurrent_dropout must be between 0 and 1.O.')
        super(LuongDecoderLayerLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding_dropout = embedding_dropout
        self.recurrent_dropout = recurrent_dropout
        self.attention_layer = attention_layer
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size)
        self.lstm = nn.LSTM(input_size=embedding_size, hidden_size=hidden_size, num_layers=n_layers,
                            dropout=(recurrent_dropout if n_layers > 1 else 0))
        self.fc1 = nn.Linear(in_features=hidden_size * 2, out_features=hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def load_embeddings(self, embeddings: torch.FloatTensor):
        """
        Load embeddings.

        Args:
            embeddings: torch.FloatTensor[vocab_size, embedding_size]

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
            fine_tune: bool, optional
                Default: False.
        """
        for param in self.embeddings.parameters():
            param.requires_grad = fine_tune

    def forward(self, input_word_index: torch.IntTensor, h_state: torch.FloatTensor, c_state: torch.FloatTensor,
                enc_outputs: torch.FloatTensor, mask: torch.BoolTensor):
        """
        Pass inputs through the model.

        Args:
            input_word_index: torch.IntTensor[batch_size,]
            h_state: torch.FloatTensor[n_layers, batch_size, hidden_size]
            c_state: torch.FloatTensor[n_layers, batch_size, hidden_size]
            enc_outputs: torch.FloatTensor[seq_len, batch_size, hidden_size]
            mask:

        Returns:
            logit: torch.FloatTensor[batch_size, vocab_size]
            h_state: torch.FloatTensor[n_layers, batch_size, hidden_size]
            c_state: torch.FloatTensor[n_layers, batch_size, hidden_size]
            attention_weights: torch.FloatTensor[n_layers, batch_size, 1]
        """
        embedded = self.embedding(input_word_index.unsqueeze(0))
        embedded = F.dropout(embedded, p=self.embedding_dropout)
        output, (h_state, c_state) = self.lstm(embedded, (h_state, c_state))
        # output: [seq_len=1, batch_size, hidden_size]
        # h_state: [n_layers, batch_size, hidden_size]
        # c_state: [n_layers, batch_size, hidden_size]

        # Compute attention weights
        attention_weights = self.attention_layer(h_state=h_state, enc_outputs=enc_outputs,
                                                 mask=mask)  # attention_weights: [seq_len, batch_size, 1]
        # Compute the context vector
        context_vector = torch.bmm(
            enc_outputs.permute(1, 2, 0),  # [batch_size, hidden_size, seq_len]
            attention_weights.permute(1, 0, 2),  # [batch_size, seq_len, 1]
        ).permute(2, 0, 1)  # [1, batch_size, hidden_size]

        # New input: concatenate context_vector with hidden_states
        new_input = torch.cat((context_vector, output), dim=2)  # [1, batch_size, hidden_size * 2]

        # Get logit
        x = self.fc1(new_input.squeeze(0))  # [batch_size, hidden_size]
        x = F.leaky_relu(x)
        x = F.dropout(x, p=self.dropout)
        logit = self.fc2(x)  # [batch_size, vocab_size]

        return logit, (h_state, c_state, attention_weights.squeeze(2))


class BadhanauDecoderLayerLSTM(nn.Module):
    """
    Badhanau Decoder layer with LSTM.

    Args:
        embedding_size: int
        hidden_size: int
        vocab_size: int
        n_layers: int
        dropout: float
            Must be in [0, 1.0]
        embedding_dropout: float
            Must be in [0, 1.0]
        recurrent_dropout: float
            Must be in [0, 1.0]
        attention_layer: nn.Module
            The wrapped attention layer.

    Raises
        ValueError: if dropout or recurrent_dropout is less than 1.
    """

    def __init__(self, embedding_size: int, hidden_size: int, vocab_size: int, n_layers: int,
                 dropout: float, embedding_dropout: float, recurrent_dropout: float, attention_layer: nn.Module):
        if not (0 <= dropout < 1.0 and 0 <= embedding_dropout < 1.0 and 0 <= recurrent_dropout < 1.0):
            raise ValueError('dropout, embedding_dropout and recurrent_dropout must be between 0 and 1.O.')
        super(BadhanauDecoderLayerLSTM, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding_dropout = embedding_dropout
        self.recurrent_dropout = recurrent_dropout
        self.attention_layer = attention_layer
        self.embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_size)
        self.lstm = nn.LSTM(input_size=embedding_size + hidden_size, hidden_size=hidden_size, num_layers=n_layers,
                            dropout=(recurrent_dropout if n_layers > 1 else 0))
        self.fc = nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def load_embeddings(self, embeddings: torch.FloatTensor):
        """
        Load embeddings.

        Args:
            embeddings: torch.FloatTensor[vocab_size, embedding_size]

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
            fine_tune: bool, optional
                Default: False.
        """
        for param in self.embeddings.parameters():
            param.requires_grad = fine_tune

    def forward(self, input_word_index: torch.IntTensor, h_state: torch.FloatTensor, c_state: torch.FloatTensor,
                enc_outputs: torch.FloatTensor, mask: torch.BoolTensor):
        """
        Pass inputs through the model.

        Args:
            input_word_index: torch.IntTensor[batch_size,]
            h_state: torch.FloatTensor[n_layers, batch_size, hidden_size]
            c_state: torch.FloatTensor[n_layers, batch_size, hidden_size]
            enc_outputs: torch.FloatTensor[seq_len, batch_size, hidden_size]
            mask:

        Returns:
            logit: torch.FloatTensor[batch_size, vocab_size]
            h_state: torch.FloatTensor[n_layers, batch_size, hidden_size]
            c_state: torch.FloatTensor[n_layers, batch_size, hidden_size]
            attention_weights: torch.FloatTensor[n_layers, batch_size]
        """
        embedded = self.embedding(input_word_index.unsqueeze(0))
        embedded = F.dropout(embedded, p=self.embedding_dropout)

        # Compute attention weights
        attention_weights = self.attention_layer(h_state=h_state, enc_outputs=enc_outputs,
                                                 mask=mask)  # attention_weights: [seq_len, batch_size, 1]
        # Compute Context Vector
        context_vector = torch.bmm(
            enc_outputs.permute(1, 2, 0),  # [batch_size, hidden_size, seq_len]
            attention_weights.permute(1, 0, 2),  # [batch_size, seq_len, 1]
        ).permute(2, 0, 1)  # [1, batch_size, hidden_size]

        # New input: concatenate context_vector with hidden_states
        new_input = torch.cat((embedded, context_vector), dim=2)  # [1, batch_size, hidden_size + embedding_size]

        outputs, (h_state, c_state) = self.lstm(new_input, (h_state, c_state))
        # outputs: [seq_len=1, batch_size, hidden_size]
        # h_state: [n_layers, batch_size, hidden_size]
        # c_state: [n_layers, batch_size, hidden_size]

        # Get logit
        logit = self.fc(outputs.squeeze(0))  # [batch_size, vocab_size]

        return logit, (h_state, c_state, attention_weights.squeeze(2))
