import torch
import unittest
import numpy as np
from nmt.models.encoder import EncoderLayerLSTM
from nmt.models.encoder import EncoderLayerBiLSTM


class TestEncoderLayerLSTM(unittest.TestCase):

    def setUp(self) -> None:
        self.embedding_size = 10
        self.vocab_size = 30
        self.hidden_size = 32
        self.n_layers = 4
        self.dropout = 0.5
        self.recurrent_dropout = 0.5
        self.encoder = EncoderLayerLSTM(embedding_size=self.embedding_size, hidden_size=self.hidden_size,
                                        vocab_size=self.vocab_size, n_layers=self.n_layers,
                                        dropout=self.dropout, recurrent_dropout=self.recurrent_dropout)

    def test_build_model(self):
        with self.assertRaises(ValueError):
            _ = EncoderLayerLSTM(embedding_size=self.embedding_size, hidden_size=self.hidden_size,
                                 vocab_size=self.vocab_size, n_layers=self.n_layers,
                                 dropout=1.2, recurrent_dropout=-0.1)
        with self.assertRaises(ValueError):
            _ = EncoderLayerLSTM(embedding_size=self.embedding_size, hidden_size=self.hidden_size,
                                 vocab_size=self.vocab_size, n_layers=self.n_layers,
                                 dropout=0.2, recurrent_dropout=-0.1)
        with self.assertRaises(ValueError):
            _ = EncoderLayerLSTM(embedding_size=self.embedding_size, hidden_size=self.hidden_size,
                                 vocab_size=self.vocab_size, n_layers=self.n_layers,
                                 dropout=-0.2, recurrent_dropout=0.1)

    def test_load_embeddings(self):
        with self.assertRaises(ValueError):
            embeddings = torch.FloatTensor(np.random.randn(12, 54))
            self.encoder.load_embeddings(embeddings=embeddings)
        embeddings = torch.FloatTensor(np.random.randn(self.vocab_size, self.embedding_size))
        self.encoder.load_embeddings(embeddings=embeddings)

    def test_fine_tune_embeddings(self):
        self.encoder.fine_tune_embeddings(fine_tune=True)
        for param in self.encoder.embeddings.parameters():
            self.assertTrue(param.requires_grad)

    def test_forward(self):
        seq_len, batch_size = 10, 16
        input_sequences = torch.randint(low=0, high=self.vocab_size, size=(seq_len, batch_size))
        sequence_lengths = torch.randint(low=1, high=seq_len + 1, size=(batch_size,))
        sequence_lengths, sorted_indices = torch.sort(sequence_lengths, dim=0, descending=True)
        input_sequences = input_sequences[:, sorted_indices]
        with self.assertRaises(ValueError):
            _, (_, _) = self.encoder(input_sequences=torch.randint(low=0, high=self.vocab_size,
                                                                   size=(seq_len, batch_size + 1)),
                                     sequence_lengths=torch.randint(low=1, high=seq_len + 1, size=(batch_size,)))
        outputs, (h_state, c_state) = self.encoder(input_sequences=input_sequences, sequence_lengths=sequence_lengths)
        self.assertEqual(outputs.size(), torch.Size([seq_len, batch_size, self.hidden_size]))
        self.assertEqual(h_state.size(), torch.Size([self.n_layers, batch_size, self.hidden_size]))
        self.assertEqual(c_state.size(), torch.Size([self.n_layers, batch_size, self.hidden_size]))


class TestEncoderLayerBiLSTM(unittest.TestCase):

    def setUp(self) -> None:
        self.embedding_size = 10
        self.vocab_size = 30
        self.hidden_size = 32
        self.n_layers = 4
        self.dropout = 0.5
        self.recurrent_dropout = 0.5
        self.encoder = EncoderLayerBiLSTM(embedding_size=self.embedding_size, hidden_size=self.hidden_size,
                                          vocab_size=self.vocab_size, n_layers=self.n_layers,
                                          dropout=self.dropout, recurrent_dropout=self.recurrent_dropout)

    def test_build_model(self):
        with self.assertRaises(ValueError):
            _ = EncoderLayerBiLSTM(embedding_size=self.embedding_size, hidden_size=self.hidden_size,
                                   vocab_size=self.vocab_size, n_layers=self.n_layers,
                                   dropout=1.2, recurrent_dropout=-0.1)
        with self.assertRaises(ValueError):
            _ = EncoderLayerBiLSTM(embedding_size=self.embedding_size, hidden_size=self.hidden_size,
                                   vocab_size=self.vocab_size, n_layers=self.n_layers,
                                   dropout=0.2, recurrent_dropout=-0.1)
        with self.assertRaises(ValueError):
            _ = EncoderLayerBiLSTM(embedding_size=self.embedding_size, hidden_size=self.hidden_size,
                                   vocab_size=self.vocab_size, n_layers=self.n_layers,
                                   dropout=-0.2, recurrent_dropout=0.1)

    def test_load_embeddings(self):
        with self.assertRaises(ValueError):
            embeddings = torch.FloatTensor(np.random.randn(12, 54))
            self.encoder.load_embeddings(embeddings=embeddings)
        embeddings = torch.FloatTensor(np.random.randn(self.vocab_size, self.embedding_size))
        self.encoder.load_embeddings(embeddings=embeddings)

    def test_fine_tune_embeddings(self):
        self.encoder.fine_tune_embeddings(fine_tune=True)
        for param in self.encoder.embeddings.parameters():
            self.assertTrue(param.requires_grad)

    def test_forward(self):
        seq_len, batch_size = 10, 16
        input_sequences = torch.randint(low=0, high=self.vocab_size, size=(seq_len, batch_size))
        sequence_lengths = torch.randint(low=1, high=seq_len + 1, size=(batch_size,))
        sequence_lengths, sorted_indices = torch.sort(sequence_lengths, dim=0, descending=True)
        input_sequences = input_sequences[:, sorted_indices]
        with self.assertRaises(ValueError):
            _, (_, _) = self.encoder(input_sequences=torch.randint(low=0, high=self.vocab_size,
                                                                   size=(seq_len, batch_size + 1)),
                                     sequence_lengths=torch.randint(low=1, high=seq_len + 1, size=(batch_size,)))
        outputs, (h_state, c_state) = self.encoder(input_sequences=input_sequences, sequence_lengths=sequence_lengths)
        self.assertEqual(outputs.size(), torch.Size([seq_len, batch_size, self.hidden_size]))
        self.assertEqual(h_state.size(), torch.Size([self.n_layers, batch_size, self.hidden_size]))
        self.assertEqual(c_state.size(), torch.Size([self.n_layers, batch_size, self.hidden_size]))
