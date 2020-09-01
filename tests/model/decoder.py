import unittest
import numpy as np
import torch
from nmt.model.decoder import DecoderLayerLSTM
from nmt.model.attention import LuongAttentionLayer
from nmt.model.attention import BadhanauAttentionLayer
from nmt.model.decoder import LuongDecoderLayerLSTM
from nmt.model.decoder import BadhanauDecoderLayerLSTM


class TestDecoderLayerLSTM(unittest.TestCase):

    def setUp(self) -> None:
        self.embedding_size = 10
        self.vocab_size = 30
        self.hidden_size = 32
        self.n_layers = 4
        self.embedding_dropout = 0.5
        self.recurrent_dropout = 0.5
        self.decoder = DecoderLayerLSTM(embedding_size=self.embedding_size, hidden_size=self.hidden_size,
                                        vocab_size=self.vocab_size, n_layers=self.n_layers,
                                        embedding_dropout=self.embedding_dropout,
                                        recurrent_dropout=self.recurrent_dropout)

    def test_build_model(self):
        with self.assertRaises(ValueError):
            _ = DecoderLayerLSTM(embedding_size=self.embedding_size, hidden_size=self.hidden_size,
                                 vocab_size=self.vocab_size, n_layers=self.n_layers,
                                 embedding_dropout=-0.1, recurrent_dropout=1.1)

    def test_load_embeddings(self):
        with self.assertRaises(ValueError):
            embeddings = torch.FloatTensor(np.random.randn(12, 54))
            self.decoder.load_embeddings(embeddings=embeddings)
        embeddings = torch.FloatTensor(np.random.randn(self.vocab_size, self.embedding_size))
        self.decoder.load_embeddings(embeddings=embeddings)

    def test_fine_tune_embeddings(self):
        self.decoder.fine_tune_embeddings(fine_tune=True)
        for param in self.decoder.embedding.parameters():
            self.assertTrue(param.requires_grad)

    def test_forward(self):
        seq_len, batch_size = 10, 16
        input_word_index = torch.randint(low=0, high=self.vocab_size, size=(batch_size,))
        h_state = torch.randn((self.n_layers, batch_size, self.hidden_size))
        c_state = torch.randn((self.n_layers, batch_size, self.hidden_size))
        logit, (h_state, c_state) = self.decoder(input_word_index=input_word_index, h_state=h_state, c_state=c_state)
        self.assertEqual(logit.size(), torch.Size([batch_size, self.vocab_size]))
        self.assertEqual(h_state.size(), torch.Size([self.n_layers, batch_size, self.hidden_size]))
        self.assertEqual(c_state.size(), torch.Size([self.n_layers, batch_size, self.hidden_size]))


class TestLuongDecoderLayerLSTM(unittest.TestCase):

    def setUp(self) -> None:
        self.embedding_size = 10
        self.vocab_size = 30
        self.hidden_size = 32
        self.n_layers = 4
        self.dropout = 0.5
        self.embedding_dropout = 0.5
        self.recurrent_dropout = 0.5
        self.decoder = LuongDecoderLayerLSTM(embedding_size=self.embedding_size, hidden_size=self.hidden_size,
                                             vocab_size=self.vocab_size, n_layers=self.n_layers, dropout=self.dropout,
                                             embedding_dropout=self.embedding_dropout,
                                             recurrent_dropout=self.recurrent_dropout,
                                             attention_layer=LuongAttentionLayer(hidden_size=self.hidden_size))

    def test_build_model(self):
        with self.assertRaises(ValueError):
            _ = LuongDecoderLayerLSTM(embedding_size=self.embedding_size, hidden_size=self.hidden_size,
                                      vocab_size=self.vocab_size, n_layers=self.n_layers, dropout=-0.1,
                                      embedding_dropout=1.1, recurrent_dropout=1.2,
                                      attention_layer=LuongAttentionLayer(hidden_size=self.hidden_size))

    def test_load_embeddings(self):
        with self.assertRaises(ValueError):
            embeddings = torch.FloatTensor(np.random.randn(12, 54))
            self.decoder.load_embeddings(embeddings=embeddings)
        embeddings = torch.FloatTensor(np.random.randn(self.vocab_size, self.embedding_size))
        self.decoder.load_embeddings(embeddings=embeddings)

    def test_fine_tune_embeddings(self):
        self.decoder.fine_tune_embeddings(fine_tune=True)
        for param in self.decoder.embedding.parameters():
            self.assertTrue(param.requires_grad)

    def test_forward(self):
        seq_len, batch_size = 10, 16
        input_word_index = torch.randint(low=0, high=self.vocab_size, size=(batch_size,))
        h_state = torch.randn((self.n_layers, batch_size, self.hidden_size))
        c_state = torch.randn((self.n_layers, batch_size, self.hidden_size))
        enc_outputs = torch.randn((seq_len, batch_size, self.hidden_size))
        mask = torch.BoolTensor(np.random.randint(low=0, high=2, size=(seq_len, batch_size, 1)))
        logit, (h_state, c_state, attention_weights) = self.decoder(input_word_index=input_word_index, h_state=h_state,
                                                                    c_state=c_state, enc_outputs=enc_outputs, mask=mask)
        self.assertEqual(logit.size(), torch.Size([batch_size, self.vocab_size]))
        self.assertEqual(h_state.size(), torch.Size([self.n_layers, batch_size, self.hidden_size]))
        self.assertEqual(c_state.size(), torch.Size([self.n_layers, batch_size, self.hidden_size]))
        self.assertEqual(attention_weights.size(), torch.Size([self.n_layers, batch_size, 1]))


class TestBadhanauDecoderLayerLSTM(unittest.TestCase):

    def setUp(self) -> None:
        self.embedding_size = 10
        self.vocab_size = 30
        self.hidden_size = 32
        self.n_layers = 4
        self.dropout = 0.5
        self.embedding_dropout = 0.5
        self.recurrent_dropout = 0.5
        self.decoder = LuongDecoderLayerLSTM(embedding_size=self.embedding_size, hidden_size=self.hidden_size,
                                             vocab_size=self.vocab_size, n_layers=self.n_layers, dropout=self.dropout,
                                             embedding_dropout=self.embedding_dropout,
                                             recurrent_dropout=self.recurrent_dropout,
                                             attention_layer=LuongAttentionLayer(hidden_size=self.hidden_size))

    def test_build_model(self):
        with self.assertRaises(ValueError):
            _ = LuongDecoderLayerLSTM(embedding_size=self.embedding_size, hidden_size=self.hidden_size,
                                      vocab_size=self.vocab_size, n_layers=self.n_layers, dropout=-0.1,
                                      embedding_dropout=1.1, recurrent_dropout=1.2,
                                      attention_layer=LuongAttentionLayer(hidden_size=self.hidden_size))

    def test_load_embeddings(self):
        with self.assertRaises(ValueError):
            embeddings = torch.FloatTensor(np.random.randn(12, 54))
            self.decoder.load_embeddings(embeddings=embeddings)
        embeddings = torch.FloatTensor(np.random.randn(self.vocab_size, self.embedding_size))
        self.decoder.load_embeddings(embeddings=embeddings)

    def test_fine_tune_embeddings(self):
        self.decoder.fine_tune_embeddings(fine_tune=True)
        for param in self.decoder.embedding.parameters():
            self.assertTrue(param.requires_grad)

    def test_forward(self):
        seq_len, batch_size = 10, 16
        input_word_index = torch.randint(low=0, high=self.vocab_size, size=(batch_size,))
        h_state = torch.randn((self.n_layers, batch_size, self.hidden_size))
        c_state = torch.randn((self.n_layers, batch_size, self.hidden_size))
        enc_outputs = torch.randn((seq_len, batch_size, self.hidden_size))
        mask = torch.BoolTensor(np.random.randint(low=0, high=2, size=(seq_len, batch_size, 1)))
        logit, (h_state, c_state, attention_weights) = self.decoder(input_word_index=input_word_index, h_state=h_state,
                                                                    c_state=c_state, enc_outputs=enc_outputs, mask=mask)
        self.assertEqual(logit.size(), torch.Size([batch_size, self.vocab_size]))
        self.assertEqual(h_state.size(), torch.Size([self.n_layers, batch_size, self.hidden_size]))
        self.assertEqual(c_state.size(), torch.Size([self.n_layers, batch_size, self.hidden_size]))
        self.assertEqual(attention_weights.size(), torch.Size([self.n_layers, batch_size, 1]))
