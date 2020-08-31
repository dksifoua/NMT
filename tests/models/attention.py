import torch
import unittest
import numpy as np
from nmt.models.attention import BadhanauAttentionLayer
from nmt.models.attention import LuongAttentionLayer


class TestBadhanauAttentionLayer(unittest.TestCase):

    def setUp(self) -> None:
        self.hidden_size = 16
        self.attention = BadhanauAttentionLayer(hidden_size=self.hidden_size)

    def test_forward(self):
        n_layers, batch_size, seq_len = 2, 16, 30
        with self.assertRaises(ValueError):
            h_state = torch.randn((n_layers, batch_size, self.hidden_size))
            enc_outputs = torch.randn((seq_len, batch_size, self.hidden_size + 1))
            _ = self.attention(h_state, enc_outputs)

        h_state = torch.randn((n_layers, batch_size, self.hidden_size))
        enc_outputs = torch.randn((seq_len, batch_size, self.hidden_size))
        mask = torch.BoolTensor(np.random.randint(low=0, high=2, size=(seq_len, batch_size, 1)))
        attention_weights = self.attention(h_state, enc_outputs, mask)
        self.assertEqual(attention_weights.size(), torch.Size([seq_len, batch_size, 1]))


class TestLuongAttentionLayer(unittest.TestCase):

    def setUp(self) -> None:
        self.hidden_size = 16
        self.attention = LuongAttentionLayer(hidden_size=self.hidden_size)

    def test_forward(self):
        n_layers, batch_size, seq_len = 2, 16, 30
        with self.assertRaises(ValueError):
            h_state = torch.randn((n_layers, batch_size, self.hidden_size))
            enc_outputs = torch.randn((seq_len, batch_size, self.hidden_size + 1))
            _ = self.attention(h_state, enc_outputs)

        h_state = torch.randn((n_layers, batch_size, self.hidden_size))
        enc_outputs = torch.randn((seq_len, batch_size, self.hidden_size))
        mask = torch.BoolTensor(np.random.randint(low=0, high=2, size=(seq_len, batch_size, 1)))
        attention_weights = self.attention(h_state, enc_outputs, mask)
        self.assertEqual(attention_weights.size(), torch.Size([seq_len, batch_size, 1]))
