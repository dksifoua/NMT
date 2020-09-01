import unittest
import torch
from nmt.model.encoder import EncoderLayerLSTM
from nmt.model.decoder import DecoderLayerLSTM
from nmt.model.seq2seq import SeqToSeqLSTM


class TestSeqToSeqLSTM(unittest.TestCase):

    def setUp(self) -> None:
        self.embedding_size = 10
        self.vocab_size = 30
        self.hidden_size = 32
        self.n_layers = 4
        self.dropout = 0.5
        self.embedding_dropout = 0.5
        self.recurrent_dropout = 0.5
        self.seq2seq = SeqToSeqLSTM(
            encoder=EncoderLayerLSTM(embedding_size=self.embedding_size, hidden_size=self.hidden_size,
                                     vocab_size=self.vocab_size, n_layers=self.n_layers,
                                     dropout=self.dropout, recurrent_dropout=self.recurrent_dropout),
            decoder=DecoderLayerLSTM(embedding_size=self.embedding_size, hidden_size=self.hidden_size,
                                     vocab_size=self.vocab_size, n_layers=self.n_layers,
                                     embedding_dropout=self.embedding_dropout,
                                     recurrent_dropout=self.recurrent_dropout),
            device=torch.device('cpu')
        )

    def test_build_model(self):
        with self.assertRaises(ValueError):
            self.seq2seq = SeqToSeqLSTM(
                encoder=EncoderLayerLSTM(embedding_size=self.embedding_size, hidden_size=self.hidden_size,
                                         vocab_size=self.vocab_size, n_layers=self.n_layers,
                                         dropout=self.dropout, recurrent_dropout=self.recurrent_dropout),
                decoder=DecoderLayerLSTM(embedding_size=self.embedding_size, hidden_size=self.hidden_size + 1,
                                         vocab_size=self.vocab_size, n_layers=self.n_layers + 1,
                                         embedding_dropout=self.embedding_dropout,
                                         recurrent_dropout=self.recurrent_dropout),
                device=torch.device('cpu')
            )

    def test_forward(self):
        seq_len, batch_size, tf_ratio = 10, 16, 0.8
        src_sequences = torch.randint(low=0, high=self.vocab_size, size=(seq_len, batch_size))
        src_lengths = torch.randint(low=1, high=seq_len + 1, size=(batch_size,))
        src_lengths, sorted_indices = torch.sort(src_lengths, dim=0, descending=True)
        src_sequences = src_sequences[:, sorted_indices]
        dest_sequences = torch.randint(low=0, high=self.vocab_size, size=(seq_len, batch_size))
        dest_lengths = torch.randint(low=1, high=seq_len + 1, size=(batch_size,))
        logits, sorted_dest_sequences, sorted_decode_lengths, sorted_indices = \
            self.seq2seq(src_sequences=src_sequences, src_lengths=src_lengths, dest_sequences=dest_sequences,
                         dest_lengths=dest_lengths, tf_ratio=tf_ratio)
        self.assertEqual(logits.size(), torch.Size([max(sorted_decode_lengths), batch_size, self.vocab_size]))
        self.assertEqual(sorted_dest_sequences.size(), torch.Size([seq_len, batch_size]))
        self.assertEqual(len(sorted_decode_lengths), batch_size)
        self.assertEqual(sorted_indices.size(), torch.Size([batch_size]))
