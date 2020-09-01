import unittest
import torch
from nmt.model.encoder import EncoderLayerLSTM, EncoderLayerBiLSTM
from nmt.model.attention import LuongAttentionLayer, BadhanauAttentionLayer
from nmt.model.decoder import DecoderLayerLSTM, LuongDecoderLayerLSTM, BadhanauDecoderLayerLSTM
from nmt.model.seq2seq import SeqToSeqLSTM, SeqToSeqBiLSTM, SeqToSeqLuongAttentionLSTM, SeqToSeqBadhanauAttentionLSTM


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


class TestSeqToSeqBiLSTM(unittest.TestCase):

    def setUp(self) -> None:
        self.embedding_size = 10
        self.vocab_size = 30
        self.hidden_size = 32
        self.n_layers = 4
        self.dropout = 0.5
        self.embedding_dropout = 0.5
        self.recurrent_dropout = 0.5
        self.seq2seq = SeqToSeqBiLSTM(
            encoder=EncoderLayerBiLSTM(embedding_size=self.embedding_size, hidden_size=self.hidden_size,
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
            self.seq2seq = SeqToSeqBiLSTM(
                encoder=EncoderLayerBiLSTM(embedding_size=self.embedding_size, hidden_size=self.hidden_size,
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


class TestSeqToSeqLuongAttentionLSTM(unittest.TestCase):

    def setUp(self) -> None:
        self.embedding_size = 10
        self.vocab_size = 30
        self.hidden_size = 32
        self.n_layers = 4
        self.dropout = 0.5
        self.embedding_dropout = 0.5
        self.recurrent_dropout = 0.5
        self.seq2seq = SeqToSeqLuongAttentionLSTM(
            encoder=EncoderLayerBiLSTM(embedding_size=self.embedding_size, hidden_size=self.hidden_size,
                                       vocab_size=self.vocab_size, n_layers=self.n_layers,
                                       dropout=self.dropout, recurrent_dropout=self.recurrent_dropout),
            decoder=LuongDecoderLayerLSTM(embedding_size=self.embedding_size, hidden_size=self.hidden_size,
                                          vocab_size=self.vocab_size, n_layers=self.n_layers,
                                          dropout=self.dropout, embedding_dropout=self.embedding_dropout,
                                          recurrent_dropout=self.recurrent_dropout,
                                          attention_layer=LuongAttentionLayer(hidden_size=self.hidden_size)),
            device=torch.device('cpu'), pad_index=0
        )

    def test_build_model(self):
        with self.assertRaises(ValueError):
            self.seq2seq = SeqToSeqLuongAttentionLSTM(
                encoder=EncoderLayerBiLSTM(embedding_size=self.embedding_size, hidden_size=self.hidden_size,
                                           vocab_size=self.vocab_size, n_layers=self.n_layers,
                                           dropout=self.dropout, recurrent_dropout=self.recurrent_dropout),
                decoder=LuongDecoderLayerLSTM(embedding_size=self.embedding_size, hidden_size=self.hidden_size + 1,
                                              vocab_size=self.vocab_size, n_layers=self.n_layers + 1,
                                              dropout=self.dropout, embedding_dropout=self.embedding_dropout,
                                              recurrent_dropout=self.recurrent_dropout,
                                              attention_layer=LuongAttentionLayer(hidden_size=self.hidden_size + 2)),
                device=torch.device('cpu'), pad_index=0
            )

    def test_create_mask(self):
        src_sequences = torch.IntTensor([[1, 0], [1, 1], [0, 0]])
        mask = self.seq2seq.create_mask(src_sequences)
        self.assertEqual(mask.sum(), 3)
        self.assertEqual(mask.size(), torch.Size([3, 2, 1]))

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


class TestSeqToSeqBadhanauAttentionLSTM(unittest.TestCase):

    def setUp(self) -> None:
        self.embedding_size = 10
        self.vocab_size = 30
        self.hidden_size = 32
        self.n_layers = 4
        self.dropout = 0.5
        self.embedding_dropout = 0.5
        self.recurrent_dropout = 0.5
        self.seq2seq = SeqToSeqBadhanauAttentionLSTM(
            encoder=EncoderLayerBiLSTM(embedding_size=self.embedding_size, hidden_size=self.hidden_size,
                                       vocab_size=self.vocab_size, n_layers=self.n_layers,
                                       dropout=self.dropout, recurrent_dropout=self.recurrent_dropout),
            decoder=BadhanauDecoderLayerLSTM(embedding_size=self.embedding_size, hidden_size=self.hidden_size,
                                             vocab_size=self.vocab_size, n_layers=self.n_layers,
                                             dropout=self.dropout, embedding_dropout=self.embedding_dropout,
                                             recurrent_dropout=self.recurrent_dropout,
                                             attention_layer=BadhanauAttentionLayer(hidden_size=self.hidden_size)),
            device=torch.device('cpu'), pad_index=0
        )

    def test_build_model(self):
        with self.assertRaises(ValueError):
            self.seq2seq = SeqToSeqBadhanauAttentionLSTM(
                encoder=EncoderLayerBiLSTM(embedding_size=self.embedding_size, hidden_size=self.hidden_size,
                                           vocab_size=self.vocab_size, n_layers=self.n_layers,
                                           dropout=self.dropout, recurrent_dropout=self.recurrent_dropout),
                decoder=BadhanauDecoderLayerLSTM(embedding_size=self.embedding_size, hidden_size=self.hidden_size + 1,
                                                 vocab_size=self.vocab_size, n_layers=self.n_layers + 1,
                                                 dropout=self.dropout, embedding_dropout=self.embedding_dropout,
                                                 recurrent_dropout=self.recurrent_dropout,
                                                 attention_layer=BadhanauAttentionLayer(
                                                     hidden_size=self.hidden_size + 2)),
                device=torch.device('cpu'), pad_index=0
            )

    def test_create_mask(self):
        src_sequences = torch.IntTensor([[1, 0], [1, 1], [0, 0]])
        mask = self.seq2seq.create_mask(src_sequences)
        self.assertEqual(mask.sum(), 3)
        self.assertEqual(mask.size(), torch.Size([3, 2, 1]))

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
