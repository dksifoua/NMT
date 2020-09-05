SPACY_CONFIG = {
    'fr': 'fr_core_news_lg',
    'en': 'en_core_web_lg'
}


class EncoderLSTMConfig:
    EMBEDDING_SIZE = 300
    HIDDEN_SIZE = 256
    N_LAYERS = 2
    EMBEDDING_DROPOUT = 0.
    REC_DROPOUT = 0.25


class DecoderLSTMConfig:
    EMBEDDING_SIZE = 300
    HIDDEN_SIZE = 256
    N_LAYERS = 2
    EMBEDDING_DROPOUT = 0.
    REC_DROPOUT = 0.25
    DROPOUT = 0.15
