import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from TransformerBlock import TransformerEncoderBlock as TransformerBlock
from TransformerBlock import TransformerDecoderBlock as DecoderBlock
from TransformerBlock import get_sinusoid_encoding_table
# from bert_embedding import get_bert_embedding
from layers import BERT, PreNet, LinearProjection
from text.symbols import symbols
# print(len(symbols))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class WESS_Encoder(nn.Module):
    """
    Encoder
    """

    def __init__(self,
                 vocab_max_size=2000,
                 embedding_size=768,
                 tb_hidden=768,
                 tb_attn_heads=4,
                 tb_feed_forward_hidden=4*768,
                 bert_hidden=768,
                 bert_n_layers=3,
                 bert_attn_heads=4,
                 bert_postnet_hidden=1024,
                 bert_postnet_output=256,
                 embedding_postnet_hidden=1024,
                 embedding_postnet_output=256,
                 dropout=0.1):
        """
        :param encoder_hparams
        """

        super(WESS_Encoder, self).__init__()
        self.vocab_max_size = vocab_max_size
        self.embedding_size = embedding_size
        self.tb_hidden = tb_hidden
        self.tb_attn_heads = tb_attn_heads
        self.tb_feed_forward_hidden = tb_feed_forward_hidden
        self.bert_hidden = bert_hidden
        self.bert_n_layers = bert_n_layers
        self.bert_attn_heads = bert_attn_heads
        self.bert_postnet_hidden = bert_postnet_hidden
        self.bert_postnet_output = bert_postnet_output
        self.embedding_postnet_hidden = embedding_postnet_hidden
        self.embedding_postnet_output = embedding_postnet_output
        self.dropout = dropout

        # Embeddings
        self.pre_embedding = nn.Embedding(len(symbols)+1, self.embedding_size)
        self.position_embedding = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(self.vocab_max_size, self.embedding_size), freeze=True)
