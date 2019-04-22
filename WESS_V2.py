import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

from TransformerBlock import TransformerEncoderBlock as TransformerBlock
from TransformerBlock import TransformerDecoderBlock as DecoderBlock
from TransformerBlock import get_sinusoid_encoding_table
# from bert_embedding import get_bert_embedding
from layers import BERT, LinearProjection, LinearNet_TwoLayer
from text.symbols import symbols
# print(len(symbols))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class WESS_Encoder(nn.Module):
    """
    Encoder
    (pre-transformer replaced by GRU)
    """

    def __init__(self,
                 vocab_max_size=2000,
                 embedding_size=768,
                 GRU_hidden_size=768,
                 GRU_num_layers=2,
                 GRU_batch_first=True,
                 GRU_bidirectional=True,
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
        self.GRU_hidden = GRU_hidden_size
        self.GRU_num_layers = GRU_num_layers
        self.GRU_batch_first = GRU_batch_first
        self.GRU_bidirectional = GRU_bidirectional
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

        self.pre_GRU = nn.GRU(input_size=self.embedding_size,
                              hidden_size=self.GRU_hidden,
                              num_layers=self.GRU_num_layers,
                              batch_first=self.GRU_batch_first,
                              dropout=self.dropout,
                              bidirectional=self.GRU_bidirectional)

        self.bert_encoder = BERT(hidden=self.bert_hidden,
                                 n_layers=self.bert_n_layers,
                                 attn_heads=self.bert_attn_heads,
                                 dropout=self.dropout)

        self.bert_post_net = LinearNet_TwoLayer(
            self.embedding_size, self.bert_postnet_hidden, self.bert_postnet_output)

    def init_GRU_hidden(self, batch_size, num_layers, hidden_size):
        if self.GRU_bidirectional:
            return torch.zeros(num_layers*2, batch_size,  hidden_size)
        else:
            return torch.zeros(num_layers*1, batch_size,  hidden_size)

    def get_GRU_embedding(self, GRU_output):
        out_1 = GRU_output[:, 0:1, :]
        out_2 = GRU_output[:, GRU_output.size(1)-1:, :]

        out = out_1 + out_2
        out = out[:, :, 0:out.size(2)//2] + out[:, :, out.size(2)//2:]

        # print(out.size())
        return out

    def forward(self, x):
        # Test
        GRU_h0 = self.init_GRU_hidden(
            x.size(0), self.GRU_num_layers, self.GRU_hidden)
        out, _ = self.pre_GRU(x, GRU_h0)
        # print(out.size())
        out = self.get_GRU_embedding(out)
        # print(out.size())

        return out


if __name__ == "__main__":
    # Test
    test_GRU = WESS_Encoder()
    print(test_GRU)
    x = torch.randn(2, 180, 768)
    print("x: ", x.size())

    out = test_GRU(x)
