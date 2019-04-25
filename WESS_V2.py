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

    def cal_P_GRU(self, batch, gate_for_words_batch):
        list_input = list()
        list_output = list()

        for ind in range(len(gate_for_words_batch)-1):
            list_input.append(
                batch[gate_for_words_batch[ind]:gate_for_words_batch[ind+1]])

        # print(len(list_input))
        for one_word in list_input:
            one_word = torch.stack([one_word])

            pos_input = torch.Tensor(
                [i for i in range(one_word.size(1))]).long().to(device)
            position_embedding = self.position_embedding(pos_input)
            position_embedding = position_embedding.unsqueeze(0)

            one_word = one_word + position_embedding
            # output_one_word = self.P_transformer_block(one_word)
            # print(output_one_word.size())
            output_one_word = self.pre_GRU(one_word)
            output_one_word = self.get_GRU_embedding(output_one_word)
            output_one_word = output_one_word.squeeze(0)
            # word = output_one_word[output_one_word.size()[0]-1]
            list_output.append(word)

        output = torch.stack(list_output)
        return output

    def pad_by_word(self, words_batch):
        len_arr = np.array(list())
        for ele in words_batch:
            len_arr = np.append(len_arr, ele.size(0))
        max_size = int(len_arr.max())
        # print(max_size)

        def pad(tensor, target_length):
            embedding_size = tensor.size(1)
            pad_tensor = torch.zeros(1, embedding_size).to(device)

            for i in range(target_length-tensor.size(0)):
                tensor = torch.cat((tensor, pad_tensor))

            return tensor

        padded = list()
        for one_batch in words_batch:
            one_batch = pad(one_batch, max_size)
            padded.append(one_batch)
        padded = torch.stack(padded)

        return padded

    def pad_all(self, word_batch, embeddings):
        # print(word_batch.size())
        # print(embeddings.size())
        if word_batch.size(1) == embeddings.size(1):
            return word_batch, embeddings

        if word_batch.size(1) > embeddings.size(1):
            pad_len = word_batch.size(1) - embeddings.size(1)
            pad_vec = torch.zeros(word_batch.size(
                0), pad_len, embeddings.size(2)).float().to(device)
            embeddings = torch.cat((embeddings, pad_vec), 1)
            return word_batch, embeddings

        if word_batch.size(1) < embeddings.size(1):
            pad_len = embeddings.size(1) - word_batch.size(1)
            pad_vec = torch.zeros(word_batch.size(
                0), pad_len, embeddings.size(2)).float().to(device)
            word_batch = torch.cat((word_batch, pad_vec), 1)
            return word_batch, embeddings

    def forward(self, x, bert_embeddings, gate_for_words, mel_target=None):
        """
        :param: x: (batch, length)
        :param: bert_embeddings: (batch, length, 768)
        :param: gate_for_words: (batch, indexs)
        :param: mel_target: (batch, length, num_mel)
        """

        # Embedding
        x = self.embedding(x)

        # P_Transformer
        words_batch = list()
        for index, batch in enumerate(x):
            words_batch.append(self.cal_P_GRU(batch, gate_for_words[index]))

        words_batch = self.pad_by_word(words_batch)
        bert_embeddings = self.pad_by_word(bert_embeddings)
        words_batch, bert_embeddings = self.pad_all(
            words_batch, bert_embeddings)
        # print(words_batch.size())
        # print(bert_embeddings.size())
        bert_input = words_batch + bert_embeddings

        # Add Position Embedding
        pos_input = torch.stack([torch.Tensor([i for i in range(
            bert_input.size(1))]).long() for i in range(bert_input.size(0))]).to(device)
        pos_embedding = self.position_embedding(pos_input)
        bert_input = bert_input + pos_embedding

        encoder_output = self.bert_encoder(bert_input)


if __name__ == "__main__":
    # Test
    test_GRU = WESS_Encoder()
    print(test_GRU)
    x = torch.randn(2, 180, 768)
    print("x: ", x.size())

    out = test_GRU(x)
