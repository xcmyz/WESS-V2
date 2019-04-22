import torch
import torch.nn as nn
from collections import OrderedDict

from TransformerBlock import TransformerEncoderBlock as TransformerBlock
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM


def add_cls_sep(text):
    return "[CLS] " + text + " [SEP]"


def get_bert_embedding(text, model, tokenizer, return_token=False):
    text = add_cls_sep(text)
    tokenized_text = tokenizer.tokenize(text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0 for i in range(len(indexed_tokens))]
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)

    output = encoded_layers[11][0]

    if return_token:
        return output, tokenized_text
    else:
        return output


class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, hidden, n_layers, attn_heads, dropout):
        """
        :param hidden: BERT model hidden size
        :param n_layers: numbers of Transformer blocks(layers)
        :param attn_heads: number of attention heads
        :param dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # multi-layers transformer blocks, deep network
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x):
        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x)

        return x


class LinearNet(nn.Module):
    """
    Linear layer for sequences
    """

    def __init__(self, input_size, output_size):
        # :param input_size: dimension of input
        # :param output_size: dimension of output
        # :param time_dim: index of time dimension

        super(LinearNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, input_):
        out = self.linear(input_)
        return out


class LinearNet_TwoLayer(nn.Module):
    """
    Linear Network for two layers
    """

    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        super(LinearNet_TwoLayer, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.layer_1 = nn.Linear(self.input_size, self.hidden_size)
        self.dropout_1 = nn.Dropout(dropout)
        self.layer_2 = nn.Linear(self.hidden_size, self.output_size)
        self.dropout_2 = nn.Dropout(dropout)

        nn.init.xavier_uniform_(self.layer_1.weight,
                                gain=nn.init.calculate_gain("linear"))
        nn.init.xavier_uniform_(self.layer_2.weight,
                                gain=nn.init.calculate_gain("linear"))

    def forward(self, x):
        x = self.layer_1(x)
        x = self.dropout_1(x)
        x = self.layer_2(x)
        output = self.dropout_2(x)

        return output


class FFN(nn.Module):
    """
    Feed Forward Network
    """

    def __init__(self, input_size, hidden_size, output_size, dropout=0.1):
        # :param input_size: dimension of input
        # :param hidden_size: dimension of hidden unit
        # :param output_size: dimension of output

        super(FFN, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.layer = nn.Sequential(OrderedDict([
            ('fc1', LinearNet(self.input_size, self.hidden_size)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(dropout)),
            ('fc2', LinearNet(self.hidden_size, self.output_size)),
            ('relu2', nn.ReLU()),
            ('dropout2', nn.Dropout(dropout)),
        ]))

    def forward(self, input_):
        out = self.layer(input_)
        return out


class LinearProjection(nn.Module):
    """
    Predict Gate
    """

    def __init__(self, input_size, hidden_size, output_size, dropout=0.1, w_init_gain="sigmoid"):
        # :param input_size: dimension of input
        # :param output_size: dimension of output

        super(LinearProjection, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.linear_layer_1 = nn.Linear(self.input_size, self.hidden_size)
        self.sigmoid_1 = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        self.linear_layer_2 = nn.Linear(self.hidden_size, self.output_size)
        self.sigmoid_2 = nn.Sigmoid()

        nn.init.xavier_uniform_(
            self.linear_layer_1.weight, gain=nn.init.calculate_gain(w_init_gain))
        nn.init.xavier_uniform_(
            self.linear_layer_2.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        x = self.linear_layer_1(x)
        x = self.sigmoid_1(x)
        x = self.dropout(x)
        x = self.linear_layer_2(x)
        output = self.sigmoid_2(x)

        # print(output)
        return output


if __name__ == "__main__":
    # Test
    linear_projection = LinearProjection(80, 256, 1)
    test_input = torch.randn(2, 123, 80)
    test_output = linear_projection(test_input)
    # print(test_output.size())
    # print(test_output)

    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # model_bert = BertModel.from_pretrained('bert-base-uncased')

    # text = "I am writing a paper for NeurISP."
    # test_embeddings = get_bert_embedding(text, model_bert, tokenizer)
    # print(test_embeddings)

    test_input = torch.randn(2, 123, 60)
    # print(test_input)

    test_FFN = FFN(60, 256, 3)
    test_output = test_FFN(test_input)
    # print(test_output)

    test_LN_T = LinearNet_TwoLayer(80, 256, 70)
    test_input = torch.randn(2, 167, 80)
    test_output = test_LN_T(test_input)
    print(test_output.size())

    print(test_input)
    print(test_output)
