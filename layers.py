import torch
import torch.nn as nn
from collections import OrderedDict

from TransformerBlock import TransformerEncoderBlock as TransformerBlock
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM


def add_cls_sep(text):
    return "[CLS] " + text + " [SEP]"


def get_bert_embedding(text, model, tokenizer, return_token=False):
    text = add_cls_sep(text)
    # print(text)
    tokenized_text = tokenizer.tokenize(text)
    # print(tokenized_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # print(indexed_tokens)
    segments_ids = [0 for i in range(len(indexed_tokens))]
    # print(segments_ids)
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    with torch.no_grad():
        encoded_layers, _ = model(tokens_tensor, segments_tensors)

    # print(torch.Tensor(encoded_layers))
    # print(len(encoded_layers))
    # print(len(encoded_layers[0]))
    # print(len(encoded_layers[0][0]))
    # print(len(encoded_layers[0][0][0]))
    output = encoded_layers[11][0]
    # print(len(output))
    # print(len(output[0]))
    # print(output[0])

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


class SeqLinear(nn.Module):
    """
    Linear layer for sequences
    """

    def __init__(self, input_size, output_size):
        # :param input_size: dimension of input
        # :param output_size: dimension of output
        # :param time_dim: index of time dimension

        super(SeqLinear, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.linear = nn.Linear(input_size, output_size)

    def forward(self, input_):
        out = self.linear(input_)
        return out


class PreNet(nn.Module):
    """
    Prenet before passing through the network
    """

    def __init__(self, input_size, hidden_size, output_size, dropout=0.5):
        # :param input_size: dimension of input
        # :param hidden_size: dimension of hidden unit
        # :param output_size: dimension of output

        super(PreNet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.layer = nn.Sequential(OrderedDict([
            ('fc1', SeqLinear(self.input_size, self.hidden_size)),
            ('relu1', nn.ReLU()),
            ('dropout1', nn.Dropout(dropout)),
            ('fc2', SeqLinear(self.hidden_size, self.output_size)),
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

    def __init__(self, input_size, output_size, hidden_size=256, dropout=0.5, bias=True, w_init_gain="sigmoid"):
        # :param input_size: dimension of input
        # :param output_size: dimension of output

        super(LinearProjection, self).__init__()
        self.linear_layer = nn.Linear(input_size, output_size, bias=bias)
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        # self.linear_layer = nn.Sequential(OrderedDict([
        #     ('fc1', SeqLinear(self.input_size, self.hidden_size)),
        #     ('relu1', nn.ReLU()),
        #     ('dropout1', nn.Dropout(dropout)),
        #     ('fc2', SeqLinear(self.hidden_size, self.output_size)),
        #     ('relu2', nn.ReLU()),
        #     ('dropout2', nn.Dropout(dropout)),
        # ]))

        nn.init.xavier_uniform_(self.linear_layer.weight,
                                gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        output = self.linear_layer(x)
        # output = torch.softmax(output, dim=2)

        # print(output)
        return output


if __name__ == "__main__":
    # Test

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    print(model)

    # text = "I love you."
    # text = add_cls_sep(text)
    # print(text)
    # tokenized_text = tokenizer.tokenize(text)
    # print(tokenized_text)
    # indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    # print(indexed_tokens)
    # segments_ids = [0 for i in range(len(indexed_tokens))]
    # print(segments_ids)
    # tokens_tensor = torch.tensor([indexed_tokens])
    # segments_tensors = torch.tensor([segments_ids])

    # with torch.no_grad():
    #     encoded_layers, _ = model(tokens_tensor, segments_tensors)

    # # print(torch.Tensor(encoded_layers))
    # print(len(encoded_layers))
    # print(len(encoded_layers[0]))
    # print(len(encoded_layers[0][0]))
    # print(len(encoded_layers[0][0][0]))
    # output = encoded_layers[11][0]
    # print(len(output))
    # print(len(output[0]))
    # print(output[0])

    output = get_bert_embedding("I love you.", model, tokenizer)
    print(len(output))
    print(len(output[0]))

    test_BERT = BERT(768, 3, 4, 0.1)
    print(test_BERT)

    test_prenet = PreNet(80, 1000, 768, 0.5)
    test_input = torch.randn(2, 188, 80)
    test_output = test_prenet(test_input)
    print(test_output.size())

    test_LP = LinearProjection(1024, 1)
    test_LP_input = torch.randn(2, 168, 1024)
    test_LP_output = test_LP(test_LP_input)
    print(test_LP_output.size())
    print(test_LP_output)
