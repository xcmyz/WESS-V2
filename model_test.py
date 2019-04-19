import torch
import torch.nn as nn

from TransformerBlock import TransformerEncoderBlock as EncoderBlock
from TransformerBlock import TransformerDecoderBlock as DecoderBlock

if __name__ == "__main__":
    # Test
    test_decoder = DecoderBlock(768, 4, 4*768, 0.1)
    # print(test_decoder)
    encoder_input_1 = torch.randn(2, 12, 768)
    encoder_input_2 = torch.randn(2, 12, 768)
    decoder_input = torch.randn(2, 13, 768)
    output = test_decoder(decoder_input, encoder_input_1, encoder_input_2)
    # print(output.size())
