import torch
import torch.nn as nn
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import os
import numpy as np

from WESS_V2 import WESS
from layers import get_bert_embedding
from data_utils import gen_text_sep
import hparams
import audio

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    # Test
    model = nn.DataParallel(WESS()).to(device)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model_bert = BertModel.from_pretrained('bert-base-uncased')
    print("Model Have Been Loaded.")

    checkpoint = torch.load(os.path.join(
        hparams.checkpoint_path, 'checkpoint_Tacotron_196500.pth.tar'))
    model.load_state_dict(checkpoint['model'])
    print("Sucessfully Loaded.")

    model.eval()
    text = "What is your name?"
    embeddings, tokens = get_bert_embedding(
        text, model_bert, tokenizer, return_token=True)
    # print(tokens)

    embeddings = embeddings[1:(embeddings.size(0)-1)]
    tokens = tokens[1:(len(tokens)-1)]
    characters, sep_list = gen_text_sep(tokens)
    print(np.shape(characters))
    print(sep_list)

    embeddings = [embeddings]
    characters = np.stack([characters])
    characters = torch.from_numpy(characters).long().to(device)
    mel_input = np.zeros([1, hparams.num_mels, 1], dtype=np.float32)
    mel_input = torch.Tensor(mel_input).to(device)
    sep_list = [sep_list]

    with torch.no_grad():
        mel_output = model(characters, embeddings, sep_list, mel_input)
        mel_output = mel_output.cpu().numpy()[0]
        # print(np.shape(mel_output))
    wav = audio.inv_mel_spectrogram(mel_output)
    # print(np.shape(linear_spec))
    print(np.shape(wav))
    wav = wav[:audio.find_endpoint(wav)]
    print(np.shape(wav))
    audio.save_wav(wav, "result.wav")
