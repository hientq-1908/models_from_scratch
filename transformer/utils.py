import re
from collections import Counter
import torch

def clean_data(path):
    file = open(path, 'r', encoding="UTF-8")
    lines = file.readlines()
    file.close()
    eng_data = []
    vie_data = []
    for line in lines:
        data = re.sub('CC.*\n$', '', line).split('\t')
        puntuation = re.compile('[!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]')
        eng = re.sub(puntuation, '', data[0]).lower()
        vie = re.sub(puntuation, '', data[1]).lower()
        eng_data.append(eng)
        vie_data.append(vie)  
    return eng_data, vie_data 

def create_vocab(path, sentences):
    words = [w for sentence in sentences for w in sentence.split()]
    vocab = Counter(words)
    vocab = [w for w in vocab.keys()]
    specials = ['<pad>', '<unk>', '<bos>', '<eos>']
    vocab = specials + vocab
    open(path, 'w+', encoding="UTF-8").write('\n'.join(vocab))

def get_tokenized(sentence, token2input, max_length):
    words = sentence.split()
    words = ['<bos>'] + words + ['<eos>']
    tokenized = []
    for i in range(max_length):
        if i < len(words):
            tokenized.append(token2input.get(words[i], token2input['<unk>']))
        else:
            tokenized.append(token2input['<pad>'])
    return tokenized        

def translate(model, text, eng_token2input, vie_input2token, max_length, device):
    with torch.no_grad():
        model.eval()
        model = model.to(device)
        inputs = get_tokenized(text, eng_token2input, max_length)
        inputs = torch.tensor(inputs).unsqueeze(0).to(device)
        outputs = torch.tensor([eng_token2input['<bos>']]).unsqueeze(0).to(device)
        in_pad_mask = make_pad_mask(inputs).to(device)
        context = model.encoder(inputs, in_pad_mask)
        specials = ['<pad>', '<unk>', '<bos>', '<eos>']
        special_idxs = [eng_token2input.get(i) for i in specials]
        for _ in range(24):
            out_tril_mask = make_tril_mask(outputs).to(device)
            logits = model.decoder(outputs, context, in_pad_mask, out_tril_mask)
            preds = logits.argmax(-1)[:, -1]
            preds = preds.view(1, 1)
            outputs = torch.cat((outputs, preds), 1).to(device)
        
        outputs = outputs.detach().cpu().numpy().squeeze()
        translated = [vie_input2token.get(idx) for idx in outputs if idx not in special_idxs]
        return translated
        
def make_pad_mask(tensor, pad=0):
    pad_mask = (tensor != 0).unsqueeze(1).unsqueeze(2)
    return pad_mask

def make_tril_mask(tensor):
        N, trg_len = tensor.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask
