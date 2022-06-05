from torch.utils.data import Dataset
from utils import clean_data, get_tokenized
import torch

class EngVieDataset(Dataset):
    def __init__(self, txt_path, eng_vocab, vie_vocab) -> None:
        super().__init__()
        self.eng_data, self.vie_data = clean_data(txt_path)
        self.eng_vocab = eng_vocab
        self.vie_vocab = vie_vocab
        self.eng_token2input = {t: i for i, t in enumerate(self.eng_vocab)}
        self.eng_input2token = {v: k for k, v in self.eng_token2input.items()}
        self.vie_token2input = {t: i for i, t in enumerate(self.vie_vocab)}
        self.vie_input2token = {v: k for k, v in self.vie_token2input.items()}

        # self.UNK = self.eng_vocab['<unk>']
        # self.PAD = self.eng_vocab['<pad>']
        # self.BOS = self.eng_vocab['<bos>']
        # self.EOS = self.eng_vocab['<eos>']
        self.max_length = 25
    
    def __len__(self):
        return len(self.eng_data)
    
    def __getitem__(self, index):
        eng_sentence, vie_sentence = self.eng_data[index], self.vie_data[index]
        eng_tokenized = get_tokenized(eng_sentence, self.eng_token2input, self.max_length)
        vie_tokenized = get_tokenized(vie_sentence, self.vie_token2input, self.max_length)
        return torch.tensor(eng_tokenized), torch.tensor(vie_tokenized)
