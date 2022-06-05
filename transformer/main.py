
from network import Transformer
from trainer import Trainer
from dataset import EngVieDataset
from utils import translate
from utils import create_vocab, clean_data
if __name__ == '__main__':
    path = 'vie.txt'
    eng_data, vie_data = clean_data(path)
    create_vocab('eng_vocab.txt', eng_data)
    create_vocab('vie_vocab.txt', vie_data)
    eng_vocab = open('eng_vocab.txt', 'r').read().splitlines()
    vie_vocab = open('vie_vocab.txt', 'r', encoding="UTF-8").read().splitlines()
    eng_token2input = {t: i for i, t in enumerate(eng_vocab)}
    vie_token2input = {t: i for i, t in enumerate(vie_vocab)}
    vie_input2token = {v: k for k, v in vie_token2input.items()}
    device = 'cuda'
    transformer = Transformer(
        src_vocab_size=len(eng_vocab),
        trg_vocab_size=len(vie_vocab),
        src_max_length=25,
        trg_max_length=25,
        device=device
    )
    dataset = EngVieDataset(path, eng_vocab, vie_vocab)
    trainer = Trainer(transformer, dataset, device, load=False)
    # model = trainer.model
    # text = 'hello i am hien'
    # translate(model, text, eng_token2input, vie_input2token, 25, device)
    trainer.train()