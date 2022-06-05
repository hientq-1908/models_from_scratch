import torch
import torch.nn as nn
from network import Transformer
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import sys
from torch.optim import lr_scheduler

from torch.utils.tensorboard import SummaryWriter

class Trainer():
    def __init__(self, model, dataset, device) -> None:
        self.device = device
        self.model = model
        self.batch_size = 64
        self.num_epochs = 100
        self.learning_rate = 2e-4
        self.dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        self.checkpoint = 'checkpoint.pth'
        self.writer = SummaryWriter('runs/')

    def train_one_epoch(self, epoch, model, dataloader, criterion, optimizer, device):
        model.train()
        losses = []
        loop = enumerate(tqdm(dataloader, total=len(dataloader)))
        for i, (src, trg) in loop:
            src, trg = src.to(device), trg.to(device)
            trg_input = trg[:, :-1]
            trg_out = trg[:, 1:].reshape(-1)
            logits = model(src, trg_input)
            logits = logits.reshape(-1, logits.shape[-1])
            loss = criterion(logits, trg_out)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            step = epoch * len(dataloader) + i
            self.writer.add_scalar('training loss', loss, step)

        return np.mean(losses)

    def train(self):
        model = self.model.to(self.device)
        for epoch in range(self.num_epochs):
            loss = self.train_one_epoch(
                epoch,
                model,
                self.dataloader,
                self.criterion,
                self.optimizer,
                self.device
            )
            print(f'epoch [{epoch+1}], loss {loss:.4f}')

            if (epoch+1) % 5 == 0:
                self._save_checkpoint(model, self.checkpoint)
                print(f'saved at epoch {epoch+1}')

    def _save_checkpoint(self, model, path):
        state_dict = {
            'model': model.state_dict()
        }
        torch.save(state_dict, path)
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
