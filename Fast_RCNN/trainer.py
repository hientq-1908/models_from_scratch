import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from network import FastRCNN
from tqdm import tqdm
import numpy as np
from utils import extract_regions, show_image
from torchvision.ops import nms
from torchvision import transforms as tf
from PIL import Image

class Trainer():
    def __init__(self, model, dataset=None, device=None):
        self.learning_rate = 2e-4
        self.num_epochs = 10
        self.batch_size = 8
        self.dataset = dataset
        self.model = model
        self.device = device
        self.criterion_class = nn.CrossEntropyLoss()
        self.criterion_bbox = nn.L1Loss()
        self.checkpoint = 'checkpoint.pth'
        self.cur_epoch = -1
        self.device = device if device else 'cpu'

    def _train_one_epoch(epoch, model, dataloader, criterion_class, criterion_bbox, optimizer, device):
        losses = []
        loop = enumerate(tqdm(dataloader, total=len(dataloader)))
        for i, (image, roi, label, offset) in loop:
            image = image.to(device)
            roi = roi.to(device)
            # ids = ids.to(device)
            label = label.to(device)
            offset = offset.to(device)
            pred_class, pred_bbox = model(image, roi)
            ###########
            loss_class = criterion_class(pred_class, label.long())
            loss_bbox = criterion_bbox(pred_bbox, offset.long())
            loss = loss_class + loss_bbox
            # backward and update
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += [loss.item()]
        
        return np.mean(losses)
    
    def train(self):
        print('####### training #########')
        model = self.model.to(self.device)
        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=True,
            # collate_fn=self.dataset.collate_fn
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        for epoch in range(self.num_epochs):
            self.cur_epoch = epoch
            loss = self._train_one_epoch(
                model,
                dataloader,
                self.criterion_class,
                self.criterion_bbox,
                optimizer,
                self.device
            )
            print('epoch {}, loss {:.4f}'.format(epoch+1, loss))

            if (epoch+1) % 5 == 0:
                self._save_checkpoint(self.checkpoint)

    def _save_checkpoint(self, path):
        state_dict = {
            'model': self.model.state_dict(),
            'epoch': self.cur_epoch
        }
        torch.save(state_dict, path)
        print('saved successfully')
    
    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        self.cur_epoch = checkpoint['epoch']
        print('loaded successfully')
    
 