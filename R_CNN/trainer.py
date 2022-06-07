from cProfile import label
import torch
import torch.nn as nn
import os
import numpy as np
from network import RCNN
from torch.utils.data import DataLoader
import sys
from tqdm import tqdm
from utils import extract_regions, show_image
from PIL import Image
from torchvision import transforms as tf
from torchvision.ops import nms

class Trainer():
    def __init__(self, dataset, load=False) -> None:
        self.num_epochs = 50
        self.learning_rate = 1e-5
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = RCNN().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.batch_size = 64
        self.c_lambda = 10
        self.label_criterion = nn.CrossEntropyLoss()
        self.offset_criterion = nn.L1Loss()
        self.checkpoint_path = 'checkpoint.pth'
        
        self.dataloader = None
        if dataset:
            self.dataloader = DataLoader(dataset, self.batch_size, shuffle=False, drop_last=True)

        if load:
            self.load_checkpoint()

    def train_one_epoch(self, dataloader, model, optimizer, device, label_criterion, offset_criteron, c_lambda):
        assert self.dataloader == None
        losses = list()
        model.train()
        loop = tqdm(dataloader, total=len(dataloader), leave=False)
        for image, labels, offsets in loop:
            optimizer.zero_grad()

            image = image.to(device)
            labels = labels.to(device)
            offsets = offsets.to(device)
            # forward pass
            label_preds, offset_preds = model(image)
            # classifier loss
            class_loss = label_criterion(label_preds, labels)
            # bbox loss
            # do not compute bbox loss if label is background
            idxs, = torch.where(labels != 0)
            _offsets = offsets[idxs]
            _offset_preds = offset_preds[idxs]
            if len(idxs) > 0:
                offset_loss = offset_criteron(_offset_preds, _offsets)
            else:
                offset_loss = 0
            
            # sum up
            loss = class_loss + c_lambda * offset_loss
            losses.append(loss.item())
            # backward and update
            loss.backward()
            optimizer.step()
            # accuracy
            probs = torch.nn.functional.softmax(label_preds, dim=-1)
            class_labels = torch.argmax(probs, dim=-1)
            print(labels)
            accuracy = (class_labels==labels).sum() / image.shape[0]
        return np.mean(losses), accuracy

    def train(self):
        for epoch in range(self.num_epochs):
            loss, accuracy = self.train_one_epoch(
                self.dataloader,
                self.model,
                self.optimizer,
                self.device,
                self.label_criterion,
                self.offset_criterion,
                self.c_lambda
            )
            print(f'epoch [{epoch+1}/{self.num_epochs}], loss: {loss:.4f}, accuracy: {accuracy:.4f}')

            if epoch + 1 % 5 == 0:
                self.save_check_point(self.model, self.optimizer, self.checkpoint_path)

    def save_check_point(self, model, optimizer, path):
        state_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state_dict, path)
        print('saved successfully')
    
    def load_checkpoint(self):
        checkpoint = torch.load(self.checkpoint_path)
        self.model.load_state_dict(checkpoint['model'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print('loaded successfully')

    def get_transform(self):
        return tf.Compose([
            tf.Resize((128, 128)),
            tf.ToTensor(),
            tf.Normalize(mean=[0.485, 0.456, 0.406], \
                std=[0.229, 0.224, 0.225])
        ])

    def predict(self, image):
        image = np.asarray(image)
        regions = extract_regions(image)
        regions = [(x,y,x+w,y+h) for (x,y,w,h) in regions]
        crops = []
        for region in regions:
            x, y, X, Y = region
            crop = image[y:Y, x:X,:]
            crop = Image.fromarray(crop)
            crop = self.get_transform()(crop)
            crops.append(crop)
        crops = torch.stack(crops, dim=0)
        crops = crops.to(self.device)
        probs, offsets = self.model(crops)
        probs = torch.nn.functional.softmax(probs, dim=-1)
        scores, labels = torch.max(probs, -1)
        idxs = labels != 0

        scores = scores.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
        probs = probs.detach().cpu().numpy()
        offsets = offsets.detach().cpu().numpy()
        regions = np.array(regions)
        idxs = idxs.cpu()

        scores = scores[idxs]
        labels = labels[idxs]
        probs = probs[idxs]
        offsets = offsets[idxs]
        regions = regions[idxs]
        bboxes = (regions + offsets).astype(np.uint16)
 
        idxs = nms(torch.tensor(bboxes.astype(np.float32)), torch.tensor(scores), 0.05)

        
        bboxes = bboxes[idxs]
        if len(idxs) == 1:
            bboxes = [bboxes]

        show_image(image, bboxes)