from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from network import ViT
import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms as tf
from tqdm import tqdm

class Trainer:
    def __init__(self) -> None:
        self.num_epochs = 10
        self.learning_rate = 1e-4
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_size = 24

    def train_one_epoch(self, model, optimizer, criterion, train_loader, test_loader, device):
        train_losses = []
        train_acc = []
        test_acc = []
        model.train()
        for images, labels in tqdm(train_loader, total=(len(train_loader))):
            images, labels = images.to(device), labels.to(device)
            preds = model(images)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # train info
            train_losses.append(loss.item())
            probs = torch.softmax(preds, dim=-1)
            pred_labels = torch.argmax(probs, dim=-1)
            accuracy = (pred_labels==labels).sum()
            train_acc.append(accuracy.cpu().numpy()/len(labels))
        
        model.eval()
        for images, labels in tqdm(test_loader, total=len(test_loader)):
            images, labels = images.to(device), labels.to(device)
            preds = model(images)
            probs = torch.softmax(preds, dim=-1)
            pred_labels = torch.argmax(probs, dim=-1)
            accuracy = (pred_labels==labels).sum()
            test_acc.append(accuracy.cpu().numpy()/len(labels))
        
        return {
            'train_loss': np.mean(train_losses),
            'train_acc': np.mean(train_acc),
            'test_acc': np.mean(test_acc)
        }
            
    def train(self):
        train_loader = DataLoader(
            MNIST('.', train=True, download=True, transform=tf.ToTensor()),
            batch_size=self.batch_size,
            shuffle=True,
        )
        test_loader = DataLoader(
            MNIST(',', train=False, download=True, transform=tf.ToTensor()),
            batch_size=self.batch_size,
        )
        model = ViT(
            img_channels=1,
            embed_size=128,
            n_grid=4,
            img_size=28,
            num_classes=10,
            num_heads=4,
            num_features=128*4,
            num_layers=4,
            dropout=0.2,
            device=self.device
        ).to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        for epoch in range(self.num_epochs):
            res = self.train_one_epoch(
                model,
                optimizer,
                criterion,
                train_loader,
                test_loader,
                self.device
            )
            print(f'epoch {epoch+1} train_acc {res["train_acc"]}, test acc {res["test_acc"]}')