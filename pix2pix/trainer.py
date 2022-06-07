import torch
import torch.nn as nn
from generator import UNetGenerator
from discriminator import Discriminator
from dataset import ShoeDataset
import numpy as np
from torch.utils.data import DataLoader
import os
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

class Trainer():
    def __init__(self, load=False) -> None:
        self.num_epochs = 6
        self.learning_rate = 1e-4
        self.c_lambda = 100
        self.img_train = None
        self.img_val = None
        self.batch_size = 1
        self.path_gen = 'checkpoint/gen.pth'
        self.path_disc = 'checkpoint/disc.pth'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator = UNetGenerator().to(self.device)
        self.discriminator = Discriminator().to(self.device) 
        self.opt_gen = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.learning_rate,
            betas=(0.5, 0.999)
        )
        self.opt_disc = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=self.learning_rate,
            betas=(0.5, 0.999)
        )
        self.gan_criterion = nn.MSELoss()
        self.pixel_criterion = nn.L1Loss()
        
        if not os.path.exists('checkpoint'):
            os.makedirs('checkpoint')

        if load:
            self.load_checkpoint(
                self.generator, self.opt_gen, self.path_gen
            )
            self.load_checkpoint(
                self.discriminator, self.opt_disc, self.path_disc
            )

    def train_one_epoch(
        self,
        gen,
        disc,
        opt_gen,
        opt_disc,
        gan_criterion,
        pixel_criterion,
        device,
        c_lambda,
        dataloader
    ):
        gen_losses = []
        disc_losses = []
        loop = tqdm(dataloader, total=len(dataloader), leave=True)
        for (source, real_target) in loop:
            source, real_target = source.to(device), real_target.to(device)
            batch_size = source.shape[0]
            fake_target = gen(source)
            # train discriminator
            #####################
            opt_disc.zero_grad()
            real_preds = disc(real_target, source)
            fake_preds = disc(fake_target.detach(), source)
            # loss
            real_loss = gan_criterion(
                real_preds,
                torch.ones((batch_size, 1, 16,16), device=device) # patchGAN
            )
            fake_loss = gan_criterion(
                fake_preds,
                torch.zeros((batch_size, 1, 16, 16), device=device)
            )
            disc_loss = (real_loss + fake_loss) /2
            disc_losses.append(disc_loss.item())
            # backward and update
            disc_loss.backward()
            opt_disc.step()
            # train generator
            ########################
            opt_gen.zero_grad()
            # loss
            fake_preds = disc(fake_target, source)
            gan_loss = gan_criterion(
                fake_preds,
                torch.ones((batch_size, 1, 16, 16), device=device)
            )
            pixel_loss = pixel_criterion(
                fake_target,
                real_target
            )
            gen_loss = gan_loss + c_lambda * pixel_loss
            gen_losses.append(gen_loss.item())
            # backward and update
            gen_loss.backward()
            opt_gen.step()
        
        return np.mean(gen_losses), np.mean(disc_losses)
    
    def train(self):
        train_set = ShoeDataset(self.img_train, mode='train')
        train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=8,
            drop_last=True
        )
        for epoch in range(self.num_epochs):
            gen_loss, disc_loss = self.train_one_epoch(
                self.generator,
                self.discriminator,
                self.opt_gen,
                self.opt_disc,
                self.gan_criterion,
                self.pixel_criterion,
                self.device,
                self.c_lambda,
                train_loader
            )
            print(f'Epoch [{epoch+1}/{self.num_epochs}] gen loss: {gen_loss:.4f} disc loss {disc_loss:.4f}')

            if (epoch+1) % 1 == 0:
                # save checkpoints
                self.save_checkpoint(
                    self.generator, self.opt_gen, self.path_gen
                )
                self.save_checkpoint(
                    self.discriminator, self.opt_disc, self.path_disc
                )

    
    def save_checkpoint(self, model, optimizer, path):
        dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(dict, path)
        print('saved successfully')
    
    def load_checkpoint(self, model, optimizer, path):
        dict = torch.load(path)
        model.load_state_dict(dict['model'])
        optimizer.load_state_dict(dict['optimizer'])
        print('loaded successfully')

    def eval(self, img_val):
        if img_val:
            self.img_val = img_val
        assert self.img_val is not None
        writer = SummaryWriter('runs/eval')
        val_set = ShoeDataset(self.img_val, mode='train')
        val_loader = DataLoader(
            val_set,
            batch_size=1,
            shuffle=False
        )
        for i, (source, target) in enumerate(val_loader):
            source = source.to(self.device)
            target = target.to(self.device)
            pred = self.generator(source)
            stacked_imags = torch.cat([source, pred.detach(), target], dim=0)
            grid = make_grid(stacked_imags, normalize=True)
            writer.add_image('validate', grid, i)
        
        writer.close()
