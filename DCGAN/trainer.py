from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms as tf
from tqdm import tqdm
import torch
import numpy as np
from network import Generator, Discriminator
import torch.nn as nn
import os
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
import sys
class Trainer():
    def __init__(self, img_dir, load=True) -> None:
        super().__init__()
        self.epochs = 50
        self.batch_size = 64
        self.learning_rate = 0.0002
        self.z_dim = 100
        self.feature_dim = 64
        self.img_channels = 3
        self.gen_path = 'checkpoint/gen.pth'
        self.disc_path = 'checkpoint/disc.pth'

        self.dataset = ImageFolder(
            root=img_dir,
            transform=tf.Compose([
                tf.Resize((64,64)),
                tf.ToTensor(),
                tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]),
        )
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=8,
            drop_last=True
        )
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.generator = Generator(self.img_channels, self.z_dim, self.feature_dim).to(self.device)
        self.discriminator = Discriminator(self.img_channels, self.feature_dim).to(self.device)

        self.criterion = nn.BCELoss()
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

        if not os.path.exists('checkpoint'):
            os.makedirs('checkpoint')

        if load:
            self.load_checkpoint(
                self.generator, self.opt_gen, self.gen_path
            )
            self.load_checkpoint(
                self.discriminator, self.opt_disc, self.disc_path
            )
    def train_one_epoch(
        self,
        generator,
        discriminator,
        opt_gen,
        opt_disc,
        criterion,
        dataloader,
        device,
        z_dim
    ):
        gen_losses = list()
        disc_losses = list()

        loop = tqdm(dataloader, total=len(dataloader), leave=False)
        for real_images, _ in loop:
            batch_size = real_images.shape[0]
            real_images = real_images.to(device)
            # train discriminator
            ####################
            noise = self.random_noise(batch_size, z_dim)
            fake_images = generator(noise).detach()
            # predictions on fake images and real images
            real_preds = discriminator(real_images)
            fake_preds = discriminator(fake_images)
            opt_disc.zero_grad()
            # loss
            real_loss = criterion(
                real_preds.squeeze(),
                torch.ones(batch_size, device=device)
            )
            fake_loss = criterion(
                fake_preds.squeeze(),
                torch.zeros(batch_size, device=device)
            )
            disc_loss = real_loss + fake_loss
            real_loss.backward()
            fake_loss.backward()
            opt_disc.step()
            # train generator
            #####################################
            opt_gen.zero_grad()
            noise = self.random_noise(batch_size, self.z_dim)
            fake_images = generator(noise)
            fake_preds = discriminator(fake_images)
            gen_loss = criterion(
                fake_preds.squeeze(),
                torch.ones(batch_size, device=device)
            )
            gen_loss.backward()
            opt_gen.step()

            gen_losses.append(gen_loss.item())
            disc_losses.append(disc_loss.item())
        return np.mean(gen_losses), np.mean(disc_losses)

    def train(self):
        writer = SummaryWriter('runs/')
        for epoch in range(self.epochs):
            gen_loss, disc_loss = self.train_one_epoch(
                self.generator,
                self.discriminator,
                self.opt_gen,
                self.opt_disc,
                self.criterion,
                self.dataloader,
                self.device,
                self.z_dim
            )

            print(f'Epoch [{epoch+1}/{self.epochs}], gen loss: {gen_loss:.4f} disc loss {disc_loss:.4f}')

            if (epoch+1) % 5 == 0:
                self.save_check_point(
                    self.generator,
                    self.opt_gen,
                    self.gen_path
                )
                self.save_check_point(
                    self.discriminator,
                    self.opt_disc,
                    self.disc_path
                )
                # output samples
                random_noise = self.random_noise(25, self.z_dim)
                fake_images = self.generator(random_noise)
                grid = make_grid(fake_images, nrow=5)
                writer.add_image('generated images', grid, epoch)
        writer.close()

    def random_noise(self, batch_size, z_dim):
        n = torch.randn(batch_size, z_dim, 1, 1, device=self.device)
        return n.to(self.device)
    
    def save_check_point(self, model, optimizer, path):
        state_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(state_dict, path)
        print('saved successfully')
    
    def load_checkpoint(self, model, optimizer, path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print('loaded successfully')
    
    def test(self):
        writer = SummaryWriter("runs/testing")
        random_noise = self.random_noise(25, self.z_dim)
        fake_images = self.generator(random_noise).detach()
        # unnormalize
        grid = make_grid(fake_images, nrow=5, normalize=True)
        writer.add_image('test images', grid)
        writer.close()
