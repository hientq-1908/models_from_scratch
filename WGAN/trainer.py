import torch
import torch.nn as nn
from model import Generator, Critic
from tqdm import tqdm
import numpy as np
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms as T
import sys

class Trainer():
    def __init__(self, load_model=False):
        self.generator = Generator()
        self.critic = Critic()
        self.num_epochs = 200
        self.batch_size = 128
        self.z_dim = 64
        self.learning_rate = 1e-5
        self.c_lambda = 10
        self.gen_checkpoint = 'checkpoint/gen.pth'
        self.critic_checkpoint = 'checkpoint/critic.pth'
        self.device = torch.device('cuda' if torch.cuda.is_available()else 'cpu')
        self.crit_repeats = 5
        self.gen_opt = torch.optim.Adam(self.generator.parameters(), lr=self.learning_rate)
        self.crit_opt = torch.optim.Adam(self.critic.parameters(), lr=self.learning_rate)
        self.dataloader = DataLoader(
            MNIST('.', train=True, transform=T.Compose([T.ToTensor()]), download=True),
            batch_size=self.batch_size,
            shuffle=True
        )
        if load_model:
            self.load_checkpoint(
                self.generator, self.gen_opt, self.gen_checkpoint
            )
            self.load_checkpoint(
                self.critic, self.crit_opt, self.critic_checkpoint
            )

    def get_gradient(self, critic, real_images, fake_images, epsilon):
        mixed_images = real_images * epsilon + fake_images * (1 - epsilon)
        mix_scores = critic(mixed_images)
        gradient = torch.autograd.grad(
            inputs=mixed_images,
            outputs=mix_scores,
            grad_outputs=torch.ones_like(mix_scores),
            create_graph=True,
            retain_graph=True 
        )[0]
        return gradient
    
    def gradient_penalty(self, gradient):
        gradient = gradient.reshape(len(gradient), -1)
        gradient_norm = gradient.norm(2, dim=-1)
        penalty = torch.mean(torch.square(gradient_norm - 1))
        return penalty
    
    def get_gen_loss(self, fake_scores):
        # calculate loss for gen
        # loss is the negative of the mean of the critic's scores
        gen_loss = - torch.mean(fake_scores)
        return gen_loss
    
    def get_crit_loss(self, real_scores, fake_scores, gp, c_lambda):
        # critic's loss
        critic_loss = torch.mean(real_scores) - torch.mean(fake_scores) + gp * c_lambda
        return critic_loss

    def train_one_epoch(self, dataloader, generator, critic, gen_opt, crit_opt, z_dim, crit_repeats, device, c_lambda):
        crit_losses = list()
        gen_losses = list()
        generator, critic = generator.to(device), critic.to(device)
        for real_images, _ in tqdm(dataloader, total=len(dataloader)):
            real_images = real_images.to(device)
            # train critic
            tmp_crit_losses = list()
            for _ in range(crit_repeats):
                crit_opt.zero_grad()
                batch_size = real_images.shape[0]
                latent_vector = torch.randn(size=(batch_size, z_dim), device=device)
                fake_images = generator(latent_vector)
                # scores on real and fake images
                real_scores = critic(real_images)
                fake_scores = critic(fake_images.detach())
                
                epsilon = torch.rand(len(real_images), 1, 1, 1, device=device, requires_grad=True)
                gradient = self.get_gradient(critic, real_images, fake_images.detach(), epsilon)
                grad_penalty = self.gradient_penalty(gradient)
                # crit's loss
                crit_loss = self.get_crit_loss(real_scores, fake_scores, grad_penalty, c_lambda)
                tmp_crit_losses.append(crit_loss.item())
                # backward and update
                crit_loss.backward(retain_graph=True)
                crit_opt.step()
            crit_losses.append(np.mean(tmp_crit_losses))

            # train generator
            gen_opt.zero_grad()
            latent_vector = torch.randn(size=(batch_size, z_dim), device=device)
            fake_images = generator(latent_vector)
            # scores on fake images
            fake_scores = critic(fake_images)
            # gen loss
            gen_loss = self.get_gen_loss(fake_scores)
            gen_losses.append(gen_loss.item())
            # backward and update
            gen_loss.backward()
            gen_opt.step()
        return np.mean(gen_losses), np.mean(crit_losses)
    
    def save_checkpoint(self, model, optimizer, path):
        checkpoint = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }
        torch.save(checkpoint, path)
        print('saved successfully')
    
    def load_checkpoint(self, model, optimizer, path):
        checkpoint = torch.load(path)
        model = model.load_state_dict(checkpoint['model'])
        model = optimizer.load_state_dict(checkpoint['optimizer'])
    
    def train(self):
        for epoch in range(self.num_epochs):
            gen_loss, crit_loss = self.train_one_epoch(
                self.dataloader, 
                self.generator, 
                self.critic, 
                self.gen_opt, 
                self.crit_opt, 
                self.z_dim, 
                self.crit_repeats, 
                self.device,
                self.c_lambda
            )
            print(f'Epoch [{epoch+1}/{self.num_epochs}] \n')
            print(f'gen loss : {gen_loss:.4f}, crit loss {crit_loss:.4f}')
        
            if epoch + 1 % 100 == 0:
                self.save_checkpoint(self.generator, self.gen_opt)
                self.save_checkpoint(self.critic, self.crit_opt)



