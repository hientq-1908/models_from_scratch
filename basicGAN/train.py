import torch
import numpy as np
from tqdm import tqdm

def get_disc_loss(gen, disc, criterion, real_images, z_dim, device):
    batch_size = real_images.shape[0]
    real_images = real_images.reshape(batch_size, -1)
    # create latent vector and generate fake images
    latent_vector = torch.randn(size=(batch_size, z_dim), device=device)
    fake_images = gen(latent_vector).detach()
    # get prediction on fake images
    fake_preds = disc(fake_images)
    fake_labels = torch.zeros(batch_size,).to(device).unsqueeze(1)
    # prediction on real images
    real_preds = disc(real_images)
    real_labels = torch.ones_like(fake_labels)
    # disc loss
    real_loss = criterion(real_preds, real_labels)
    fake_loss = criterion(fake_preds, fake_labels)
    disc_loss = real_loss + fake_loss
    disc_loss /= 2
    return disc_loss

def get_gen_loss(gen, disc, criterion, real_images, z_dim, device):
    batch_size = real_images.shape[0]
    # create latent vector and generate fake images
    latent_vector = torch.randn(size=(batch_size, z_dim), device=device)
    fake_images = gen(latent_vector)
    # prediction on fake images
    fake_preds = disc(fake_images)
    target = torch.ones(size=(batch_size,), device=device).unsqueeze(1)
    # gen loss
    gen_loss = criterion(fake_preds, target)
    return gen_loss

def train_epoch(
    gen,
    disc,
    dataloader,
    gen_opt,
    disc_opt,
    criterion,
    device,
    z_dim
):
    arr_gen = list()
    arr_disc = list()
    loop = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, (real_images, _) in loop:
        real_images = real_images.to(device)
        # train discriminator
        disc_opt.zero_grad()
        disc_loss = get_disc_loss(
            gen, disc, criterion, real_images, z_dim, device
        )
        arr_disc.append(disc_loss.item())
        disc_loss.backward(retain_graph=True)
        disc_opt.step()

        # train generator
        gen_opt.zero_grad()
        gen_loss = get_gen_loss(
            gen, disc, criterion, real_images, z_dim ,device
        )
        arr_gen.append(gen_loss.item())
        gen_loss.backward()
        gen_opt.step()
    
    return np.mean(arr_gen), np.mean(arr_disc)
        
