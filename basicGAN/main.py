from matplotlib import image
import torch
import  torch.nn as nn
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms as T
from train import train_epoch
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from model import Generator, Discriminator
from utils import save_checkpoint
import sys
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 200
    batch_size = 128
    learning_rate = 1e-5
    z_dim = 64
    gen_checkpoint = 'checkpoint/gen.pth'
    disc_checkpoint = 'checkpoint/disc.pth'
    writer = SummaryWriter('runs/')

    generator = Generator(latent_dim=z_dim).to(device)
    discriminator = Discriminator().to(device)
    criterion = nn.BCEWithLogitsLoss()
    gen_opt = torch.optim.Adam(generator.parameters(), lr=learning_rate)
    disc_opt = torch.optim.Adam(discriminator.parameters(), lr=learning_rate)
    dataloader = DataLoader(
        MNIST('.', train=True, transform=T.Compose([T.ToTensor()]), download=True),
        batch_size=batch_size,
        shuffle=True
    )

    for epoch in range(num_epochs):
        gen_loss, disc_loss = train_epoch(
            generator,
            discriminator,
            dataloader,
            gen_opt,
            disc_opt,
            criterion,
            device,
            z_dim
        )
        print(f'{epoch+1} gen loss: {gen_loss}, disc loss: {disc_loss}')

        if (epoch+1) % 10 == 0:
            latent_vector = torch.randn(size=(25, z_dim)).to(device)
            generated_images = generator(latent_vector)
            generated_images = generated_images.reshape(-1, 1, 28, 28)
            img_grid = make_grid(generated_images, nrow=5)
            writer.add_image('generated_images', img_grid, epoch+1)
            save_checkpoint(generator, gen_opt, gen_checkpoint)
            save_checkpoint(discriminator, disc_opt, disc_checkpoint)
if __name__ == "__main__":
    main()
