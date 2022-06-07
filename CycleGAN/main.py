import sys
import torch
import torch.nn as nn
from generator import Generator
from discriminator import Discriminator
from dataset import CustomDataset
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import save_checkpoint
from tqdm import tqdm
#hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
img_channels = 3
img_size = 224
batch_size = 2
num_features_gen = 64
num_features_disc = [64, 128, 256, 512]
lamda_cycle_loss = 1
learning_rate = 1e-5
num_epochs = 100
checkpoint_dir = 'checkpoint/'
checkpoint_gen_X = checkpoint_dir + 'genX.pth'
checkpoint_gen_Y = checkpoint_dir + 'genY.pth'
checkpoint_disc_X = checkpoint_dir + 'discX.pth'
checkpoint_disc_Y = checkpoint_dir + 'discY.pth' 

#train function
def train_epoch(X_loader, Y_loader, gen_X, gen_Y, disc_X, disc_Y, opt_gen, opt_disc):
    total_loss_gen = 0
    total_loss_disc = 0
    length = len(list(zip(X_loader, Y_loader)))
    gen_X = gen_X.to(device)
    gen_Y = gen_Y.to(device)
    disc_X = disc_X.to(device)
    disc_Y = disc_Y.to(device)
    for idx, (x, y) in tqdm(enumerate(zip(X_loader, Y_loader)), total=length):
    
        real_x = x.to(device)
        real_y = y.to(device)

        # train discriminator
        # loss for disc_X
        # generate fake x from y by gen_x
        fake_x = gen_X(real_y)
        # disc X distinguish between fake x and real x
        pred_fake = disc_X(fake_x.detach())
        pred_real = disc_X(real_x)
        # loss = fake_loss(labeled 0) + real_loss (labeled 1)
        loss_disc_X = nn.MSELoss()(pred_fake, torch.zeros_like(pred_fake)) + nn.MSELoss()(pred_real, torch.ones_like(pred_real))
        # similar with disc_Y
        fake_y = gen_Y(real_x)
        pred_fake = disc_Y(fake_y.detach())
        pred_real = disc_Y(real_y)
        loss_disc_Y = nn.MSELoss()(pred_fake, torch.zeros_like(pred_fake)) + nn.MSELoss()(pred_real, torch.ones_like(pred_real))
        
        # sum up
        loss_disc = (loss_disc_X + loss_disc_Y) * 0.5
        total_loss_disc += loss_disc

        opt_disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()
        # train generator
        # total loss = adversarial loss + cycle loss

        # adversarial loss
        # loss for gen_X, minimize the loss with fake_x to label 1
        pred_fake = disc_X(fake_x.detach())
        loss_gen_X = nn.MSELoss()(pred_fake, torch.ones_like(pred_fake))
        # floss for gen Y, minimizing the loss with fake_y to label 1
        pred_fake = disc_Y(fake_y.detach())
        loss_gen_Y = nn.MSELoss()(pred_fake, torch.ones_like(pred_fake))

        loss_gan = loss_gen_X + loss_gen_Y

        # cycle loss
        # cycle: X -> Y_fake -> X_restored and Y -> X_fake -> Y_restored
        restored_x = gen_X(fake_y)
        restored_y = gen_Y(fake_x)
        loss_cycle = nn.L1Loss()(real_x, restored_x) + nn.L1Loss()(real_y, restored_y)

        # sum up
        loss_gen = loss_gan + loss_cycle * lamda_cycle_loss
        total_loss_gen += loss_gen
        opt_gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

    avg_loss_gen = total_loss_gen / length
    avg_loss_disc = total_loss_disc / length

    return avg_loss_gen, avg_loss_disc

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    monetDataset = CustomDataset('monet_jpg', transform=transform)
    photoDataset = CustomDataset('photo_jpg', transform=transform)

    monet_loader = DataLoader(monetDataset, batch_size=batch_size, shuffle=False)
    photo_loader = DataLoader(photoDataset, batch_size=batch_size, shuffle=False)

    # model includes 2 mappings function between two domains X and Y
    # so that, we have 2 generators
    gen_X = Generator(img_channels=img_channels, num_features=num_features_gen) # generate x from input y
    gen_Y = Generator(img_channels=img_channels, num_features=num_features_gen)
    # corresponding to that, we have 2 adversarial discriminators to distinguish the outputs of the two gen models
    disc_X = Discriminator(img_channels=img_channels, num_features=num_features_disc)
    disc_Y = Discriminator(img_channels=img_channels, num_features=num_features_disc)

    opt_gen = torch.optim.Adam(
        params = list(gen_X.parameters()) + list(gen_Y.parameters()),
        lr = learning_rate,
        betas = (0.5, 0.99)
    )

    opt_disc = torch.optim.Adam(
        params = list(disc_X.parameters()) + list(disc_Y.parameters()),
        lr = learning_rate,
        betas = (0.5, 0.99)
    )

    #train
    for epoch in range(num_epochs):
        loss_gen, loss_disc = train_epoch(
            X_loader = photo_loader,
            Y_loader = monet_loader,
            gen_X = gen_X,
            gen_Y = gen_Y,
            disc_X = disc_X,
            disc_Y = disc_Y,
            opt_gen = opt_gen,
            opt_disc = opt_disc
        )

        print(f'Epoch [{epoch}/{num_epochs}]')
        print(f'avg gen loss: {loss_gen}.4f, avg disc loss: {loss_disc}.4f')

        save_checkpoint(gen_X, opt_gen, checkpoint_gen_X)
        save_checkpoint(gen_Y, opt_gen, checkpoint_gen_Y)
        save_checkpoint(disc_X, opt_disc, checkpoint_disc_X)
        save_checkpoint(disc_X, opt_disc, checkpoint_disc_Y)
