import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, latent_dim=10, img_dim=28*28, hidden_dim=128):
        super().__init__()
        self.generator = nn.Sequential(
            self.block(latent_dim, hidden_dim),
            self.block(hidden_dim, hidden_dim * 2),
            self.block(hidden_dim * 2, hidden_dim * 4),
            self.block(hidden_dim * 4, hidden_dim * 8),
            nn.Linear(hidden_dim * 8, img_dim)
        )

    def block(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.generator(x)
    
    
class Discriminator(nn.Module):
    def __init__(self, img_dim=28*28, hidden_dim=128):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            self.block(img_dim, hidden_dim * 4),
            self.block(hidden_dim * 4, hidden_dim * 2),
            self.block(hidden_dim * 2, hidden_dim),
            nn.Linear(hidden_dim, 1)
        )

    def block(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(0.2)
        )

    def forward(self, x):
        return self.disc(x)

if __name__ == '__main__':
    # sample = torch.randn(size=(128, 10))
    gen = Generator().to('cuda')
    disc = Discriminator()
    # output = gen(sample)
    # print(output.shape)
    # output = disc(output)