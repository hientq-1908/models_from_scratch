import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=64, im_channels=1, hidden_dim=64):
        super().__init__()
        self.z_dim = z_dim
        self.generator = nn.Sequential(
            self.convtrans_block(z_dim, hidden_dim*4),
            self.convtrans_block(hidden_dim*4, hidden_dim*2, 4, 1),
            self.convtrans_block(hidden_dim*2, hidden_dim),
            self.convtrans_block(hidden_dim, im_channels, 4, final_layer=True)
        )

    def forward(self, x):
        x = x.view(len(x), self.z_dim, 1, 1)
        return self.generator(x)

    def convtrans_block(self, in_channels, out_channels, kernel_size=3, stride=2, final_layer=False):
        if final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride),
                nn.Tanh()
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            )
        
class Critic(nn.Module):
    def __init__(self, img_channels=1, hidden_dim=64):
        super().__init__()
        self.critic = nn.Sequential(
            self.conv_block(img_channels, hidden_dim),
            self.conv_block(hidden_dim, hidden_dim*2),
            self.conv_block(hidden_dim*2, 1, final_layer=True)
        )
    
    def forward(self, x):
        x = self.critic(x)
        return x.view(len(x), -1)

    def conv_block(self, in_channels, out_channels, kernel_size=3, stride=2, final_layer=False):
        if final_layer:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2)
            )