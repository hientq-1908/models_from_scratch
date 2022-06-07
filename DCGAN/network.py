import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, image_channels, feature_dim=64):
        super().__init__()
        self.discriminator = nn.Sequential(
            self._conv_block(image_channels, feature_dim),
            self._conv_block(feature_dim, feature_dim*2),
            self._conv_block(feature_dim*2, feature_dim*4),
            self._conv_block(feature_dim*4, feature_dim*8),
            nn.Conv2d(feature_dim*8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )
        # self.discriminator.apply(weights_init)
        
    def _conv_block(self, in_channels, out_channels, batch_norm=True):
        if batch_norm:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2)
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.discriminator(x)

class Generator(nn.Module):
    def __init__(self, img_channels, z_dim, feature_dim):
        super().__init__()
        self.generator = nn.Sequential(
            self._conv_transpose(z_dim, feature_dim*8, first=True),
            self._conv_transpose(feature_dim*8, feature_dim*4),
            self._conv_transpose(feature_dim*4, feature_dim*2),
            self._conv_transpose(feature_dim*2, feature_dim),
            nn.ConvTranspose2d(feature_dim, img_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )
        # self.generator(weights_init)
        
    def forward(self, x):
        return self.generator(x)

    def _conv_transpose(self, in_channels, out_channels, first=False):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size=4,
                stride=1 if first else 2,
                padding=0 if first else 1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

def weights_init(m):
    class_name = m.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif class_name.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def test():
    generator = Generator(img_channels=3, z_dim=100, feature_dim=64)
    discriminator = Discriminator(image_channels=3, feature_dim=64)
    sample = torch.randn(size=(1,3,64,64))
    assert tuple(discriminator(sample).shape) == (1,1,1,1)
    sample = torch.randn(size=(1,100,1,1))
    assert tuple(generator(sample).shape) == (1,3,64,64)
    print('all good')
if __name__ == '__main__':
    test()