import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, img_channels=3) -> None:
        super().__init__()
        self.disc = nn.ModuleList()
        self.disc += [
            self._conv_block(img_channels*2, 64, norm=False), # concat with target
            self._conv_block(64, 128),
            self._conv_block(128, 256),
            self._conv_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        ]
    
    def forward(self, x, target):
        x = torch.cat((x, target), 1)
        for _model in self.disc:
            x = _model(x)
        return x
    
    def _conv_block(self, in_channels, out_channels, norm=True):
        layers = []
        layers += [nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1
        )]
        if norm:
            layers += [nn.InstanceNorm2d(out_channels)]
        layers += [nn.LeakyReLU(0.2)]
        return nn.Sequential(*layers)

def test():
    x = torch.randn((1, 3, 256, 256))
    target = torch.randn((1, 3, 256, 256))
    disc = Discriminator()
    out = disc(x, target)
    assert tuple(out.shape) == (1, 1, 16, 16)
    print('all good')


if __name__ == "__main__":
    test()