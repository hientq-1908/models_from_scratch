import torch
import torch.nn as nn

class UNetGenerator(nn.Module):
    def __init__(self, in_channels = 3, img_channels=3):
        super().__init__()
        # down sampling
        self.down_block_1 = self.down_block(in_channels, 64, norm=False)
        self.down_block_2 = self.down_block(64, 128)
        self.down_block_3 = self.down_block(128, 256)
        self.down_block_4 = self.down_block(256, 512)
        self.down_block_5 = self.down_block(512, 512, dropout=0.5)
        self.down_block_6 = self.down_block(512, 512, dropout=0.5)
        self.down_block_7 = self.down_block(512, 512, dropout=0.5)
        self.down_block_8 = self.down_block(512, 512, norm=False, dropout=0.5)
        # up sampling
        self.up_block_1 = self.up_block(512, 512, dropout=0.5)
        self.up_block_2 = self.up_block(1024, 512, dropout=0.5)
        self.up_block_3 = self.up_block(1024, 512, dropout=0.5)
        self.up_block_4 = self.up_block(1024, 512, dropout=0.5)
        self.up_block_5 = self.up_block(1024, 256)
        self.up_block_6 = self.up_block(512, 128)
        self.up_block_7 = self.up_block(256, 64)
        # head
        self.head = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, img_channels, kernel_size=4, padding=1),
            nn.Tanh()
        )
    def forward(self, x):
        batch_size = x.shape[0]
        # down sampling 
        d1 = self.down_block_1(x)
        d2 = self.down_block_2(d1)
        d3 = self.down_block_3(d2)
        d4 = self.down_block_4(d3)
        d5 = self.down_block_5(d4)
        d6 = self.down_block_6(d5)
        d7 = self.down_block_7(d6)
        d8 = self.down_block_8(d7)
        assert tuple(d8.shape) == (batch_size, 512, 1, 1)
        out = d8
        u1 = self.up_block_1(out)
        out = torch.cat((u1, d7), 1)
        u2 = self.up_block_2(out)
        out = torch.cat((u2, d6), 1)
        u3 = self.up_block_3(out)
        out = torch.cat((u3, d5), 1)
        u4 = self.up_block_4(out)
        out = torch.cat((u4, d4), 1)
        u5 = self.up_block_5(out)
        out = torch.cat((u5, d3), 1)
        u6 = self.up_block_6(out)
        out = torch.cat((u6, d2), 1)
        u7 = self.up_block_7(out)
        out = torch.cat((u7, d1), 1)
        # head
        out = self.head(out)
        return out

    def down_block(self, in_channels, out_channels, norm=True, dropout=None):
        layers = []
        layers += [nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )]
        if norm:
            layers += [nn.InstanceNorm2d(out_channels)]
        if dropout is not None:
            layers += [nn.Dropout(dropout)]
        layers += [nn.LeakyReLU(0.2)]
        return nn.Sequential(*layers)

    def up_block(self, in_channels, out_channels, norm=True, dropout=None):
        layers = []
        layers += [nn.ConvTranspose2d(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=False
        )]
        if norm:
            layers += [nn.InstanceNorm2d(out_channels)]
        if dropout is not None:
            layers += [nn.Dropout(dropout)]
        layers += [nn.LeakyReLU(0.2)]
        return nn.Sequential(*layers)