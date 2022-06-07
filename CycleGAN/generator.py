import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """
    A residual block consists of:
    conv -> norm -> act func -> conv
    """
    def __init__(self, channels: int, norm_layer=nn.InstanceNorm2d, act_func=nn.ReLU):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1),
            norm_layer(channels),
            act_func(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        )
        self.norm_layer = norm_layer(channels)
        self.act_func = act_func()
    
    def forward(self, x):
        return self.act_func(self.norm_layer(x + self.conv_block(x)))

class ConvBlock(nn.Module):
    """
    A convolutional block consists of:
    conv/convtrans layer -> norm -> activation function
    """
    def __init__(self, in_channels: int, out_channels: int, down_sampling: bool, norm_layer=nn.InstanceNorm2d, act_func=nn.ReLU, **kwargs):
        super().__init__()
        layer_list = list()
        if down_sampling:
            layer_list.append(
                nn.Conv2d(in_channels, out_channels, **kwargs)
            )
        else:
            layer_list.append(
                nn.ConvTranspose2d(in_channels, out_channels, **kwargs)
            )
        layer_list.extend([
            norm_layer(out_channels),
            act_func()
        ])
        self.conv_block = nn.Sequential(*layer_list)
    
    def forward(self, x):
        return self.conv_block(x)

class Generator(nn.Module):
    def __init__(
        self,
        img_channels=3,
        num_features=64
    ):
        super().__init__()
        layer_list = list()

        #down sampling
        layer_list.extend([
            nn.ReflectionPad2d(3),
            ConvBlock(img_channels, num_features, down_sampling=True, kernel_size=7, stride=1, padding=0),
            ConvBlock(num_features, 2*num_features, down_sampling=True, kernel_size=3, stride=1, padding=1),
            ConvBlock(2*num_features, 4*num_features, down_sampling=True, kernel_size=3, stride=1, padding=1)
        ])

        #residual block
        layer_list.append(ResidualBlock(channels=4*num_features))

        #up sampling
        layer_list.extend([
            ConvBlock(4*num_features, 2*num_features, down_sampling=False, kernel_size=3, stride=1, padding=1),
            ConvBlock(2*num_features, num_features, down_sampling=False, kernel_size=3, stride=1, padding=1),
            nn.ReflectionPad2d(3),
            nn.Conv2d(num_features, img_channels, kernel_size=7, stride=1, padding=0),
            nn.Tanh()
        ])

        self.generator =  nn.Sequential(*layer_list)
    
    def forward(self, x):
        return self.generator(x)

def test():
    img_channels = 3
    img_size = 256
    sample =  torch.rand((1, img_channels, img_size, img_size)).to('cuda')
    gen = Generator().to('cuda')
    out = gen(sample)
    assert out.shape == (1, img_channels, img_size, img_size)
    print("all good!")
if __name__ == "__main__":
    test()
