import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    """
    A convolutional block consists of:
    conv -> norm -> activation function
    """
    def __init__(self, in_channels, out_channels, norm_layer=nn.InstanceNorm2d, act_func=nn.LeakyReLU(0.2), **kwargs):
        super().__init__()
        layer_list = list()
        layer_list.append(
            nn.Conv2d(in_channels, out_channels, **kwargs)
        )
        if norm_layer is not None:
            layer_list.append(
                norm_layer(out_channels)
            )
        layer_list.append(
            act_func
        )
        self.conv_block = nn.Sequential(*layer_list)
    
    def forward(self, x):
        return self.conv_block(x)

class Discriminator(nn.Module):
    def __init__(self, img_channels, num_features=[64, 128, 256, 512]):
        super().__init__()
        self.layer_list = list()
        self.layer_list.append(
            ConvBlock(img_channels, num_features[0], norm_layer=None, kernel_size=4, stride=2, padding=1, padding_mode='reflect')
        )
        in_channels = num_features[0]
        for feature in num_features[1:]:
            self.layer_list.append(ConvBlock(
                in_channels, 
                feature, 
                kernel_size=4, 
                stride=1 if feature==num_features[-1] else 2, 
                padding=1, 
                padding_mode='reflect'
            ))
            in_channels = feature
        
        self.layer_list.append(nn.Conv2d(num_features[-1], 1, kernel_size=4, stride=1, padding=1, padding_mode='reflect'))
        self.layer_list.append(nn.Sigmoid())

        self.disc = nn.Sequential(*self.layer_list)
        
    def forward(self, x):
        return self.disc(x)

def test():
    img_channels = 3
    img_size = 256
    sample = torch.rand((1, img_channels, img_size, img_size)).to('cuda')
    disc = Discriminator(img_channels=img_channels).to('cuda')
    out = disc(sample)
    assert out.shape == (1, 1, 30, 30)
    print('all good')

if __name__ == "__main__":
    test()