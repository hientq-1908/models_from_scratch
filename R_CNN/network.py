import torch
import torch.nn as nn
from torchvision.models import vgg16

class RCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = vgg16(pretrained=True)
        self.feature_dim = self.vgg.classifier[0].in_features
        self.vgg.classifier = nn.Sequential() # drop original classifer
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.classifier = nn.Linear(self.feature_dim, 3)
        self.localizer = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 4),
            nn.Tanh(),
        )
    
    def forward(self, x):
        x = self.vgg(x)
        out_classifier = self.classifier(x)
        out_localizer = self.localizer(x)
        return out_classifier, out_localizer