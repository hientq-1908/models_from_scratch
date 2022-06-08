from turtle import forward
import torch
import torch.nn as nn
from torchvision import models
from torchvision.ops import RoIPool
from sys import exit
class FastRCNN(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.backbone = models.vgg16(pretrained=True)
        self.backbone = nn.Sequential(*list(self.backbone.features.children())[:-1])
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.roi_pool = RoIPool(7, 14/224).to('cuda')
        feature_dim = 512*7*7
        self.head_classifier = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.Dropout(0.4),
            nn.Linear(512, n_classes)
        )
        self.head_localizer = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.Dropout(0.4),
            nn.Linear(512, 4),
            nn.Tanh()
        )
    
    def forward(self, images, rois):
        batch_size = images.shape[0]
        x = self.backbone(images)
        # rois = torch.cat([idxs, rois*224], dim=-1)
        rois = [rois[i,:] for i in range(len(rois))]
        rois = [roi.unsqueeze(0).float() for roi in rois]
        x = self.roi_pool(x, rois)
        x = x.reshape(batch_size, -1)
        class_ = self.head_classifier(x)
        bbox = self.head_localizer(x)
        return class_, bbox