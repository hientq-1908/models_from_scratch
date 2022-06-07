import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os
import numpy as np
from torchvision import transforms as tf

class RegionDataset(Dataset):
    """
    Return the data to feed  to the network
    Return:
        region proposals (2D tensor): the region cut out of image based on the box
        label (scalar): the label corresponding to that region
        cooor_offset (tuple): offset coordinate
    """
    def __init__(self, img_dir, image_ids, rois, gtlabels, coord_offsets):
        super(RegionDataset, self).__init__()
        self.img_dir = img_dir
        self.image_ids = image_ids
        self.ROIs = rois
        self.gtlabels = gtlabels
        self.coord_offsets= coord_offsets

        self.crops = [self.crop_region(image_id, ROI) for image_id, ROI in  zip(self.image_ids, self.ROIs)]

        self.str2num = {
            'Background': 0,
            'Truck': 1,
            'Bus': 2
        }
        self.transform = self.get_transform()

    def __len__(self):
        return len(self.coord_offsets)

    def __getitem__(self, index):
        crop = self.crops[index]
        crop = Image.fromarray(crop)
        label = self.gtlabels[index]
        # label string to number
        label = self.str2num[label]
        offset = self.coord_offsets[index]
        crop = self.transform(crop)

        return (
            crop,
            torch.tensor(label),
            torch.tensor(offset)
        )

    def crop_region(self, image_id, ROI):
        x, y, X, Y = ROI
        image_path = os.path.join(self.img_dir, image_id+'.jpg')
        image = Image.open(image_path)
        image = np.asarray(image)
        return image[y:Y, x:X]

    def get_transform(self):
        return tf.Compose([
            tf.Resize((128, 128)),
            tf.ToTensor(),
            tf.Normalize(mean=[0.485, 0.456, 0.406], \
                std=[0.229, 0.224, 0.225])
        ])