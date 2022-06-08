import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms as tf
import numpy as np
import os
from sys import exit
class RegionDataset(Dataset):
    """
    Return the data to feed  to the network
    Return:
        Image (2D tensor)
        region proposals (2D tensor): the region cut out of image based on the box
        label (scalar): the label corresponding to that region
        cooor_offset (tuple): offset coordinate
    """
    def __init__(self, img_dir, image_ids, rois, gtlabels, coord_offsets):
        super(RegionDataset, self).__init__()
        self.img_dir = img_dir
        self.image_ids = image_ids
        self.ROIs = rois # relative to images
        self.gtlabels = gtlabels
        self.coord_offsets= coord_offsets


        self.str2num = {
            'Background': 0,
            'Truck': 1,
            'Bus': 2
        }
        self.transform = self.get_transform()

    def __len__(self):
        return len(self.coord_offsets)

    def __getitem__(self, index):
        image_id = self.image_ids[index]
        image_path = os.path.join(self.img_dir, image_id +'.jpg')
        image = Image.open(image_path)
        label = self.gtlabels[index]
        roi = self.ROIs[index]
        # label string to number
        label = self.str2num[label]
        offset = self.coord_offsets[index]
        image = self.transform(image)

        return (
            image, 
            torch.tensor(roi),
            torch.tensor(label),
            torch.tensor(offset)
        )


    def get_transform(self):
        return tf.Compose([
            tf.Resize((224, 224)),
            tf.ToTensor(),
            tf.Normalize(mean=[0.485, 0.456, 0.406], \
                std=[0.229, 0.224, 0.225])
        ])
    
    # def collate_fn(self, batch):
    #     images, rois, labels, offsets = zip(*batch)
    #     batch_size = len(batch)
    #     images = [image.unsqueeze(0) for image in images]
    #     images = torch.cat(images, dim=0)
    #     rois = torch.from_numpy(np.asarray(rois))
    #     labels = torch.from_numpy(np.asarray(labels))
    #     offsets = torch.from_numpy(np.asarray(offsets))
    #     idxs = torch.arange(0, batch_size).unsqueeze(1)
    #     return images, rois, idxs, labels, offsets