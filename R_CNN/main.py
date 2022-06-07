from dataset import RegionDataset
from utils import extract_regions, get_iou
from utils import data_processing
import pandas as pd
from torch.utils.data import DataLoader
import torch
from network import RCNN
from trainer import Trainer
def main():
    csv_path = 'open-images-bus-trucks\df.csv'
    img_dir = 'open-images-bus-trucks\images\images'
    image_ids, rois, labels, offsets = data_processing(
        csv_path,
        img_dir
    )
    dataset = RegionDataset(
        img_dir,
        image_ids,
        rois,
        labels,
        offsets
    )
    
    trainer = Trainer(dataset, load=True)
    dataloader = DataLoader(dataset, 64, shuffle=False, drop_last=True)
    images, labels, offsets = next(iter(dataloader))
    print(labels)
    model = trainer.model
    images = images.to('cuda')
    probs, _ = model(images)
    probs = torch.nn.functional.softmax(probs, dim=-1)
    labels = torch.argmax(probs, -1)
    print(labels)
if __name__ == "__main__":
    main()
