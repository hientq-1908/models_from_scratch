from utils import data_process
from trainer import Trainer
from network import FastRCNN
from dataset import RegionDataset
import torch
if __name__ == "__main__":
    csv_path = 'open-images-bus-trucks\df.csv'
    img_dir = 'open-images-bus-trucks\images\images'
    image_ids, rois, labels, offsets = data_process(
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FastRCNN(n_classes=3)
    trainer = Trainer(
        model,
        dataset,
        device
    )
    trainer.load_checkpoint('checkpoint.pth')
    trainer.train()