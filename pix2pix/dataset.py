from torch.utils.data import Dataset
import os
from torchvision import transforms as tf
from PIL import Image

class ShoeDataset(Dataset):
    def __init__(self, img_dir, mode='train') -> None:
        super().__init__()
        assert mode in ['train', 'val']
        self.images = []
        for _, _, filename in os.walk(img_dir):
            self.images += filename
        self.mode = mode
        self.images = [os.path.join(img_dir, image) for image in self.images]
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = self.images[index]
        image = Image.open(image)
        if self.mode == 'train':
            image = self.get_train_transform()(image)
            c, h, w = image.shape
            source = image[:, :, :int(w/2)]
            target = image[:, :, int(w/2):]
            return source, target
            
    def get_train_transform(self):
        return tf.Compose([
            tf.Resize((256, 512)),
            tf.ToTensor(),
            tf.Normalize(mean=[0.5], std=[0.5])
        ])