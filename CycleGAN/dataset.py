import os
from cv2 import transform
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, img_dir, transform=None) -> None:
        super().__init__()
        self.images = os.listdir(img_dir)
        self.img_paths = [os.path.join(img_dir, image) for image in self.images]
        self.transform = transform
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])
        if transform:
            image = self.transform(image)
        
        return image
