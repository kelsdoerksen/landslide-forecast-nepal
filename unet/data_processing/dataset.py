"""
Landslide Segementation dataset module
"""

from torch.utils.data import Dataset
from torchvision import transforms
from numpy import load, sort
import os

class LandslideDataset(Dataset):
    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.image_fns = os.listdir(image_dir)
        self.label_fns = os.listdir(label_dir)

    def __len__(self):
        return len(self.image_fns)

    def __getitem__(self, index):
        image_fns = sort(self.image_fns)
        label_fns = sort(self.label_fns)
        image_fn = image_fns[index]
        label_fn = label_fns[index]
        image_fp = os.path.join(self.image_dir, image_fn)
        label_fp = os.path.join(self.label_dir, label_fn)
        multichannel_image = load('{}'.format(image_fp)).astype('double')
        label_class = load('{}'.format(label_fp))
        multichannel_image = self.transform(multichannel_image)
        label_class = self.transform(label_class)
        return multichannel_image.float(), label_class.float()

    def transform(self, image):
        transform_ops = transforms.ToTensor()
        return transform_ops(image)