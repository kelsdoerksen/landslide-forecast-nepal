"""
Landslide Dataset Module
"""

from torch.utils.data import Dataset
from torchvision import transforms
from numpy import load, sort
import os
import torch
from datetime import date, timedelta


def daterange(date1, date2):
    date_list = []
    for n in range(int((date2 - date1).days) + 1):
        dt = date1 + timedelta(n)
        date_list.append(dt.strftime("%Y-%m-%d"))
    return date_list

def monsoon_dates(year):
    return daterange(date(int(year), 4, 1), date(int(year), 10, 31))

class LandslideDataset(Dataset):
    def __init__(self, image_dir, label_dir, split):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.split = split

        # Apply monsoon date bounds
        years = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]
        monsoon_date_list = []
        for y in years:
            monsoon_date_list.append(monsoon_dates(y))

        monsoon_date_list = [x + '.npy' for x in monsoon_date_list]
        self.image_fns = ['sample_' + x for x in monsoon_date_list]
        self.label_fns = ['label_' + x for x in monsoon_date_list]

    def __len__(self):
        if self.split == 'train':
            image_fns = [x for x in self.image_fns if "2023" not in x]
        else:
            image_fns = [x for x in self.image_fns if "2023" in x]
        return len(image_fns)

    def __getitem__(self, index):
        image_fns = sort(self.image_fns)

        if self.split == 'train':
            image_fns = [x for x in image_fns if "2023" not in x]
        else:
            image_fns = [x for x in image_fns if "2023" in x]

        label_fns = list(map(lambda x: x.replace('sample', 'label'), image_fns))

        image_fn = image_fns[index]
        label_fn = label_fns[index]
        image_fp = os.path.join(self.image_dir, image_fn)
        label_fp = os.path.join(self.label_dir, label_fn)
        multichannel_image = load('{}'.format(image_fp), allow_pickle=True).astype('float32')
        label_class = load('{}'.format(label_fp), allow_pickle=True)
        multichannel_image = self.transform(multichannel_image)
        multichannel_image = torch.transpose(multichannel_image, 0, 1)
        label_class = self.transform(label_class)

        if multichannel_image.shape != torch.Size([32, 60, 100]):
            multichannel_image = torch.transpose(multichannel_image, 1, 2)
        if label_class.shape != torch.Size([1, 60, 100]):
            label_class = torch.transpose(label_class, 1, 2)

        return multichannel_image.float(), label_class.float()

    def transform(self, image):
        transform_ops = transforms.ToTensor()
        return transform_ops(image)