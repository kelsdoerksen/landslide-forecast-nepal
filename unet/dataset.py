"""
Landslide Dataset Module

split: monsoon_train refers to training the model with all the latest data (2016-2023) to be able
to run it on the 2024 test set
monsoon_test refers to the 2024 monsoon season testing model performance
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
    def __init__(self, image_dir, label_dir, split, out_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.split = split
        self.out_dir = out_dir

        # Apply monsoon date bounds
        years = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
        monsoon_date_list = []
        for y in years:
            monsoon_date_list.extend(monsoon_dates(y))

        # Get list of dates from samples
        fns = os.listdir(image_dir)
        fns = [s.strip('sample_') for s in fns]
        fns = [s.strip('.npy') for s in fns]

        fns_monsoon = [x for x in fns if x in monsoon_date_list]

        image_fns_monsoon = ['sample_' + x for x in fns_monsoon]
        label_fns_monsoon = ['label_' + x for x in fns_monsoon]
        self.image_fns = [x + '.npy' for x in image_fns_monsoon]
        self.label_fns = [x + '.npy' for x in label_fns_monsoon]

    def __len__(self):
        if self.split == 'train':
            image_fns = [x for x in self.image_fns if "2023" not in x]
            image_fns = [x for x in image_fns if "2023" not in x]
        elif self.split == 'test':
            image_fns = [x for x in self.image_fns if "2023" in x]
        elif self.split == 'monsoon_test':
            image_fns = [x for x in self.image_fns if "2024" in x]      #
        elif self.split == 'monsoon_train':
            image_fns = [x for x in self.image_fns if "2024" not in x]
        return len(image_fns)

    def __getitem__(self, index):
        image_fns = sort(self.image_fns)

        if self.split == 'train':
            image_fns = [x for x in image_fns if "2023" not in x]
            image_fns = [x for x in image_fns if "2024" not in x]
        elif self.split == 'test':
            image_fns = [x for x in image_fns if "2023" in x]
            # Save list of image_fns to file so we know what dates were used
            image_fns_save = [s.strip('sample_') for s in image_fns]
            image_fns_save = [s.strip('.npy') for s in image_fns]
            with open('{}/test_dates.txt'.format(self.out_dir), 'w') as f:
                for line in image_fns_save:
                    f.write('{}'.format(line))
        elif self.split == 'monsoon_test':
            image_fns = [x for x in image_fns if "2024" in x]
            # Save list of image_fns to file so we know what dates were used
            image_fns_save = [s.strip('sample_') for s in image_fns]
            image_fns_save = [s.strip('.npy') for s in image_fns]
            with open('{}/test_dates.txt'.format(self.out_dir), 'w') as f:
                for line in image_fns_save:
                    f.write('{}'.format(line))
        elif self.split == 'monsoon_train':
            image_fns = [x for x in image_fns if "2024" not in x]

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