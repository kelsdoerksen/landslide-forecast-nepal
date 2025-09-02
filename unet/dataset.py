"""
Landslide Dataset Module

split: monsoon_train refers to training the model with all the latest data (2016-2023) to be able
to run it on the 2024 test set
monsoon_test refers to the 2024 monsoon season testing model performance
"""
import ipdb
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

def out_of_monsoon_dates(year):
    date_list_pre = daterange(date(int(year), 1, 1), date(int(year), 3, 31))
    date_list_post = daterange(date(int(year), 11, 1), date(int(year), 12, 31))
    return date_list_pre + date_list_post

class LandslideDataset(Dataset):
    def __init__(self, image_dir, label_dir, split, exp_type, test_year, out_dir, mean=None, std=None,
                 max_val=None, min_val=None, norm=None, stride=0):
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.split = split
        self.exp_type = exp_type
        self.test_year = test_year
        self.out_dir = out_dir
        self.mean = mean
        self.std = std
        self.max_val = max_val
        self.min_val = min_val
        self.norm = norm
        self.stride = stride

        # Apply valid date bounds depending on the experiment type
        years = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
        if exp_type == 'temporal-cv':
            # Don't remove earlier or later years, we just want
            years = years
        elif exp_type == 'monsoon_tool' and test_year == 2025:
            years = [2023, 2024]
        else:
            # Remove any years greater than test year because we do not want these samples
            years = [x for x in years if x <= int(test_year)]
        valid_date_list = []

        if split == 'train':
            for y in years:
                valid_date_list.extend(monsoon_dates(y))

        # We only ever test on out of monsoon dates
        if split == 'test':
            for y in years:
                if exp_type == 'out-monsoon':
                    valid_date_list.extend(out_of_monsoon_dates(y))
                else:
                    valid_date_list.extend(monsoon_dates(y))

        # Get list of dates from samples
        fns = os.listdir(image_dir)
        fns = [s.strip('sample_') for s in fns]
        fns = [s.strip('.npy') for s in fns]

        fns_valid = [x for x in fns if x in valid_date_list]

        image_fns_valid = ['sample_' + x for x in fns_valid]
        label_fns_valid = ['label_' + x for x in fns_valid]

        image_fns = [x + '.npy' for x in image_fns_valid]
        label_fns = [x + '.npy' for x in label_fns_valid]

        if self.split == 'train':
            image_fns = [x for x in image_fns if "{}".format(self.test_year) not in x]
            if self.stride > 0:
                image_fns = image_fns[::self.stride]
        elif self.split == 'test':
            image_fns = [x for x in image_fns if "{}".format(self.test_year) in x]
            with open('{}/test_dates.txt'.format(self.out_dir), 'w') as f:
                for line in image_fns:
                    f.write('{}'.format(line))

        self.image_fns = sorted(image_fns)
        self.label_fns = [x.replace("sample", "label") for x in self.image_fns]


    def __len__(self):
        return len(self.image_fns)

    def __getitem__(self, index):
        image_fn = self.image_fns[index]
        label_fn = self.label_fns[index]

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

        # Standardize from given mean and std if not None
        # Normalize data
        if self.mean is not None and self.std is not None:
            if self.norm == 'zscore':
                # Ensure shape (32, 1, 1) for broadcasting
                mean = self.mean.view(-1, 1, 1)
                std = self.std.view(-1, 1, 1)
                multichannel_image = (multichannel_image - mean) / (std + 1e-8)
                print("Normalization mean:", mean)
                print("Normalization std:", std)
        if self.max_val is not None and self.min_val is not None:
            if self.norm == 'minmax':
                min_val = self.min_val.view(-1, 1, 1)
                max_val = self.max_val.view(-1, 1, 1)
                multichannel_image = (multichannel_image - min_val) / (max_val - min_val + 1e-8)


        return multichannel_image.float(), label_class.float()

    def transform(self, image):
        transform_ops = transforms.ToTensor()
        return transform_ops(image)