"""
Script for loading pre-trained embedding extraction, running on samples
to generate new embeddings
"""

import torch
import argparse
import os
from train import train_model, train_binary_classification_model
from predict import predict, predict_binary_classification
import wandb
from utils import *
from model import models, unet_modules
from dataset import *
import logging
from osgeo import gdal
import random
from torch.utils.data import ConcatDataset, DataLoader
from metrics import *
import numpy as np
from model import *
from losses import *
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset
import wandb
from torch import optim
from pathlib import Path
from predict import *
from augmentation import *
from torchvision.transforms import v2, Lambda


def get_args():
    parser = argparse.ArgumentParser(description='Running pre-trained embedding extraction on test dataset')
    parser.add_argument('--channels', help='Specify number of channels, 32 or 10 for aggregated', required=True),
    return parser.parse_args()


def generate_district_masks(file_name):
    '''
    Create the masks for each district from Nepal raster
    '''
    # Load in Nepal District file
    ds = gdal.Open('{}'.format(file_name))
    array = ds.GetRasterBand(1).ReadAsArray()

    # Create dict to match pixel values
    district_dict = {'Bhojpur': 1, 'Dhankuta': 2, 'Ilam': 3, 'Jhapa': 4, 'Khotang': 5, 'Morang': 6, 'Okhaldhunga': 7,
    'Panchthar': 8, 'Sankhuwasabha': 9, 'Solukhumbu': 10, 'Sunsari': 11, 'Taplejung': 12, 'Terhathum': 13,
    'Udayapur': 14, 'Bara': 15, 'Dhanusha': 16, 'Mahottari': 17, 'Parsa': 18, 'Rautahat': 19, 'Saptari': 20,
    'Sarlahi': 21, 'Siraha': 22, 'Bhaktapur': 23, 'Chitawan': 24, 'Dhading': 25, 'Dolakha': 26,
    'Kabhrepalanchok': 27, 'Kathmandu': 28, 'Lalitpur': 29, 'Makawanpur': 30, 'Nuwakot': 31, 'Ramechhap': 32,
    'Rasuwa': 33, 'Sindhuli': 34, 'Sindhupalchok': 35, 'Baglung': 36, 'Gorkha': 37, 'Kaski': 38, 'Lamjung': 39,
    'Manang': 40, 'Mustang': 41, 'Myagdi': 42, 'Nawalparasi_W': 43, 'Parbat': 44, 'Syangja': 45, 'Tanahu': 46,
    'Arghakhanchi': 47, 'Banke': 48, 'Bardiya': 49, 'Dang': 50, 'Gulmi': 51, 'Kapilbastu': 52, 'Palpa': 53,
    'Nawalparasi_E': 54, 'Pyuthan': 55, 'Rolpa': 56, 'Rukum_E': 57, 'Rupandehi': 58, 'Dailekh': 59, 'Dolpa': 60,
    'Humla': 61, 'Jajarkot': 62, 'Jumla': 63, 'Kalikot': 64, 'Mugu': 65, 'Rukum_W': 66, 'Salyan': 67,
    'Surkhet': 68, 'Achham': 78, 'Baitadi': 70, 'Bajhang': 71, 'Bajura': 72, 'Dadeldhura': 73, 'Darchula': 74,
    'Doti': 75, 'Kailali': 76, 'Kanchanpur': 77}

    new_dict = {}
    for k, v in district_dict.items():
        # Only include District of interest
        new_array = array.copy()
        new_array[new_array != v] = 0
        # Bound between 0 and 1
        filtered_array = new_array.copy()
        # Landslide class set to 1
        filtered_array[filtered_array == v] = 1
        new_dict[k] = filtered_array

    return new_dict


if __name__ == '__main__':
    args = get_args()
    n_channels = int(args.channels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    root_dir = '/Volumes/PRO-G40/landslides/Nepal_Landslides_Forecasting_Project/Monsoon2024_Prep'

    # Grabbing data to generate embeddings for
    if n_channels == 10:
        sample_dir = '{}/UNet_Samples_14Day_GPMv07/ECMWF/ensemble_0_agg'.format(root_dir)
        train_sample_dir = '{}/UNet_Samples_14Day_GPMv07/UKMO/ensemble_0_agg'.format(root_dir)
    else:
        sample_dir = '{}/UNet_Samples_14Day_GPMv07/ECMWF/ensemble_0'.format(root_dir)
        train_sample_dir = '{}/UNet_Samples_14Day_GPMv07/UKMO/ensemble_0'.format(root_dir)
    label_dir = '{}/Binary_Landslide_Labels_14day'.format(root_dir)

    district_masks = generate_district_masks('{}/District_Labels.tif'.format(root_dir))

    # Set the savedir
    save_dir = '{}/embeddings/embedding_unetmini_40e_{}channel_2024_ecmwf'.format(root_dir, n_channels)

    # Load unet model for embeddings
    model = models.UNetDistrictMini(n_channels=n_channels, n_classes=1, dropout=float(0.4),
                                   embedding_dim=n_channels,
                                   hidden_dim=64, district_masks=district_masks)

    # Going to hard-code paths to 2024 embedding models since I have already trained them
    if n_channels == 10:
        # Load 10 channel model trained 2016-2023
        in_model = '{}/embeddings/embedding_extractor_model_10channel_2016-2023_odrk65ya.pth'.format(root_dir)
    else:
        # Load 32 channel model trained 2016-2023
        in_model = '{}/embeddings/embedding_extractor_model_32channel_2016-2023_49ooo678.pth'.format(root_dir)

    # Load the pre-trained model
    model_load = torch.load(in_model, weights_only=False, map_location=torch.device('cpu'))

    # Load the original training dataset so I can get the mean and std for normalizing
    landslide_train = LandslideDataset(train_sample_dir, label_dir, 'train', 'embedding_extractor', 2024,
                                       save_dir, n_channels=n_channels)

    channel_sum = torch.zeros(n_channels)
    channel_sq_sum = torch.zeros(n_channels)
    num_pixels = 0

    train_loader = DataLoader(landslide_train, batch_size=1, shuffle=True)
    for images, _ in train_loader:
        images = images.float().contiguous()  # (n_channels, 60, 100)
        images_flat = images.view(n_channels, -1)  # (n_channels, 6000)

        channel_sum += images_flat.sum(dim=1)
        channel_sq_sum += (images_flat ** 2).sum(dim=1)
        num_pixels += images_flat.shape[1]

    mean = channel_sum / num_pixels
    std = torch.sqrt(channel_sq_sum / num_pixels - mean ** 2)

    static_channels = [0, 1, 2, 3]  # static channels dem, aspect, slope, modis

    # For static channels, compute mean/std across spatial pixels only from the first sample since its always the same
    static_sample, _ = next(iter(train_loader))
    static_sample = static_sample.float().contiguous()
    static_sample_flat = static_sample.view(n_channels, -1)

    for idx in static_channels:
        mean[idx] = static_sample_flat[idx].mean()
        std[idx] = static_sample_flat[idx].std()

    global_min = None
    global_max = None

    # Now load the dataset that we want to extract the embeddings from
    landslide_test_dataset = LandslideDataset(sample_dir, label_dir, 'test', 'embedding_extractor', 2024,
                                              save_dir, mean=mean, std=std, max_val=global_max, min_val=global_min,
                                              norm='zscore', n_channels=n_channels)

    # Now let's run embedding extraction
    model.eval()

    all_fns = []
    all_fns.extend(landslide_test_dataset.image_fns)

    # Freeze weights
    for p in model.parameters():
        p.requires_grad = False

    # Loading full set to extract
    test_loader = DataLoader(landslide_test_dataset, batch_size=32, shuffle=False)
    rows = []
    rows_by_year = {}

    with torch.no_grad():
        for idx, (inputs, labels) in enumerate(test_loader):
            batch_start = idx * test_loader.batch_size
            batch_end = batch_start + len(inputs)
            fns = all_fns[batch_start:batch_end]

            inputs = inputs.to(device)
            embeddings = model.unet(inputs)  # embeddings only

            for b, fn in enumerate(fns):
                date_str = fn.replace("sample_", "").replace(".npy", "")
                year, month, day = date_str.split("-")
                year = int(year)

                if year not in rows_by_year:
                    rows_by_year[year] = []

                for district in sorted(district_masks.keys()):
                    pooled = model.district_classifier.masked_avg_pool(
                        embeddings[b], district_masks[district]
                    )  # (C,)
                    pooled = pooled.cpu().numpy()

                    row = {"date": date_str, "district": district}
                    for i, val in enumerate(pooled):
                        row[f"embed_{i}"] = val
                    rows_by_year[year].append(row)

    # Save each year
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    for year, rows in rows_by_year.items():
        df = pd.DataFrame(rows)
        out_file = Path(save_dir) / '{}_{}channels_embeddings.csv'.format(year, n_channels)
        df.to_csv(out_file, index=False)
        print(f"Saved {year} embeddings to {out_file}")










