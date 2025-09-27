"""
Script for loading pre-trained embedding extraction, running on samples
to generate new embeddings
"""

import torch
import argparse
import os
from train import train_model, train_binary_classification_model
from predict import predict, predict_binary_classification
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
from torch import optim
from pathlib import Path
from predict import *
from augmentation import *
from collections import defaultdict
import torch.nn as nn


def get_args():
    parser = argparse.ArgumentParser(description='Running pre-trained embedding extraction on test dataset')
    parser.add_argument('--channels', help='Specify number of channels, 32 or 10 for aggregated', required=True),
    parser.add_argument('--exp', help='Experiment type, specify if ecmwf included in training. Call ecmwf', required=True)
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
    exp = args.exp

    print('Setting up directories...')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_dir = '/Volumes/PRO-G40/landslides/Nepal_Landslides_Forecasting_Project/Monsoon2024_Prep'

    # Grabbing data to generate embeddings for
    if n_channels == 10:
        ecmwf_sample_dir = '{}/UNet_Samples_14Day_GPMv07/ECMWF/ensemble_0_agg'.format(root_dir)
        train_sample_dir = '{}/UNet_Samples_14Day_GPMv07/UKMO/ensemble_0_agg'.format(root_dir)
    else:
        ecmwf_sample_dir = '{}/UNet_Samples_14Day_GPMv07/ECMWF/ensemble_0'.format(root_dir)
        train_sample_dir = '{}/UNet_Samples_14Day_GPMv07/UKMO/ensemble_0'.format(root_dir)
    label_dir = '{}/Binary_Landslide_Labels_14day'.format(root_dir)

    district_masks = generate_district_masks('{}/District_Labels.tif'.format(root_dir))

    # Set the savedir
    if exp == 'ecmwf':
        save_dir = '{}/embeddings/embedding_unetmini_40e_{}channel_2024_ecmwf_trained'.format(root_dir, n_channels)
    else:
        save_dir = '{}/embeddings/embedding_unetmini_40e_{}channel_2024_ecmwf'.format(root_dir, n_channels)

    # Going to hard-code paths to 2024 embedding models since I have already trained them
    if exp == 'ecmwf':
        if n_channels == 10:
            in_model = '{}/embeddings/embedding_extractor_model_10channel_2016-2023_ecmwf-added_hxi04ic7.pth'.format(root_dir)
        else:
            in_model = '{}/embeddings/embedding_extractor_model_32channel_2016-2023_ecmwf-added_6mpi2te1.pth'.format(root_dir)
    else:
        if n_channels == 10:
            # Load 10 channel model trained 2016-2023
            in_model = '{}/embeddings/embedding_extractor_model_10channel_2016-2023_odrk65ya.pth'.format(root_dir)
        else:
            # Load 32 channel model trained 2016-2023
            in_model = '{}/embeddings/embedding_extractor_model_32channel_2016-2023_49ooo678.pth'.format(root_dir)

    # Load the pre-trained model
    model = torch.load(in_model, map_location=device, weights_only=False)

    # ----------------- Mean and Std from 2016-2023 UKMO dataset -----------------
    # Load the original training dataset so I can get the mean and std for normalizing
    # CORRECT_CH: channels to correct
    if n_channels == 10:
        CORRECT_CH = [7, 8, 9]
    if n_channels == 32:
        CORRECT_CH = [17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31]
    STATIC_CH = [0, 1, 2, 3]  # geospatial features

    # ----------------- Mean and Std from 2016-2023 UKMO dataset -----------------
    landslide_train = LandslideDataset(
        train_sample_dir, label_dir, 'train', 'embedding_extractor', 2024,
        save_dir, n_channels=n_channels)

    channel_sum = torch.zeros(n_channels)
    channel_sq_sum = torch.zeros(n_channels)
    num_pixels = 0

    train_loader = DataLoader(landslide_train, batch_size=1, shuffle=True)
    for images, _ in train_loader:
        images = images.float().contiguous()  # (C, H, W)
        images_flat = images.view(n_channels, -1)  # (C, H*W)
        channel_sum += images_flat.sum(dim=1)
        channel_sq_sum += (images_flat ** 2).sum(dim=1)
        num_pixels += images_flat.shape[1]

    mean = channel_sum / num_pixels
    std = torch.sqrt(torch.clamp(channel_sq_sum / num_pixels - mean ** 2, min=1e-12))

    # Static channels: compute from first sample only (unchanged)
    static_sample, _ = next(iter(train_loader))
    static_sample_flat = static_sample.float().contiguous().view(n_channels, -1)
    for idx in STATIC_CH:
        mean[idx] = static_sample_flat[idx].mean()
        std[idx] = static_sample_flat[idx].std()

    ukmo_mu_glob = mean
    ukmo_sd_glob = std

    # ---- monthly RAW stats from ECMWF 2023 to compare ----
    def monthly_stats_raw(ds, n_channels):
        sum_by_m = defaultdict(lambda: torch.zeros(n_channels))
        sq_by_m = defaultdict(lambda: torch.zeros(n_channels))
        npix_by_m = defaultdict(int)
        dl = DataLoader(ds, batch_size=1, shuffle=False)
        for i, (x, _) in enumerate(dl):
            x = x.float().contiguous().view(n_channels, -1)
            fn = ds.image_fns[i]  # format is "sample_YYYY-MM-DD.npy"
            m = int(fn.split('_')[-1].split('-')[1])  # month
            sum_by_m[m] += x.sum(1)
            sq_by_m[m] += (x ** 2).sum(1)
            npix_by_m[m] += x.shape[1]
        mu, sd = {}, {}
        for m in range(1, 13):
            N = max(1, npix_by_m[m])
            mu_m = sum_by_m[m] / N
            var = torch.clamp(sq_by_m[m] / N - mu_m ** 2, min=1e-12)
            mu[m] = mu_m
            sd[m] = torch.sqrt(var)
        return mu, sd


    # ----------------- Applying Affine correction only on channels we want to correct (CORRECT_CH) -----------------
    # x′=a⋅x+b affine transformation
    print('Getting monthly ECMWF stats and building month→global UKMO affine on ecmwf channels...')

    ecmwf_2023_raw = LandslideDataset(
        ecmwf_sample_dir, label_dir, 'test', 'embedding_extractor', 2023,
        save_dir, n_channels=n_channels, norm=None  # RAW
    )
    print("ECMWF 2023 samples (RAW):", len(ecmwf_2023_raw))
    assert len(ecmwf_2023_raw) > 0, "ECMWF 2023 dataset is empty; check paths."

    muE_m, sdE_m = monthly_stats_raw(ecmwf_2023_raw, n_channels)

    # Build per-month affine to target UKMO GLOBAL, but ONLY for CORRECT_CH
    A, B = {}, {}
    correct_idx = torch.tensor(CORRECT_CH, dtype=torch.long)
    for m in range(1, 13):
        # start as identity
        a = torch.ones(n_channels)
        b = torch.zeros(n_channels)
        # set affine only on the channels that need correction
        a[correct_idx] = ukmo_sd_glob[correct_idx] / torch.clamp(sdE_m[m][correct_idx], min=1e-12)
        b[correct_idx] = ukmo_mu_glob[correct_idx] - a[correct_idx] * muE_m[m][correct_idx]
        A[m], B[m] = a, b

    print('Getting ECMWF and aligning (affine on selected channels) → UKMO GLOBAL z-score...')
    # ---- dataset wrapper: (monthly affine on CORRECT_CH) → (GLOBAL UKMO z-score) ----
    class ECMWFMonthlyToUKMOGlobal(LandslideDataset):
        def __init__(self, *args, A=None, B=None, mu_glob=None, sd_glob=None,
                     correct_ch=None, **kwargs):
            kwargs['norm'] = None  # request RAW from base
            super().__init__(*args, **kwargs)
            self.A = A;
            self.B = B
            self.mu = mu_glob;
            self.sd = sd_glob
            self.correct_ch = torch.tensor(correct_ch or [], dtype=torch.long)

        def __getitem__(self, index):
            x, y = super().__getitem__(index)  # RAW [C,H,W]
            fn = self.image_fns[index]
            m = int(fn.split('_')[-1].split('-')[1])

            if len(self.correct_ch) > 0:
                # vectorized apply only on the chosen channels
                a = self.A[m][self.correct_ch].view(-1, 1, 1)
                b = self.B[m][self.correct_ch].view(-1, 1, 1)
                x[self.correct_ch] = x[self.correct_ch] * a + b

            # GLOBAL UKMO z-score (the training contract) on all channels
            x = (x - self.mu.view(-1, 1, 1)) / (self.sd.view(-1, 1, 1) + 1e-8)
            return x.float(), y.float()


    if exp != 'ecmwf':
        # ---- BN refresh on aligned 2023 if I haven't added this to the training already (selected channels corrected) ----
        ecmwf_bn_ds = ECMWFMonthlyToUKMOGlobal(
            ecmwf_sample_dir, label_dir, 'test', 'embedding_extractor', 2023,
            save_dir, n_channels=n_channels,
            A=A, B=B, mu_glob=ukmo_mu_glob, sd_glob=ukmo_sd_glob,
            correct_ch=CORRECT_CH
        )
        ecmwf_calib_loader = DataLoader(ecmwf_bn_ds, batch_size=32, shuffle=False)

        print('Running BN refresh on aligned 2023...')
        def set_only_bn_train(module):
            module.eval()
            for m in module.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    m.train()

        model = model.to(device)
        set_only_bn_train(model)
        for p in model.parameters():
            p.requires_grad = False

        with torch.no_grad():
            for _ in range(2):
                for xb, _ in ecmwf_calib_loader:
                    _ = model.unet(xb.to(device))

    model.eval()

    # ---- 2024 inference on aligned dataset (selected channels corrected) ----
    landslide_test_dataset = ECMWFMonthlyToUKMOGlobal(
        ecmwf_sample_dir, label_dir, 'test', 'embedding_extractor', 2024,
        save_dir, n_channels=n_channels,
        A=A, B=B, mu_glob=ukmo_mu_glob, sd_glob=ukmo_sd_glob,
        correct_ch=CORRECT_CH
    )

    # ---- sanity check: post-correction, post-UKMO-zscore channel stats ----
    z = next(iter(DataLoader(landslide_test_dataset, 32)))[0]
    print('means (first 16):', z.mean(dim=(0, 2, 3))[:16])
    print('stds  (first 16):', z.std(dim=(0, 2, 3))[:16])

    # Now let's run embedding extraction
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
        out_file = Path(save_dir) / '{}_{}channels_embeddings_affine.csv'.format(year, n_channels)
        df.to_csv(out_file, index=False)
        print(f"Saved {year} embeddings to {out_file}")










