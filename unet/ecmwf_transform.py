"""
Transforming the ecmwf data to be similar in distribution
to the UKMO past data so I can use it in training with the feature
embedding extractor to test it out on the 2024 data better (and include
more samples)
"""

import torch
import argparse
from osgeo import gdal
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from collections import defaultdict
import torch.nn as nn
from dataset import *


def get_args():
    parser = argparse.ArgumentParser(description='Running pre-trained embedding extraction on test dataset')
    parser.add_argument('--channels', help='Specify number of channels, 32 or 10 for aggregated', required=True),
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    n_channels = int(args.channels)

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

    # Set the savedir
    save_dir = '{}/embeddings/ecmwf_transformed_{}'.format(root_dir, n_channels)


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


    # ---- Align 2023 (selected channels corrected) ----
    ecmwf_aligned = ECMWFMonthlyToUKMOGlobal(
        ecmwf_sample_dir, label_dir, 'test', 'embedding_extractor', 2023,
        save_dir, n_channels=n_channels,
        A=A, B=B, mu_glob=ukmo_mu_glob, sd_glob=ukmo_sd_glob,
        correct_ch=CORRECT_CH
    )

    fns = ecmwf_aligned.image_fns
    ecmwf_loader = DataLoader(ecmwf_aligned, batch_size=32, shuffle=False)
    # Iterate through and save the transformed data to file
    for i, data in enumerate(train_loader):
        inputs, labels = data
        inputs = inputs.detach().cpu().numpy()
        for j in range(inputs.shape[1]):
            print('Saving newly transformed ecmwf: {}'.format(fns[j]))
            np.save('{}/{}'.format(save_dir, fns[j]), inputs[0,j,:,:])
        del fns[0:inputs.shape[1]-1]

