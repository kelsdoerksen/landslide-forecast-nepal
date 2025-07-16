"""
Data Augmentations
"""

import random
from torch.utils.data import DataLoader, TensorDataset
import torch
import wandb
import numpy as np



def drop_channels(data_loader, num_channels, batch_size, wandb_experiment, split):
    """
    Function to drop n number of channels to generate new samples
    and run this num_iters times
    :split: train, test, or val
    """

    # Load all data in one batch
    full_loader = DataLoader(data_loader.dataset, batch_size=len(data_loader.dataset), shuffle=False)
    original_data, original_labels = next(iter(full_loader))

    # Generate list of channels to drop
    random_list = []
    for i in range(num_channels):
        n = random.randint(0,31)
        random_list.append(n)

    wandb_experiment.log({
        '{} split channels dropped'.format(split): random_list
    })

    aug_data = original_data.clone()
    for c in random_list:
        aug_data[:, c, :, :] = 0

    # Concatenate original and augmented data
    combined_data = torch.cat([original_data, aug_data], dim=0)
    combined_labels = torch.cat([original_labels, original_labels], dim=0)

    #  Wrap in a TensorDataset
    combined_dataset = TensorDataset(combined_data, combined_labels)

    # Create new dataloader and return this
    new_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

    return new_loader


def rand_bbox_batch(size, lam):
    """Generate bounding boxes for a batch."""
    B, C, H, W = size
    cut_rat = np.sqrt(1. - lam)
    cut_h = (H * cut_rat).astype(int)
    cut_w = (W * cut_rat).astype(int)

    # Uniform center points per sample
    cx = np.random.randint(W, size=B)
    cy = np.random.randint(H, size=B)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix(data_loader, alpha=1.0, batch_size=32):
    """
    Implement CutMix to preserve the spatial structure
    since we care about the locations and return the full dataset
    (original + cutmix)
    """
    if alpha <= 0:
        return data_loader

    # Load all data into once batch for further processing
    full_loader = DataLoader(data_loader.dataset, batch_size=len(data_loader.dataset), shuffle=False)
    original_data, original_labels = next(iter(full_loader))

    B, C, H, W = original_data.size()
    lam = np.random.beta(alpha, alpha)

    aug_data = original_data.clone()
    aug_labels = original_labels.clone()

    rand_index = torch.randperm(B).to(original_data).detach().numpy()

    bbx1, bby1, bbx2, bby2 = rand_bbox_batch((B, C, H, W), lam)

    for i in range(B):
        aug_data[i, :, bby1[i]:bby2[i], bbx1[i]:bbx2[i]] = aug_data[int(rand_index[i]), :, bby1[i]:bby2[i], bbx1[i]:bbx2[i]]
        aug_labels[i, :, bby1[i]:bby2[i], bbx1[i]:bbx2[i]] = aug_labels[int(rand_index[i]), :, bby1[i]:bby2[i], bbx1[i]:bbx2[i]]

    # Concatenate original and augmented data
    combined_data = torch.cat([original_data, aug_data], dim=0)
    combined_labels = torch.cat([original_labels, aug_labels], dim=0)

    #  Wrap in a TensorDataset
    combined_dataset = TensorDataset(combined_data, combined_labels)

    # Create new dataloader and return this
    new_loader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=True)

    return new_loader










