"""
Data Augmentations
"""

import random
from torch.utils.data import DataLoader, TensorDataset
import torch

def drop_channels(data_loader, num_channels, batch_size):
    """
    Function to drop n number of channels to generate new samples
    and run this num_iters times
    """

    # Load all data in one batch
    full_loader = DataLoader(data_loader.dataset, batch_size=len(data_loader.dataset), shuffle=False)
    original_data, original_labels = next(iter(full_loader))

    # Generate list of channels to drop
    random_list = []
    for i in range(num_channels):
        n = random.randint(0,31)
        random_list.append(n)

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