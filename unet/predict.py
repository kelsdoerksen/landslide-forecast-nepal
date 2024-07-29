"""
Script to predict on test set after training model
"""

from dataset import *
import numpy as np
from model import *
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from train import *
from operator import add
from losses import *
from utils import *


def predict(in_model, target, test_dataset, wandb_experiment, channels, seed, out_dir, device):
    """
    Predict standard way (no dropout at test time)
    """
    # Make deterministic
    make_deterministic(seed)

    # Setting model to eval mode
    unet = models.UNet(n_channels=channels, n_classes=1)
    unet.load_state_dict(torch.load(in_model)['state_dict'])
    unet.eval()

    # Data loader for test set
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

    loss_criterion = nn.MSELoss()
    mse_score = 0
    # iterate over the test set
    preds = []
    gt = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # predict the mask
            outputs = unet(inputs)
            test_mask = ~torch.isnan(labels)

            # Append first to preserve image shape for future plotting
            gt.append(labels.detach().numpy())
            preds.append(outputs.detach().numpy())

            outputs = outputs[test_mask]
            labels = labels[test_mask]

            mse_score += loss_criterion(outputs, labels)

    print('test set mse is: {}'.format(mse_score / len(test_loader)))
    print('test set rmse is: {}'.format(np.sqrt((mse_score / len(test_loader)).detach().numpy())))

    wandb_experiment.log({
        'test set mse': mse_score / len(test_loader),
        'test set rmse': np.sqrt((mse_score / len(test_loader)).detach().numpy())
    })

    for i in range(len(gt)):
        np.save('{}/{}channels_{}_groundtruth_{}.npy'.format(out_dir, channels, target, i), gt[i])
        np.save('{}/{}channels_{}_pred_{}.npy'.format(out_dir, channels, target, i), preds[i])

