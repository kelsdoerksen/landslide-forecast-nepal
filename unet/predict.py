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
from utils import *


def predict(in_model, test_dataset, wandb_experiment, channels, seed, out_dir, device, district_masks):
    """
    Predict standard way (no dropout at test time)
    """
    # Make deterministic
    make_deterministic(seed)

    threshold = 0.2

    # Setting model to eval mode
    unet = models.UNet(n_channels=channels, n_classes=1)
    unet.load_state_dict(torch.load(in_model)['state_dict'])
    unet.eval()

    # Data loader for test set
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=True)

    loss_criterion = nn.BCELoss()
    bce_score = 0
    precision = 0
    recall = 0
    # iterate over the test set
    preds = []
    gt = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # predict the mask
            outputs = unet(inputs)

            # Append first to preserve image shape for future plotting
            gt.append(labels.detach().numpy())
            preds.append(outputs.detach().numpy())

            bce_score += loss_criterion(outputs, labels)
            precision += precision_threshold(labels, inputs, threshold, district_masks)
            recall += recall_threshold(labels, inputs, threshold, district_masks)

    print('test set BCE is: {}'.format(bce_score / len(test_loader)))
    print('test set Precision is: {}'.format(precision / len(test_loader)))
    print('test set Recall is: {}'.format(recall / len(test_loader)))

    wandb_experiment.log({
        'test set BCE': bce_score / len(test_loader),
        'test set Precision': precision / len(test_loader),
        'test set Recall': recall / len(test_loader)
    })

    for i in range(len(gt)):
        np.save('{}/groundtruth_{}.npy'.format(out_dir, i), gt[i])
        np.save('{}/pred_{}.npy'.format(out_dir, i), preds[i])

