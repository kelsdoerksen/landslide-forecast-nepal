"""
Script to predict on test set after training model
"""

from dataset import *
import numpy as np
from model import models, unet_modules
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from train import *
from operator import add
from utils import *
from metrics import *


def predict(in_model, test_dataset, wandb_experiment, seed, out_dir, device, district_masks):
    """
    Predict standard way (no dropout at test time)
    """
    # Make deterministic
    make_deterministic(seed)

    threshold = 0.2

    # Setting model to eval mode
    unetmodel = models.UNet(n_channels=32, n_classes=1)
    unetmodel.load_state_dict(torch.load(in_model)['state_dict'])
    unetmodel.eval()

    # Data loader for test set
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    loss_criterion = nn.BCEWithLogitsLoss()
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
            outputs = unetmodel(inputs)

            # Apply sigmoid for predictions
            outputs_probs = torch.sigmoid(outputs)

            # Append first to preserve image shape for future plotting
            gt.append(labels.detach().numpy())
            preds.append(outputs_probs.detach().numpy())

            bce_score += loss_criterion(outputs, labels)
            p, r = precision_recall_threshold(labels, inputs, threshold, district_masks)
            precision += p
            recall += r

    print('test set BCE is: {}'.format(bce_score / len(test_loader)))

    # Writing things to file
    wandb_experiment.log({
        'test set BCE': bce_score / len(test_loader),
        'test set Precision': precision / len(test_loader),
        'test set Recall': recall / len(test_loader)
    })


    for i in range(len(gt)):
        np.save('{}/groundtruth_{}.npy'.format(out_dir, i), gt[i])
        np.save('{}/pred_{}.npy'.format(out_dir, i), preds[i])

    with open('{}/model_testing_results.txt'.format(out_dir), 'w') as f:
        f.write('Test set Precision is: {}'.format(precision / len(test_loader)))
        f.write('Test set Recall is: {}'.format(recall / len(test_loader)))

