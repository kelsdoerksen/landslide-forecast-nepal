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


def predict(in_model, test_dataset, wandb_experiment, out_dir, device, district_masks, exp_type, test_loss):
    """
    Predict
    """

    if test_loss == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    if test_loss == 'bce_pos_weight':
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.3]))    # penalizes false positives

    threshold = 0.2
    unetmodel = models.UNet(n_channels=32, n_classes=1)
    # Setting model to eval mode
    if exp_type == 'monsoon_test':
        unetmodel.load_state_dict(in_model['state_dict'])
    else:
        unetmodel.load_state_dict(torch.load(in_model)['state_dict'])
    unetmodel.eval()

    # Data loader for test set
    test_loader = DataLoader(test_dataset, batch_size=10, shuffle=False)

    loss_criterion = nn.BCEWithLogitsLoss()
    bce_score = 0
    precision = 0
    recall = 0
    epoch_pct_cov_recall = 0
    epoch_pct_cov_precision = 0
    epoch_loss = 0
    # iterate over the test set
    preds = []
    gt = []

    print('length of test dataset is: {}'.format(len(test_loader)))

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

            if test_loss == 'tversky':
                loss = tversky_loss(outputs, labels)
            else:
                loss = criterion(outputs, labels)       # Calculate loss

            p, r = precision_recall_threshold(labels, outputs_probs, threshold, district_masks)
            pct_cov_precision, pct_cov_recall = precision_and_recall_threshold_pct_cov(labels, outputs_probs, threshold,
                                                                                       district_masks)
            precision += p
            recall += r
            epoch_pct_cov_precision += pct_cov_precision
            epoch_pct_cov_recall += pct_cov_recall
            epoch_loss += loss.item()

    print('test set BCE is: {}'.format(bce_score / len(test_loader)))

    # Writing things to file
    wandb_experiment.log({
        'test set loss': epoch_loss / len(test_loader),
        'test set Precision': precision / len(test_loader),
        'test set Recall': recall / len(test_loader),
        'train Precision pct cov': epoch_pct_cov_precision / len(test_loader),
        'train Recall pct cov': epoch_pct_cov_recall / len(test_loader),
    })


    for i in range(len(gt)):
        np.save('{}/groundtruth_{}.npy'.format(out_dir, i), gt[i])
        np.save('{}/pred_{}.npy'.format(out_dir, i), preds[i])

    with open('{}/model_testing_results.txt'.format(out_dir), 'w') as f:
        f.write('Test set Precision is: {}'.format(precision / len(test_loader)))
        f.write('Test set Recall is: {}'.format(recall / len(test_loader)))

