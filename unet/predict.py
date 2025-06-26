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
from losses import *


def predict(in_model, test_dataset, wandb_experiment, out_dir, device, district_masks, exp_type, test_loss):
    """
    Predict
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if test_loss == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    if test_loss == 'bce_pos_weight_01':
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.1], device=device))    # penalizes false positives
    if test_loss == 'bce_pos_weight_02':
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.2], device=device))    # penalizes false positives
    if test_loss == 'bce_pos_weight_03':
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.3], device=device))    # penalizes false positives
    if test_loss == 'bce_pos_weight_04':
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.4], device=device))  # penalizes false positives
    if test_loss == 'dice_bce':
        criterion = DiceBCELoss()
    if test_loss == 'dice_bce_w3':
        criterion = DiceWeightedBCE03Loss()
    if test_loss == 'bce_pos_weight_15':
        weight = 1.5
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight], device=device))  # penalizes false negatives
    if test_loss == 'bce_pos_weight_3':
        weight = 3
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight], device=device))  # penalizes false negatives

    threshold = 0.2
    if exp_type == 'unet_mini':
        unetmodel = models.UNetMini(n_channels=32, n_classes=1, dropout=0)
        unetmodel.load_state_dict(torch.load(in_model)['state_dict'])
    else:
        unetmodel = models.UNet(n_channels=32, n_classes=1, dropout=0)
        if exp_type == 'monsoon_test':
            unetmodel.load_state_dict(in_model['state_dict'])
        else:
            unetmodel.load_state_dict(torch.load(in_model)['state_dict'])

    # Move model to device
    unetmodel.to(device)
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
            gt.append(labels.cpu().numpy())
            preds.append(outputs_probs.cpu().numpy())

            if test_loss == 'tversky':
                loss = tversky_loss(outputs, labels)
            elif test_loss == 'tversky_FN_penalize':
                loss = tversky_loss_penalize_fn(outputs, labels)
            elif test_loss == 'dice':
                loss = dice_loss(outputs, labels)
            elif test_loss == 'logcosh_dice':
                loss = logcosh_dice_loss(outputs, labels)
            else:
                loss = criterion(outputs, labels)       # Calculate loss

            p, r, tp, fp, fn = precision_recall_threshold(labels, outputs_probs, threshold, district_masks)
            #pct_cov_precision, pct_cov_recall = precision_and_recall_threshold_pct_cov(labels, outputs_probs, threshold, district_masks)
            precision += p
            recall += r
            #epoch_pct_cov_precision += pct_cov_precision
            #epoch_pct_cov_recall += pct_cov_recall
            epoch_loss += loss.item()

    print('test set loss is: {}'.format(bce_score / len(test_loader)))

    # Writing things to file
    wandb_experiment.log({
        'test set loss': epoch_loss / len(test_loader),
        'test set Precision': precision / len(test_loader),
        'test set Recall': recall / len(test_loader),
        'test set Precision pct cov': 'N/A',
        'test set Recall pct cov': 'N/A',
    })


    for i in range(len(gt)):
        np.save('{}/groundtruth_{}.npy'.format(out_dir, i), gt[i])
        np.save('{}/pred_{}.npy'.format(out_dir, i), preds[i])

    with open('{}/model_testing_results.txt'.format(out_dir), 'w') as f:
        f.write('Test set Precision is: {}'.format(precision / len(test_loader)))
        f.write('Test set Recall is: {}'.format(recall / len(test_loader)))

