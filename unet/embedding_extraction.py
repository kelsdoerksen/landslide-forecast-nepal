"""
Script just for embedding extraction for features in final ML classifier
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


def run_embedding_extraction(model,
                             train_set,
                             test_set,
                             district_masks,
                             batch_size,
                             learning_rate,
                             weight_decay,
                             opt,
                             training_loss,
                             save_dir,
                             experiment,
                             epochs=40
                             ):
    """
    Train model for x epochs, freeze and run
    on test set to get final embeddings to be
    used in ML classifier for binary prediction
    """

    """
    Train model for binary classificaiton based 
    on the masks and pooling
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    threshold = 0.15

    # --- Setting up optimizer
    if opt == 'rms':
        optimizer = optim.RMSprop(model.parameters(),
                                  lr=learning_rate, weight_decay=weight_decay)
    if opt == 'adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=learning_rate, weight_decay=weight_decay)

    if opt == 'adam_explr':
        optimizer = optim.Adam(model.parameters(),
                               lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)

    # Setting up loss for final binary classification
    if training_loss == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    if training_loss == 'bce_fp_2':
        criterion = BCE_FP(false_positive_weight=2.0, false_negative_weight=1.0, eps=1e-7)
    if training_loss == 'bce_fp_3':
        criterion = BCE_FP(false_positive_weight=3.0, false_negative_weight=1.0, eps=1e-7)
    if training_loss == 'bce_fp_4':
        criterion = BCE_FP(false_positive_weight=4.0, false_negative_weight=1.0, eps=1e-7)
    if training_loss == 'bce_fn_5':
        criterion = BCE_FP(false_positive_weight=1.0, false_negative_weight=5.0, eps=1e-7)
    if training_loss == 'bce_fn_2':
        pos_weight = torch.tensor([2])
        pos_weight = pos_weight.to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    if training_loss == 'bce_fn_6':
        pos_weight = torch.tensor([6])
        pos_weight = pos_weight.to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    grad_scaler = torch.cuda.amp.GradScaler()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    model = model.to(device)
    model.train()

    global_step = 0
    epoch_number = 0

    for epoch in range(epochs):
        print('Training EPOCH {}:'.format(epoch_number))
        epoch_number += 1
        epoch_loss = 0
        epoch_precision = 0
        epoch_recall = 0
        epoch_f1 = 0

        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            print("Input stats - mean:", inputs.mean().item(), "std:", inputs.std().item(),
                  "min:", inputs.min().item(), "max:", inputs.max().item())

            # Zero gradients for every batch
            optimizer.zero_grad()

            # Get embeddings from input
            embeddings, district_logits = model(inputs)
            embeddings = embeddings.to(device)
            district_logits = district_logits.to(device)

            # Get binary labels
            binary_labels = get_binary_label(labels, district_masks)
            binary_labels = binary_labels.to(device)

            loss = criterion(district_logits.squeeze(2), binary_labels.float())       # Calculate loss

            # Probability conversion so I can do the other metric calculations
            precision, recall, f1 = binary_classification_precision_recall(threshold, district_logits, binary_labels)

            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            global_step += 1
            epoch_loss += loss.item()
            epoch_precision += np.sum(precision)/len(precision)
            epoch_recall += np.sum(recall)/len(recall)
            epoch_f1 += np.sum(f1)/len(f1)

        experiment.log({
            'train loss': epoch_loss / len(train_loader),
            'train Precision': epoch_precision / len(train_loader),
            'train Recall': epoch_recall / len(train_loader),
            'train F1': epoch_f1 / len(train_loader),
            'train Precision pct cov': 'N/A',
            'train Recall pct cov': 'N/A',
            'step': global_step,
            'epoch': epoch,
            'optimizer': opt
        })


        # Save model for reference
        torch.save(model, "{}/embedding_extractor_model.pth".format(save_dir))

    # Now extract embeddings on the test set
    model.eval()

    all_fns = []
    for d in test_set.datasets:
        all_fns.extend(d.image_fns)

    # Freeze weights
    for p in model.parameters():
        p.requires_grad = False

    # Loading full set to extract
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
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
        out_file = Path(save_dir) / f"{year}_embeddings.csv"
        df.to_csv(out_file, index=False)
        print(f"Saved {year} embeddings to {out_file}")










