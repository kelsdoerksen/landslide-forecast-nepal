"""
Training script for UNet
"""


from dataset import *
from metrics import *
import numpy as np
from model import *
import torch
import torch.nn as nn
import logging
from torch.utils.data import DataLoader, random_split
import wandb
from torch import optim
from pathlib import Path
from predict import *
from utils import *
import random

def train_model(model,
                device,
                dataset,
                save_dir,
                experiment,
                epochs: int,
                batch_size: int,
                learning_rate: float,
                opt,
                val_percent,
                weight_decay: float = 0,
                save_checkpoint: bool=True,
                district_masks = None
                ):

    # --- Split dataset into training and validation
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().
                                      manual_seed(random.randint(0,1000)))

    # --- DataLoaders
    # The DataLoader pulls instances of data from the Dataset, collects them in batches,
    # and returns them for consumption by your training loop.
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    threshold = 0.2

    # --- Setting up optimizer
    if opt == 'rms':
        optimizer = optim.RMSprop(model.parameters(),
                                  lr=learning_rate, weight_decay=weight_decay)

    if opt == 'adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=learning_rate, weight_decay=weight_decay)

    # Setting up loss
    criterion = nn.BCELoss()

    # --- Setting up schedulers
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)  # goal: minimize MSE score.
    grad_scaler = torch.cuda.amp.GradScaler()

    global_step = 0
    epoch_number = 0
    for epoch in range(epochs):
        print('Training EPOCH {}:'.format(epoch_number))
        epoch_number += 1
        model.train()
        epoch_loss = 0
        epoch_thr_precision = 0
        epoch_thr_recall = 0

        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # Zero gradients for every batch
            optimizer.zero_grad()  # if set_to_none=True, sets gradients of all optimized torch.Tensors to None, will have a lower memory footprint, can modestly improve performance

            outputs = model(inputs)                 # predict on input

            loss = criterion(outputs, labels)       # Calculate loss
            thr_precision = precision_threshold(labels, outputs, threshold, district_masks)
            thr_recall = precision_threshold(labels, outputs, threshold, district_masks)

            grad_scaler.scale(loss).backward()      # Compute partial derivative of the output f with respect to each of the input variables
            grad_scaler.step(optimizer)             # Updates value of parameters according to strategy
            grad_scaler.update()

            global_step += 1
            epoch_loss += loss.item()
            epoch_thr_precision += thr_precision
            epoch_thr_recall += thr_recall

        experiment.log({
            'train BCE loss': epoch_loss/len(train_loader),
            'train Precision': epoch_thr_precision/len(train_loader),
            'train Recall': epoch_thr_recall/len(train_loader),
            'step': global_step,
            'epoch': epoch,
            'optimizer': opt
        })

        # Evaluation -> Validation set
        histograms = {}
        for tag, value in model.named_parameters():
            tag = tag.replace('/', '.')
            if not (torch.isinf(value) | torch.isnan(value)).any():
                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

        # Run validation
        model.eval()
        # Disable gradient computation and reduce memory consumption.
        running_vloss = 0.0
        running_thr_precision = 0
        running_thr_recall = 0
        with torch.no_grad():
            for k, vdata in enumerate(val_loader):
                vinputs, vlabels = vdata
                vinputs, vlabels = vinputs.to(device), vlabels.to(device)
                voutputs = model(vinputs)
                vloss = criterion(voutputs, vlabels)
                v_prec = precision_threshold(vlabels, voutputs, threshold, district_masks)
                v_rec = recall_threshold(labels, voutputs, threshold, district_masks)

                running_vloss += vloss
                running_thr_recall += v_rec
                running_thr_precision += v_prec

        avg_vloss = running_vloss / len(val_loader)
        avg_prec = running_thr_precision / len(val_loader)
        avg_rec = running_thr_precision / len(val_loader)

        logging.info('Validation BCE score: {}'.format(avg_vloss))
        try:
            experiment.log({
                'learning rate': optimizer.param_groups[0]['lr'],
                'validation BCE loss': avg_vloss,
                'validation Precision': avg_prec,
                'validation Recall': avg_rec,
                'step': global_step,
                'epoch': epoch,
                **histograms
            })
        except:
            pass

        '''
        if save_checkpoint:
            out_model = '{}/checkpoint_epoch{}.pth'.format(save_dir, epoch)
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            torch.save({'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict()},
                       out_model)
            logging.info(f'Checkpoint {epoch} saved!')
        '''
    # Saving model at end of epoch with experiment name
    out_model = '{}/{}_last_epoch.pth'.format(save_dir, experiment.name)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    torch.save({'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()},
               out_model)

    return out_model

