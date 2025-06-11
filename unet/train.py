"""
Training script for UNet
"""


from dataset import *
from metrics import *
import numpy as np
from model import *
from losses import *
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
                training_loss,
                opt,
                val_percent,
                weight_decay: float = 0.00001,
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

    threshold = 0.1

    # --- Setting up optimizer
    if opt == 'rms':
        optimizer = optim.RMSprop(model.parameters(),
                                  lr=learning_rate, weight_decay=weight_decay)

    if opt == 'adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=learning_rate, weight_decay=weight_decay)

    # Setting up loss
    if training_loss == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    if training_loss == 'bce_pos_weight':
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.3]))    # penalizes false positives


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
        epoch_pct_cov_precision = 0
        epoch_pct_cov_recall = 0

        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            # Zero gradients for every batch
            optimizer.zero_grad()  # if set_to_none=True, sets gradients of all optimized torch.Tensors to None, will have a lower memory footprint, can modestly improve performance

            outputs = model(inputs)                 # predict on input
            if training_loss == 'tversky':
                loss = tversky_loss(outputs, labels)
            else:
                loss = criterion(outputs, labels)       # Calculate loss

            # Apply sigmoid for probabilities for precision recall
            outputs_probs = torch.sigmoid(outputs)

            thr_precision, thr_recall = precision_recall_threshold(labels, outputs_probs, threshold, district_masks)
            pct_cov_precision, pct_cov_recall = precision_and_recall_threshold_pct_cov(labels, outputs_probs, threshold, district_masks)

            grad_scaler.scale(loss).backward()      # Compute partial derivative of the output f with respect to each of the input variables
            grad_scaler.step(optimizer)             # Updates value of parameters according to strategy
            grad_scaler.update()

            global_step += 1
            epoch_loss += loss.item()
            epoch_thr_precision += thr_precision
            epoch_thr_recall += thr_recall
            epoch_pct_cov_precision += pct_cov_precision
            epoch_pct_cov_recall += pct_cov_recall

        experiment.log({
            'train BCE loss': epoch_loss/len(train_loader),
            'train Precision': epoch_thr_precision/len(train_loader),
            'train Recall': epoch_thr_recall/ len(train_loader),
            'train Precision pct cov': epoch_pct_cov_precision/len(train_loader),
            'train Recall pct cov': epoch_pct_cov_recall/len(train_loader),
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
        running_pct_cov_precision = 0
        running_pct_cov_recall = 0
        with torch.no_grad():
            for k, vdata in enumerate(val_loader):
                vinputs, vlabels = vdata
                vinputs, vlabels = vinputs.to(device), vlabels.to(device)

                voutputs = model(vinputs)

                vloss = criterion(voutputs, vlabels)

                # Apply sigmoid for probabilities
                voutputs_probs = torch.sigmoid(voutputs)

                # Calculating precision recall
                v_prec, v_rec = precision_recall_threshold(vlabels, voutputs_probs, threshold, district_masks)
                v_pct_cov_precision, v_pct_cov_recall = precision_and_recall_threshold_pct_cov(vlabels, voutputs_probs,
                                                                                           threshold, district_masks)

                running_vloss += vloss
                running_thr_recall += v_rec
                running_thr_precision += v_prec
                running_pct_cov_precision += v_pct_cov_precision
                running_pct_cov_recall += v_pct_cov_recall

        avg_vloss = running_vloss / len(val_loader)
        avg_prec = running_thr_precision / len(val_loader)
        avg_rec = running_thr_precision / len(val_loader)
        avg_pct_cov_prec = running_pct_cov_precision / len(val_loader)
        avg_pct_cov_recall = running_pct_cov_recall / len(val_loader)


        logging.info('Validation BCE score: {}'.format(avg_vloss))
        try:
            experiment.log({
                'learning rate': optimizer.param_groups[0]['lr'],
                'validation BCE loss': avg_vloss,
                'validation Precision': avg_prec,
                'validation Recall': avg_rec,
                'validation Precision pct cov': avg_pct_cov_prec,
                'valiation Recall pct cov': avg_pct_cov_recall,
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

    # Save final model with the weights and architecture itself
    torch.save(model, "{}/pretrained_model.pth".format(save_dir))

    return out_model

