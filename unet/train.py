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
from torch.utils.data import DataLoader, random_split, Subset
import wandb
from torch import optim
from pathlib import Path
from predict import *
from augmentation import *
from utils import *
import random
from torchvision.transforms import v2, Lambda


def train_binary_classification_model(model,
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
                district_masks = None,
                channel_drop=0,
                channel_drop_iter=1,
                cutmix_aug=False,
                cutmix_alpha=1.0,
                overfit=False):
    """
    Train model for binary classificaiton based on the masks and pooling
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Split dataset into training and validation
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().
                                      manual_seed(random.randint(0, 1000)))

    # --- DataLoaders
    # The DataLoader pulls instances of data from the Dataset, collects them in batches,
    # and returns them for consumption by your training loop.
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    if int(channel_drop) > 0:
        for i in range(channel_drop_iter):
            train_loader = drop_channels(train_loader, channel_drop, batch_size=32, split='train',
                                         wandb_experiment=experiment)
            val_loader = drop_channels(val_loader, channel_drop, batch_size=32, split='validation',
                                       wandb_experiment=experiment)
    if cutmix_aug:
        train_loader = cutmix(train_loader, alpha=cutmix_alpha, batch_size=32)
        val_loader = cutmix(val_loader, alpha=cutmix_alpha, batch_size=32)
        experiment.log({'CutMix alpha': cutmix_alpha})

    threshold = 0.2

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

    grad_scaler = torch.cuda.amp.GradScaler()
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

            # Printing some probabilities to see what is fucked up (if any)
            with torch.no_grad():
                probs = torch.sigmoid(district_logits)
                print("Predicted probabilities:", probs[0].cpu().numpy())
                print("Ground truth labels:", binary_labels[0].cpu().numpy())

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
        running_precision = 0
        running_recall = 0
        running_f1 = 0
        with torch.no_grad():
            for k, vdata in enumerate(val_loader):
                vinputs, vlabels = vdata
                vinputs, vlabels = vinputs.to(device), vlabels.to(device)

                # Get embeddings from input
                embeddings, district_logits = model(vinputs)

                # Get binary labels
                binary_labels = get_binary_label(vlabels, district_masks)
                binary_labels = binary_labels.to(device)
                district_logits = district_logits.to(device)

                vloss = criterion(district_logits.squeeze(2), binary_labels.float())  # Calculate loss

                # Probability conversion so I can do the other metric calculations
                vprecision, vrecall, vf1 = binary_classification_precision_recall(threshold, district_logits,
                                                                                  binary_labels)

                running_vloss += vloss
                running_recall += np.sum(vrecall)/len(vrecall)
                running_precision += np.sum(vprecision)/len(vprecision)
                running_f1 += np.sum(vf1)/len(vf1)

        avg_vloss = running_vloss / len(val_loader)
        avg_prec = running_precision / len(val_loader)
        avg_rec = running_recall / len(val_loader)
        avg_f1 = running_f1 / len(val_loader)

        logging.info('Validation loss score: {}'.format(avg_vloss))
        try:
            experiment.log({
                'learning rate': optimizer.param_groups[0]['lr'],
                'validation loss': avg_vloss,
                'validation Precision': avg_prec,
                'validation Recall': avg_rec,
                'validation F1': avg_f1,
                'validation Precision pct cov': 'N/A',
                'validation Recall pct cov': 'N/A',
                'step': global_step,
                'epoch': epoch,
                **histograms
            })
        except:
            pass

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
                district_masks = None,
                channel_drop=0,
                channel_drop_iter=1,
                cutmix_aug=False,
                cutmix_alpha=1.0,
                overfit=False
                ):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    if int(channel_drop) > 0:
        for i in range(channel_drop_iter):
            train_loader = drop_channels(train_loader, channel_drop, batch_size=32, split='train',
                                         wandb_experiment=experiment)
            val_loader = drop_channels(val_loader, channel_drop, batch_size=32, split='validation',
                                       wandb_experiment=experiment)
    if cutmix_aug:
        train_loader = cutmix(train_loader, alpha=cutmix_alpha, batch_size=32)
        val_loader = cutmix(val_loader, alpha=cutmix_alpha, batch_size=32)
        experiment.log({'CutMix alpha': cutmix_alpha})

    threshold = 0.1

    if overfit:
        # Hardcoding for now just to check things
        overfit_dir = '/Volumes/PRO-G40/landslides/Nepal_Landslides_Forecasting_Project/Monsoon2024_Prep/overfit_dir'
        files = os.listdir(overfit_dir)
        files.sort()
        sample_list = []
        label_list = []
        for file in files:
            if 'sample' in file:
                sample_list.append(np.load('{}/{}'.format(overfit_dir, file)))
            else:
                label_list.append(np.load('{}/{}'.format(overfit_dir, file)))

        sample_tensor = torch.Tensor(sample_list)
        label_tensor = torch.Tensor(label_list)

        overfit_dataset = TensorDataset(sample_tensor, label_tensor)

        # then normalize
        mean = torch.zeros(32)
        std = torch.zeros(32)
        n_samples = len(overfit_dataset)
        for images, _ in overfit_dataset:
            images = images.float().contiguous()
            images_flat = images.view(32, -1)

            # Calculate mean and std
            mean += images_flat.mean(dim=1)
            std += images_flat.std(dim=1)
            n_samples += 1

        mean /= n_samples
        std /= n_samples

        mean = mean.view(-1, 1, 1)
        std = std.view(-1, 1, 1)

        mean = mean.detach().cpu().numpy()
        std = std.detach().cpu().numpy()

        norm_sample_list = []
        for sample in sample_list:
            norm_sample = (sample - mean) / (std + 1e-8)
            norm_sample_list.append(norm_sample)

        norm_sample_tensor = torch.Tensor(norm_sample_list)
        label_tensor = label_tensor.unsqueeze(1)
        overfit_dataset_norm = TensorDataset(norm_sample_tensor, label_tensor)
        train_loader = DataLoader(overfit_dataset_norm, batch_size=3, shuffle=False)

    # --- Setting up optimizer
    if opt == 'rms':
        optimizer = optim.RMSprop(model.parameters(),
                                  lr=learning_rate, weight_decay=weight_decay)

    if opt == 'adam':
        optimizer = optim.Adam(model.parameters(),
                               lr=learning_rate, weight_decay=weight_decay)

    if opt =='adam_explr':
        optimizer = optim.Adam(model.parameters(),
                               lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)

    # Setting up loss
    if training_loss == 'bce':
        criterion = nn.BCEWithLogitsLoss()
    if training_loss == 'bce_pos_weight_01':
        weight = 0.1
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight], device=device))    # penalizes false positives
        experiment.log({'bce_pos_weight': weight})
    if training_loss == 'bce_pos_weight_02':
        weight = 0.2
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight], device=device))    # penalizes false positives
        experiment.log({'bce_pos_weight': weight})
    if training_loss == 'bce_pos_weight_03':
        weight = 0.3
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight], device=device))    # penalizes false positives
        experiment.log({'bce_pos_weight': weight})
    if training_loss == 'bce_pos_weight_04':
        weight = 0.4
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight], device=device))    # penalizes false positives
        experiment.log({'bce_pos_weight': weight})
    if training_loss == 'dice_bce':
        criterion = DiceBCELoss()
    if training_loss == 'dice_bce_w3':
        criterion = DiceWeightedBCE03Loss()
    if training_loss == 'bce_pos_weight_15':
        weight = 1.5
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight], device=device))  # penalizes false positives
        experiment.log({'bce_pos_weight': weight})
    if training_loss == 'bce_pos_weight_3':
        weight = 3
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight], device=device))  # penalizes false positives
        experiment.log({'bce_pos_weight': weight})


    # --- Setting up schedulers
    #scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)  # goal: minimize MSE score.
    grad_scaler = torch.cuda.amp.GradScaler()

    # Set to training mode
    model.train()

    global_step = 0
    epoch_number = 0
    for epoch in range(epochs):
        print('Training EPOCH {}:'.format(epoch_number))
        epoch_number += 1
        epoch_loss = 0
        epoch_thr_precision = 0
        epoch_thr_recall = 0
        epoch_pct_cov_precision = 0
        epoch_pct_cov_recall = 0
        tp_count = 0
        fp_count = 0
        fn_count = 0

        epoch_check = [100, 200, 300, 400, 500, 600, 700, 800, 900, 999]

        for i, data in enumerate(train_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # Zero gradients for every batch
            optimizer.zero_grad()  # if set_to_none=True, sets gradients of all optimized torch.Tensors to None, will have a lower memory footprint, can modestly improve performance

            outputs = model(inputs)                 # predict on input
            if training_loss == 'tversky':
                loss = tversky_loss(outputs, labels)
            elif training_loss == 'tversky_FN_penalize':
                loss = tversky_loss_penalize_fn(outputs, labels)
            elif training_loss == 'dice':
                loss = dice_loss(outputs, labels)
            elif training_loss == 'logcosh_dice':
                loss = logcosh_dice_loss(outputs, labels)
            else:
                loss = criterion(outputs, labels)       # Calculate loss

            # Apply sigmoid for probabilities for precision recall
            outputs_probs = torch.sigmoid(outputs)

            if epoch_number in epoch_check:
                if overfit:
                    overfit_check = outputs_probs.detach().numpy()
                    np.save('{}/{}_pred_{}.npy'.format(overfit_dir, experiment.name, epoch_number), overfit_check)

            thr_precision, thr_recall, tp, fp, fn = precision_recall_threshold(labels, outputs_probs, threshold, district_masks)
            #pct_cov_precision, pct_cov_recall = precision_and_recall_threshold_pct_cov(labels, outputs_probs, threshold, district_masks)

            grad_scaler.scale(loss).backward()      # Compute partial derivative of the output f with respect to each of the input variables
            grad_scaler.step(optimizer)             # Updates value of parameters according to strategy
            grad_scaler.update()

            global_step += 1
            epoch_loss += loss.item()
            epoch_thr_precision += thr_precision
            epoch_thr_recall += thr_recall
            #epoch_pct_cov_precision += pct_cov_precision
            #epoch_pct_cov_recall += pct_cov_recall

            tp_count = tp_count + tp
            fp_count = fp_count + fp
            fn_count = fn_count + fn

        experiment.log({
            'train loss': epoch_loss/len(train_loader),
            'train Precision': epoch_thr_precision/len(train_loader),
            'train Recall': epoch_thr_recall/ len(train_loader),
            'train Precision pct cov': 'N/A',
            'train Recall pct cov': 'N/A',
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

                if training_loss == 'tversky':
                    vloss = tversky_loss(voutputs, vlabels)
                elif training_loss == 'tversky_FN_penalize':
                    vloss = tversky_loss_penalize_fn(voutputs, vlabels)
                elif training_loss == 'dice':
                    vloss = dice_loss(voutputs, vlabels)
                elif training_loss == 'logcosh_dice':
                    vloss = logcosh_dice_loss(voutputs, vlabels)
                else:
                    vloss = criterion(voutputs, vlabels)  # Calculate loss

                # Apply sigmoid for probabilities
                voutputs_probs = torch.sigmoid(voutputs)

                # Calculating precision recall
                v_prec, v_rec, vtp, vfp, vfn = precision_recall_threshold(vlabels, voutputs_probs, threshold, district_masks)
                #v_pct_cov_precision, v_pct_cov_recall = precision_and_recall_threshold_pct_cov(vlabels, voutputs_probs, threshold, district_masks)

                running_vloss += vloss
                running_thr_recall += v_rec
                running_thr_precision += v_prec
                #running_pct_cov_precision += v_pct_cov_precision
                #running_pct_cov_recall += v_pct_cov_recall

        avg_vloss = running_vloss / len(val_loader)
        avg_prec = running_thr_precision / len(val_loader)
        avg_rec = running_thr_recall / len(val_loader)
        #avg_pct_cov_prec = running_pct_cov_precision / len(val_loader)
        #avg_pct_cov_recall = running_pct_cov_recall / len(val_loader)


        logging.info('Validation loss score: {}'.format(avg_vloss))
        try:
            experiment.log({
                'learning rate': optimizer.param_groups[0]['lr'],
                'validation loss': avg_vloss,
                'validation Precision': avg_prec,
                'validation Recall': avg_rec,
                'validation Precision pct cov': 'N/A',
                'validation Recall pct cov': 'N/A',
                'step': global_step,
                'epoch': epoch,
                **histograms
            })
        except:
            pass

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

