"""
Pipeline script to:
1. Train and validate model via train.py
2. Run predictions on test set via predict.py
3. Plot results using analysis/plotting_results.py
"""

import torch
import argparse
import os
from train import *
from predict import *
import wandb
from utils import *
from model import models, unet_modules
from dataset import *
import logging


def get_args():
    parser = argparse.ArgumentParser(description='Running UNet Pipeline on Landslide Dataset')
    parser.add_argument('--epochs', '-e',  type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', '-l', type=float, default=0.001, help='Learning rate', dest='lr')
    parser.add_argument('--optimizer', '-o', type=str, default='adam', help='Model optimizer')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')
    parser.add_argument('--test-year', '-t', type=str, help='Test year for analysis (sets out of training)')
    parser.add_argument('--seed', help='Seed to set to make model deterministic',
                        required=True)
    parser.add_argument('--root_dir', help='Specify root directory',
                        required=True)
    parser.add_argument('--save_dir', help='Specify root save directory',
                        required=True),
    parser.add_argument('--val_percent', help='Validation percentage',
                        required=True)
    parser.add_argument('--ensemble', help='Ensemble Model. Must be one of KMA, NCEP, UKMO',
                        required=True)
    parser.add_argument('--ensemble_member', help='Ensemble Member.',
                        required=True)

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    seed = int(args.seed)
    root_dir = args.root_dir
    root_save_dir = args.save_dir
    ens = args.ensemble
    ens_num = args.ensemble_member

    make_deterministic(seed)

    # Initializing logging in wandb for experiment
    experiment = wandb.init(project='landslide-prediction', resume='allow', anonymous='must')
    experiment.config.update(
        dict(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr,
             val_percent=0.1, save_checkpoint=True,)
    )

    # --- Setting Directories
    sample_dir = '{}/UNet_Samples_14Day_GPM/{}/ensemble_{}'.format(root_dir, ens, ens_num)
    label_dir = '{}/Binary_Landslide_Labels_14day'.format(root_dir)

    # --- Making save directory
    save_dir = '{}/{}_ensemble_{}'.format(root_save_dir, ens, ens_num)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    unet = models.UNet(n_channels=32, n_classes=1)

    if torch.cuda.is_available():
        unet.cuda()
    #unet.to(device=device)

    # ---- Grabbing Training Data ----
    print('Grabbing training data...')
    landslide_train_dataset = LandslideDataset(sample_dir, label_dir, 'train')

    # --- Grabbing Testing Data ----
    print('Grabbing testing data...')
    landslide_test_dataset = LandslideDataset(sample_dir, label_dir, 'test')

    # --- Generating District Masks for Custom Metrics calculations
    # Generating district masks to use for the precision, recall
    district_masks = generate_district_masks('{}/Nepal_Rasterized_GPMRes_PerDistrict.tif'.format(root_dir))

    # We can locate now here where all the ones in the district are
    for district in district_masks:
        district_masks[district] = np.where(district_masks[district] == 1)

    # Want to change the list to be of pair type so we can compare more easily
    for district in district_masks:
        points = []
        for i in range(len(district_masks[district][0])):
            points.append([district_masks[district][0][i], district_masks[district][1][i]])
        district_masks[district] = points

    print('Training model...')
    trained_model = train_model(
        model=unet,
        device=device,
        dataset=landslide_train_dataset,
        save_dir=save_dir,
        experiment=experiment,
        val_percent=float(args.val_percent),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        opt = args.optimizer,
        save_checkpoint=True,
        district_masks = district_masks)

    print('Running Test set...')
    predict(trained_model, landslide_test_dataset, experiment, seed, save_dir, device=device,
            district_masks = district_masks)