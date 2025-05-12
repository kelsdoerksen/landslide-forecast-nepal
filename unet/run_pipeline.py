"""
Pipeline script to:
1. Train and validate model via train.py
2. Run predictions on test set via predict.py
3. Plot results using analysis/plotting_results.py
"""

import torch
import argparse
import os
from train import train_model
from predict import predict
import wandb
from utils import *
from model import models, unet_modules
from dataset import *
import logging
from osgeo import gdal
import random


def get_args():
    parser = argparse.ArgumentParser(description='Running UNet Pipeline on Landslide Dataset')
    parser.add_argument('--epochs', '-e',  type=int, default=200, help='Number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', '-l', type=float, default=0.001, help='Learning rate', dest='lr')
    parser.add_argument('--optimizer', '-o', type=str, default='adam', help='Model optimizer')
    parser.add_argument('--classes', '-c', type=int, default=1, help='Number of classes')
    parser.add_argument('--test-year', '-t', type=str, help='Test year for analysis (sets out of training)')
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
    parser.add_argument('--tags', help='wandb tag',
                        required=True)
    parser.add_argument('--exp_type', help='experiment type; standard or monsoon_test',
                        required=True)

    return parser.parse_args()


def generate_district_masks(file_name):
    '''
    Create the masks for each district from Nepal raster
    '''
    # Load in Nepal District file
    ds = gdal.Open('{}'.format(file_name))
    array = ds.GetRasterBand(1).ReadAsArray()

    # Create dict to match pixel values
    district_dict = {'Bhojpur': 1, 'Dhankuta': 2, 'Ilam': 3, 'Jhapa': 4, 'Khotang': 5, 'Morang': 6, 'Okhaldhunga': 7,
    'Panchthar': 8, 'Sankhuwasabha': 9, 'Solukhumbu': 10, 'Sunsari': 11, 'Taplejung': 12, 'Terhathum': 13,
    'Udayapur': 14, 'Bara': 15, 'Dhanusha': 16, 'Mahottari': 17, 'Parsa': 18, 'Rautahat': 19, 'Saptari': 20,
    'Sarlahi': 21, 'Siraha': 22, 'Bhaktapur': 23, 'Chitawan': 24, 'Dhading': 25, 'Dolakha': 26,
    'Kabhrepalanchok': 27, 'Kathmandu': 28, 'Lalitpur': 29, 'Makawanpur': 30, 'Nuwakot': 31, 'Ramechhap': 32,
    'Rasuwa': 33, 'Sindhuli': 34, 'Sindhupalchok': 35, 'Baglung': 36, 'Gorkha': 37, 'Kaski': 38, 'Lamjung': 39,
    'Manang': 40, 'Mustang': 41, 'Myagdi': 42, 'Nawalparasi_W': 43, 'Parbat': 44, 'Syangja': 45, 'Tanahu': 46,
    'Arghakhanchi': 47, 'Banke': 48, 'Bardiya': 49, 'Dang': 50, 'Gulmi': 51, 'Kapilbastu': 52, 'Palpa': 53,
    'Nawalparasi_E': 54, 'Pyuthan': 55, 'Rolpa': 56, 'Rukum_E': 57, 'Rupandehi': 58, 'Dailekh': 59, 'Dolpa': 60,
    'Humla': 61, 'Jajarkot': 62, 'Jumla': 63, 'Kalikot': 64, 'Mugu': 65, 'Rukum_W': 66, 'Salyan': 67,
    'Surkhet': 68, 'Achham': 78, 'Baitadi': 70, 'Bajhang': 71, 'Bajura': 72, 'Dadeldhura': 73, 'Darchula': 74,
    'Doti': 75, 'Kailali': 76, 'Kanchanpur': 77}

    new_dict = {}
    for k, v in district_dict.items():
        # Only include District of interest
        new_array = array.copy()
        new_array[new_array != v] = 0
        # Bound between 0 and 1
        filtered_array = new_array.copy()
        # Landslide class set to 1
        filtered_array[filtered_array == v] = 1
        new_dict[k] = filtered_array

    return new_dict


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == '__main__':
    args = get_args()
    root_dir = args.root_dir
    root_save_dir = args.save_dir
    ens = args.ensemble
    ens_num = args.ensemble_member
    tag = args.tags

    # Initializing logging in wandb for experiment
    experiment = wandb.init(project='landslide-prediction', resume='allow', anonymous='must',
                            tags=["{}".format(tag)])
    experiment.config.update(
        dict(epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr,
             val_percent=0.1, save_checkpoint=True, exp_type=args.exp_type)
    )

    # --- Setting Directories
    sample_dir = '{}/UNet_Samples_14Day_GPMv07/{}/ensemble_{}'.format(root_dir, ens, ens_num)
    label_dir = '{}/Binary_Landslide_Labels_14day'.format(root_dir)

    # --- Making save directory
    save_dir = '{}/{}_ensemble_{}_{}'.format(root_save_dir, ens, ens_num, experiment.id)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # --- Generating District Masks for Custom Metrics calculations
    # Generating district masks to use for the precision, recall
    district_masks = generate_district_masks('{}/District_Labels.tif'.format(root_dir))

    # We can locate now here where all the ones in the district are
    for district in district_masks:
        district_masks[district] = np.where(district_masks[district] == 1)

    # Want to change the list to be of pair type so we can compare more easily
    for district in district_masks:
        points = []
        for i in range(len(district_masks[district][0])):
            points.append([district_masks[district][0][i], district_masks[district][1][i]])
        district_masks[district] = points

    # Check experiment type and run the things
    if args.exp_type == 'monsoon_test':
        unet = models.UNet(n_channels=32, n_classes=1)
        set_seed(random.randint(0, 1000))

        print('Testing pre-trained model on latest Monsoon season...')
        model_path = '{}/Results/GPMv07/UKMO_ensemble_0_3s5mrfx1/cosmic-shadow-424_last_epoch.pth'.format(root_dir)         # Hardcoding for now until I do some investigating
        unet = torch.load(model_path, weights_only=False)

        sample_dir = '{}/2024_Season_Retro/UNet_Samples_14Day_GPMv07/ecmwf/ensemble_0'.format(root_dir)          # We used ecmwf during the monsoon season of 2024
        label_dir = '{}/2024_Season_Retro/Binary_Landslide_Labels_14day'.format(root_dir)

        print('Grabbing testing data...')
        landslide_test_dataset = LandslideDataset(sample_dir, label_dir, 'monsoon_test', save_dir)

        print('Predicting on 2024 Monsoon season...')
        predict(unet, landslide_test_dataset, experiment, save_dir, device=device,
                district_masks=district_masks, exp_type = args.exp_type)
    else:
        unet = models.UNet(n_channels=32, n_classes=1)

        if torch.cuda.is_available():
            unet.cuda()
        #unet.to(device=device)

        set_seed(random.randint(0,1000))

        # ---- Grabbing Training Data ----
        print('Grabbing training data...')
        landslide_train_dataset = LandslideDataset(sample_dir, label_dir, 'train', save_dir)

        # --- Grabbing Testing Data ----
        print('Grabbing testing data...')
        landslide_test_dataset = LandslideDataset(sample_dir, label_dir, 'test', save_dir)

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
        predict(trained_model, landslide_test_dataset, experiment, save_dir, device=device,
                district_masks = district_masks, exp_type = args.exp_type)