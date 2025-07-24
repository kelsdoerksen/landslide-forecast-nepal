"""
Script to check the trained model inference on the training set to see how things are
Psuedo-code:
1. Load trained model
2. Load landslide training dataset
3. Run inference on training set and calculate F1 score
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
from torchvision.transforms import v2
import argparse
import gdal


def get_args():
    parser = argparse.ArgumentParser(description='Running UNet Pipeline on Landslide Dataset')
    parser.add_argument('--test_year', '-t', type=str, help='Test year for analysis (sets out of training)',
                        required=True)
    parser.add_argument('--root_dir', help='Specify root directory',
                        required=True)
    parser.add_argument('--model_type', help='Model type, unet or unet_mini',
                       required=True)
    parser.add_argument('--save_dir', help='Specify root save directory',
                        required=True),
    parser.add_argument('--trained_model', help='Pre-trained model to test')

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


if __name__ == '__main__':
    args = get_args()
    root_dir = args.root_dir
    model_type = args.model_type
    save_dir = args.save_dir
    trained_model = args.trained_model


    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # --- Setting Directories
    sample_dir = '{}/UNet_Samples_14Day_GPMv07/UKMO/ensemble_0'.format(root_dir)
    label_dir = '{}/Binary_Landslide_Labels_14day'.format(root_dir)

    # Generating district masks to use for the precision, recall
    district_masks = generate_district_masks('{}/District_Labels.tif'.format(root_dir))

    landslide_train_dataset = LandslideDataset(sample_dir, label_dir, 'train', 'norm', args.test_year,
                                                       save_dir)

    if 'unet_mini' in model_type:
        unetmodel = models.UNetMini(n_channels=32, n_classes=1, dropout=0)
        unetmodel.load_state_dict(torch.load('{}/{}.pth'.format(save_dir, trained_model))['state_dict'])
    else:
        unetmodel = models.UNet(n_channels=32, n_classes=1, dropout=0)
        unetmodel.load_state_dict(torch.load('{}/{}.pth'.format(save_dir, trained_model))['state_dict'])

    unetmodel.eval()

    data_loader = DataLoader(landslide_train_dataset, batch_size=1, shuffle=True)

    precision = []
    recall = []
    F1 = []
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # predict the mask
            outputs = unetmodel(inputs)

            # Apply sigmoid for predictions
            outputs_probs = torch.sigmoid(outputs)

            p, r, tp, fp, fn = precision_recall_threshold(labels, outputs_probs, 0.1, district_masks)
            precision.append(p)
            recall.append(r)
            f1 = 2 * (p * r) / (p + r)

    import ipdb
    ipdb.set_trace()



