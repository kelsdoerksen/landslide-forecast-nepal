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
from osgeo import gdal
import matplotlib.pyplot as plt


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


def pr_generation(y_true_total, y_pred_total, threshold, d_masks, save_dir):
    '''
    Custom Precision-Recall metric.
    Computes the precision over the batch using
    the threshold_value indicated
    :param: y_true: label
    :param: y_pred: model prediction
    :param: d_masks: dictionary of
    '''

    f1_list = []
    p_list = []
    r_list = []
    fpr_list = []
    fnr_list = []

    # Set true positive and false positive count to 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    threshold_value = threshold

    no_landslide_gt = 0
    for m in range(len(y_true_total)):
        y_pred = y_pred_total[m]
        y_true = y_true_total[m]
        y_pred = (y_pred >= threshold)*1

        total_landslides = 0
        no_pred_count = 0
        for i in range(len(y_pred)):
            if np.amax(y_true[i,0,:,:]) == 0:
                print('No ground truth so lets skip')
                no_landslide_gt += 1
                continue
            else:
                non_landslide_districts = d_masks.copy()  # copy of landslides dict to manipulate
                dummy_pred = np.copy(y_pred[i, 0, :, :])  # copy of y_pred to manipulate
                # Get what districts are in label
                landslide_district_pixels = []
                landslides = np.where(y_true[i, 0, :, :] == 1)
                xcoords = landslides[0].tolist()
                ycoords = landslides[1].tolist()
                landslide_points = set(zip(xcoords, ycoords))

                for district, mask in d_masks.items():
                    district_loc = np.argwhere(mask == 1)
                    district_points = set((int(x), int(y)) for x, y in district_loc)
                    is_covered = district_points.issubset(landslide_points)
                    if is_covered:
                        landslide_district_pixels.append(d_masks[district])
                        non_landslide_districts.pop(district)

                total_overlap = 0
                for district_mask in landslide_district_pixels:
                    # Get coordinates of pixels that are part of this district
                    district_coords = np.argwhere(district_mask == 1)
                    overlap = 0
                    for x, y in district_coords:
                        if y_pred[i, 0, x, y] == 1:
                            overlap += 1
                            dummy_pred[x, y] = 0  # prevent double-counting for FP

                    if overlap > 0:
                        total_overlap += 1
                true_positives += total_overlap
                print('True positives: {}'.format(true_positives))
                total_landslides += len(landslide_district_pixels)

                fp_count = 0
                # Check if any predictions overlap districts that did not have a recorded landslide
                for district_mask in non_landslide_districts.values():
                    # Check for overlap between prediction and this non-landslide district
                    overlap_mask = np.logical_and(district_mask, dummy_pred)
                    if np.any(overlap_mask):
                        fp_count += 1

                false_positives += fp_count
                print('False positives: {}'.format(fp_count))

            if threshold == 0.1:
                # Get F1 for DOY
                landslides_doy = len(landslide_district_pixels)
                fp_doy = fp_count
                fn_doy = landslides_doy - total_overlap
                tp_doy = total_overlap
                tn_doy = 77 - fp_doy - fn_doy
                if landslides_doy == 0:
                    p_doy = np.nan
                    r_doy = np.nan
                    f1_doy = np.nan
                else:
                    try:
                        p_doy = tp_doy / (tp_doy + fp_doy)
                        r_doy = tp_doy / (tp_doy + fn_doy)
                        f1_doy = 2 * (p_doy * r_doy) / (p_doy + r_doy)
                    except ZeroDivisionError as e:
                        f1_doy = 0
                        p_doy = np.nan
                        r_doy = np.nan
                        f1_doy = np.nan
                    try:
                        fpr = fp_doy / (fp_doy + tn_doy)
                    except ZeroDivisionError as e:
                        fpr = 0
                    try:
                        fnr = fn_doy / (fn_doy + tp_doy)
                    except ZeroDivisionError as e:
                        fnr = 0
                f1_list.append(f1_doy)
                p_list.append(p_doy)
                r_list.append(r_doy)
                fpr_list.append(fpr)
                fnr_list.append(fnr)
                if tp_doy == 0 and fp_doy == 0:
                    #print('No landslides predicted')
                    no_pred_count = no_pred_count + 1

                # Plotting F1 and histogram to take a look at the distribution
                plt.plot(f1_list)
                plt.title('F1 Score')
                plt.ylabel('F1 Score')
                plt.xlabel('Sample')
                plt.savefig('{}/f1_plot.png'.format(save_dir))
                plt.close()

                plt.hist(f1_list, bins=10)
                plt.title('F1 Histogram')
                plt.ylabel('Count')
                plt.xlabel('F1 Score')
                plt.savefig('{}/f1_plot_hist.png'.format(save_dir))
                plt.close()


    if total_landslides >= true_positives:
        false_negatives = total_landslides - true_positives
    else:
        false_negatives = 0

    if false_positives == 0 and true_positives == 0:
        precision_ratio = 0
        recall_ratio = 0
    elif false_negatives == 0 and true_positives == 0:
        precision_ratio = 0
        recall_ratio = 0
    else:
        precision_ratio = true_positives / (true_positives + false_positives)
        recall_ratio = true_positives / (true_positives + false_negatives)

    if false_negatives == 0:
        fnr = 0
    else:
        fnr = false_negatives / (false_negatives + true_positives)

    if false_positives == 0:
        fpr = 0
    else:
        # Calculate total districts correctly
        sample_count = 0
        for t in range(len(y_true_total)):
            sample_count += y_true_total[t].shape[0]
        sample_count = sample_count - no_landslide_gt
        print('Sample count with landslides used for PR calculations is: {}'.format(sample_count))
        true_negatives = (77*sample_count) - false_negatives - true_positives - false_positives
        fpr = false_positives / (false_positives + true_negatives)
    try:
         F1 = 2 * (precision_ratio * recall_ratio) / (precision_ratio + recall_ratio)
    except ZeroDivisionError as e:
        F1 = 0

    results_dict = {}

    # Add all results to dictionary to return
    results_dict['threshold'] = threshold
    results_dict['precision'] = precision_ratio
    results_dict['recall'] = recall_ratio
    results_dict['TP'] = true_positives
    results_dict['FP'] = false_positives
    results_dict['FN'] = false_negatives
    results_dict['F1'] = F1
    results_dict['FNR'] = fnr
    results_dict['FPR'] = fpr

    return results_dict


if __name__ == '__main__':
    args = get_args()
    root_dir = args.root_dir
    model_type = args.model_type
    save_dir = args.save_dir
    trained_model = args.trained_model


    # Setup device
    device = torch.device("cpu")

    # --- Setting Directories
    sample_dir = '{}/UNet_Samples_14Day_GPMv07/UKMO/ensemble_0'.format(root_dir)
    label_dir = '{}/Binary_Landslide_Labels_14day'.format(root_dir)

    # Generating district masks to use for the precision, recall
    district_masks = generate_district_masks('{}/District_Labels.tif'.format(root_dir))

    landslide_train = LandslideDataset(sample_dir, label_dir, 'train', 'norm', args.test_year,
                                                       save_dir)

    n_channels = 32
    mean = torch.zeros(n_channels)
    std = torch.zeros(n_channels)
    global_min = torch.full((n_channels,), float('inf'))
    global_max = torch.full((n_channels,), float('-inf'))
    n_samples = 0

    print('Grabbing data and normalizing...')

    for images, _ in landslide_train:
        images = images.float().contiguous()
        # Reshape to (32, 6000)
        images_flat = images.view(n_channels, -1)

        # Calculate mean and std
        mean += images_flat.mean(dim=1)
        std += images_flat.std(dim=1)
        n_samples += 1

    mean /= n_samples
    std /= n_samples
    max_val = None
    min_val = None

    landslide_train_dataset = LandslideDataset(sample_dir, label_dir, 'train', 'norm', args.test_year,
                                               save_dir, mean=mean, std=std, max_val=global_max,
                                               min_val=global_min,
                                               norm='zscore')

    print('Loading model...')
    if 'unet_mini' in model_type:
        unetmodel = models.UNetMini(n_channels=32, n_classes=1, dropout=0)
    else:
        unetmodel = models.UNet(n_channels=32, n_classes=1, dropout=0)

    checkpoint = torch.load('{}/{}.pth'.format(save_dir, trained_model), map_location=device)
    unetmodel.load_state_dict(checkpoint['state_dict'])

    unetmodel.eval()

    data_loader = DataLoader(landslide_train_dataset, batch_size=32, shuffle=True)

    preds = []
    gt = []
    print('Running model inference on training set...')
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # predict the mask
            outputs = unetmodel(inputs)

            # Apply sigmoid for predictions
            outputs_probs = torch.sigmoid(outputs)

            preds.append(outputs_probs.cpu().numpy())
            gt.append(labels.cpu().numpy())

    '''
    for i in range(len(preds)):
        np.save('{}/trainset_inference_{}.npy'.format(save_dir, i), preds[i])
        np.save('{}/trainset_labels_{}.npy'.format(save_dir, i), gt[i])
    '''

    print('Getting the pr values...')
    pr = pr_generation(gt, preds, 0.1, district_masks, save_dir)
    print(pr)