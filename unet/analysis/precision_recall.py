"""
Script to generate precision, recall,
confusion matrix metrics to plot further
"""

import numpy as np
from PIL import Image
import argparse
from datetime import date, timedelta
import os
import pandas as pd


def get_args():
    parser = argparse.ArgumentParser(description='Running UNet Pipeline on Landslide Dataset')
    parser.add_argument('--root_dir', help='Specify root directory',
                        required=True)
    parser.add_argument('--results_dir', help='Results directory',
                        required=True)

    return parser.parse_args()


def generate_district_masks(file_name):
    '''
    Create the masks for each district from Nepal raster
    '''
    # Load in Nepal District file
    im = Image.open('{}'.format(file_name))
    array = np.array(im)

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


def pr_generation(y_true, y_pred, threshold, d_masks):
    '''
    Custom Precision-Recall metric.
    Computes the precision over the batch using
    the threshold_value indicated
    :param: y_true: label
    :param: y_pred: model prediction
    :param: d_masks: dictionary of
    '''
    # Set true positive and false positive count to 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    threshold_value = threshold
    y_pred = y_pred[0]
    y_true = y_true[0]

    y_pred = (y_pred > threshold_value)*1

    total_landslides = 0
    for i in range(len(y_pred)):
        non_landslide_districts = d_masks.copy()  # copy of landslides dict to manipulate
        dummy_pred = np.copy(y_pred[i, 0, :, ])  # copy of y_pred to manipulate
        # Get what districts are in label
        district_pixels = []
        landslides = np.where(y_true[i, 0, :, :] == 1)
        points = []
        for k in range(len(landslides[0])):
            points.append([landslides[0][k], landslides[1][k]])
        for district in d_masks:
            if all(item in points for item in d_masks[district]):
                district_pixels.append(d_masks[district])
                non_landslide_districts.pop(district)

        total_overlap = 0
        for j in range(len(district_pixels)):
            # iterate through the list of points containing the landslide aka true_location
            true_location = district_pixels[j]
            overlap = 0
            for w in range(len(true_location)):
                if y_pred[i, 0, true_location[w][0], true_location[w][1]] == 1:
                    overlap += 1
                    # set the overlapped pixel to 0 and see if we have any left over for FP
                    dummy_pred[true_location[w][0], true_location[w][1]] = 0
            # check if at least one pixel overlapped district
            if overlap > 0:
                total_overlap += 1

        true_positives += total_overlap
        total_landslides += len(district_pixels)

        fp_count = 0
        # Check if any predictions don't overlap our districts
        if np.amax(dummy_pred) > 0:
            # We have FPs, check how many incorrect landslides we "predicted"
            for d in non_landslide_districts:
                for point in non_landslide_districts[d]:
                    if dummy_pred[point[0], point[1]] == 1:
                        fp_count = + 1
                        break
        false_positives += fp_count

    if total_landslides > true_positives:
        false_negatives = total_landslides - true_positives

    if false_positives == 0 and true_positives == 0:
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
        true_negatives = (77*len(y_true)) - total_landslides   # Number of Districts minus the landslides that actually occurred
        fpr = false_positives / (false_positives + true_negatives)

    F1 = (true_positives) / (true_positives + (0.5*(false_positives+false_negatives)))

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


def pr_generation_timeseries(y_true, y_pred, threshold, d_masks):
    '''
    Custom Precision-Recall metric.
    Computes the precision over the batch using
    the threshold_value indicated
    :param: y_true: label
    :param: y_pred: model prediction
    :param: d_masks: dictionary of
    '''
    # Set true positive and false positive count to 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    threshold_value = threshold
    y_pred = y_pred[0]
    y_true = y_true[0]

    y_pred = (y_pred > threshold_value)*1

    total_landslides = 0
    f1_score = []
    for i in range(len(y_pred)):
        non_landslide_districts = d_masks.copy()  # copy of landslides dict to manipulate
        dummy_pred = np.copy(y_pred[i, 0, :, ])  # copy of y_pred to manipulate
        # Get what districts are in label
        district_pixels = []
        landslides = np.where(y_true[i, 0, :, :] == 1)
        points = []
        for k in range(len(landslides[0])):
            points.append([landslides[0][k], landslides[1][k]])
        for district in d_masks:
            if all(item in points for item in d_masks[district]):
                district_pixels.append(d_masks[district])
                non_landslide_districts.pop(district)

        total_overlap = 0
        for j in range(len(district_pixels)):
            # iterate through the list of points containing the landslide aka true_location
            true_location = district_pixels[j]
            overlap = 0
            for w in range(len(true_location)):
                if y_pred[i, 0, true_location[w][0], true_location[w][1]] == 1:
                    overlap += 1
                    # set the overlapped pixel to 0 and see if we have any left over for FP
                    dummy_pred[true_location[w][0], true_location[w][1]] = 0
            # check if at least one pixel overlapped district
            if overlap > 0:
                total_overlap += 1

        true_positives += total_overlap
        total_landslides += len(district_pixels)

        fp_count = 0
        # Check if any predictions don't overlap our districts
        if np.amax(dummy_pred) > 0:
            # We have FPs, check how many incorrect landslides we "predicted"
            for d in non_landslide_districts:
                for point in non_landslide_districts[d]:
                    if dummy_pred[point[0], point[1]] == 1:
                        fp_count = + 1
                        break
        false_positives += fp_count

        if total_landslides > true_positives:
            false_negatives = total_landslides - true_positives

        try:
            F1 = (true_positives) / (true_positives + (0.5*(false_positives+false_negatives)))
        except ZeroDivisionError as e:
            F1 = 0
        f1_score.append(F1)

    return f1_score


if __name__ == '__main__':
    args = get_args()
    root_dir = args.root_dir
    results_dir = args.results_dir

    # Grab list of .npy groundtruth and predictions from results dir
    predictions = [f for f in os.listdir(results_dir) if 'pred' in f]
    predictions.sort()
    groundtruth = [f for f in os.listdir(results_dir) if 'groundtruth' in f]
    groundtruth.sort()

    # Load predictions and groundtruth to list
    prediction_arrays = []
    groundtruth_arrays = []
    for i in range(len(predictions)):
        pred = np.load('{}/{}'.format(results_dir, predictions[i]))
        gt = np.load('{}/{}'.format(results_dir, groundtruth[i]))
        for j in range(len(pred)):
            prediction_arrays.append(pred[j,:,:,:])
            groundtruth_arrays.append(gt[j,:,:,:])


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

    def timeseries_f1_plots(thr):
        """
        Generate time series of F1 score to see overtime
        """
        results_list = []
        results_list.append(pr_generation_timeseries(np.array([groundtruth_arrays]), np.array([prediction_arrays]),
                                          thr, district_masks))
        return results_list

    # Generate results per date
    def get_results_avg(t):
        results_list = []
        results_list.append(pr_generation(np.array([groundtruth_arrays]), np.array([prediction_arrays]),
                                              t, district_masks))
        results_df = pd.DataFrame(results_list)
        return results_df.mean()

    '''
    # Running for 
    thr = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    thr_results = []
    for t in thr:
        thr_results.append(get_results_avg(t))

    df = pd.DataFrame(thr_results)

    # Save to csv
    df.to_csv('{}/analysis_2023_results.csv'.format(results_dir))
    '''
    # Running for date
    f1_dates = timeseries_f1_plots(0.1)
    df_f1 = pd.DataFrame()
    df_f1['f1'] = f1_dates[0]

    with open('{}/test_dates.txt'.format(results_dir), 'r') as file:
        text = file.read()
    substrings = ['sample' + part.strip() for part in text.split('sample') if part.strip()]
    doys = [s.replace('sample_', '') for s in substrings]
    df_f1['doy'] = doys
    df_f1.to_csv('{}/2023_timeseries.csv'.format(results_dir))
