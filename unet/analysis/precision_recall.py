"""
Script to generate precision, recall,
confusion matrix metrics to plot further
"""
import ipdb
import numpy as np
from PIL import Image
import argparse
from datetime import date, timedelta
import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt


def get_args():
    parser = argparse.ArgumentParser(description='Running UNet Pipeline on Landslide Dataset')
    parser.add_argument('--root_dir', help='Specify root directory',
                        required=True)
    parser.add_argument('--results_dir', help='Results directory',
                        required=True)
    parser.add_argument('--run_name', help='Name of run in results dir',
                        required=True)
    parser.add_argument('--gt_dir', help='Groundtruth directory')

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

    f1_list = []
    # Set true positive and false positive count to 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    threshold_value = threshold
    y_pred = y_pred[0,:,:,:,:]
    y_true = y_true[0,:,:,:,:]

    y_pred = (y_pred >= threshold_value)*1

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
                        fp_count += 1
                        break

        false_positives += fp_count

        if threshold == 0.1:
            # Get F1 for DOY
            landslides_doy = len(district_pixels)
            fp_doy = false_positives
            fn_doy = landslides_doy - total_overlap
            tp_doy = total_overlap
            tn_doy = 77 - fp_doy - fn_doy
            try:
                p_doy = tp_doy / (tp_doy + fp_doy)
                r_doy = tp_doy / (tp_doy + fn_doy)
                f1_doy = 2 * (p_doy * r_doy) / (p_doy + r_doy)
            except ZeroDivisionError as e:
                f1_doy = 0
            f1_list.append(f1_doy)

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
        true_negatives = (77*len(y_true)) - false_negatives - true_positives - false_positives
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


def pr_generation_threshold_pct_cov(y_true, y_pred, threshold, d_masks, pct_cov=0.2):
    '''
    Custom Precision-Recall metric.
    Computes the precision over the batch using the threshold_value indicated
    :param: y_true: label
    :param: y_pred: model prediction
    :param: d_masks: dictionary of districts
    :param: pct_cov: pct coverage of pixels in district to label as "landslide"
    '''

    # Convert y_pred to 0s and 1s based on threshold
    y_pred = y_pred[0, :, :, :, :]
    y_true = y_true[0,:,:,:,:]
    y_pred_t = (y_pred > threshold)*1

    total_landslides = 0
    f1_score = []
    precision_list = []
    recall_list = []
    true_positive_list = []
    false_positive_list = []
    false_negative_list = []
    total_landslides_list = []
    for i in range(len(y_pred_t)):
        # Set true positive and false positive count to 0
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        total_landslides = 0

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
            if overlap >= int(len(true_location)*pct_cov):
                total_overlap += 1

        true_positives = total_overlap
        total_landslides = len(district_pixels)

        # Count false positives (districts with predictions but no landslide)
        fp_count = 0
        for district in non_landslide_districts:
            nonlandslide_size = len(non_landslide_districts[district])
            overlap_count = 0
            overlap_thr = int(pct_cov * nonlandslide_size)      # overlap threshold %
            for point in non_landslide_districts[district]:
                if dummy_pred[int(point[0]), int(point[1])] == 1:
                    overlap_count += 1
            if overlap_count >= overlap_thr:
                fp_count += 1
        false_positives = fp_count

        total_landslides_list.append(total_landslides)

        if total_landslides >= true_positives:
            false_negatives = total_landslides - true_positives
        else:
            false_negatives = 0

        true_positive_list.append(true_positives)
        false_positive_list.append(false_positives)
        false_negative_list.append(false_negatives)

    tp_total = sum(true_positive_list)
    fp_total = sum(false_positive_list)
    fn_total = sum(false_negative_list)

    precision = tp_total / (tp_total + fp_total)
    recall = tp_total / (tp_total + fn_total)

    F1 = (2 * precision * recall / (precision + recall)) if (precision + recall) != 0 else 0

    if fn_total == 0:
        fnr = 0
    else:
        fnr = fn_total / (fn_total + fp_total)

    if fp_total == 0:
        fpr = 0
    else:
        tn_total = (77*len(y_true)) - fn_total - tp_total - fp_total
        fpr = fp_total / (fp_total + tn_total)

    results_dict = {}

    # Add all results to dictionary to return
    results_dict['threshold'] = threshold
    results_dict['precision'] = precision
    results_dict['recall'] = recall
    results_dict['TP'] = tp_total
    results_dict['FP'] = fp_total
    results_dict['FN'] = fn_total
    results_dict['F1'] = F1
    results_dict['FNR'] = fnr
    results_dict['FPR'] = fpr

    return results_dict


def landslide_record_gen(y_true, y_pred, threshold, d_masks):
    '''
    Record the landslide predicted by model and true landslide locations for further plotting in future
    :param: y_true: label
    :param: y_pred: model prediction
    :param: d_masks: dictionary of
    '''

    threshold_value = threshold
    y_pred = y_pred[0]
    y_true = y_true[0]

    y_pred = (y_pred > threshold_value) * 1

    true_landsliding_districts = []
    predicted_landsliding_districts = []
    for i in range(len(y_pred)):
        groundtruth_landslides = []
        non_landslide_districts = d_masks.copy()  # copy of landslides dict to manipulate
        dummy_pred = np.copy(y_pred[i, 0, :, ])  # copy of y_pred to manipulate
        # Get what districts are in label
        district_pixels = []
        landslides = np.where(y_true[i, 0, :, :] == 1)
        points = []
        gt_landslide_dict = {}
        for k in range(len(landslides[0])):
            points.append([landslides[0][k], landslides[1][k]])
        for district in d_masks:
            if all(item in points for item in d_masks[district]):
                district_pixels.append(d_masks[district])
                non_landslide_districts.pop(district)
                groundtruth_landslides.append(district)
                gt_landslide_dict[district] = d_masks[district]

        all_districts = d_masks.copy()
        pred_landslides = []
        for j in all_districts:
            # iterate through the list of points containing the landslide aka true_location
            location = all_districts[j]
            for w in range(len(location)):
                if y_pred[i, 0, location[w][0], location[w][1]] == 1:
                    pred_landslides.append(j)
                    break

        true_landsliding_districts.append(groundtruth_landslides)
        predicted_landsliding_districts.append(pred_landslides)

    return true_landsliding_districts, predicted_landsliding_districts


if __name__ == '__main__':
    args = get_args()
    root_dir = args.root_dir
    results_dir = args.results_dir
    run = args.run_name
    gt_dir = args.gt_dir

    # --- Loading Data

    # Set up final results path with pointing to the run directory
    run_dir = '{}/{}'.format(results_dir, run)

    # Grab list of .npy groundtruth and predictions from results dir
    predictions = [f for f in os.listdir(run_dir) if 'pred' in f]
    predictions.sort()

    groundtruth = [f for f in os.listdir(gt_dir) if 'groundtruth' in f]
    if len(groundtruth) < 1:
        groundtruth = [f for f in os.listdir(gt_dir) if 'label' in f]
    #groundtruth = [f for f in os.listdir(results_dir) if 'groundtruth' in f]
    groundtruth.sort()

    # Load predictions and groundtruth to list
    prediction_arrays = []
    groundtruth_arrays = []
    for i in range(len(predictions)):
        pred = np.load('{}/{}'.format(run_dir, predictions[i]))
        gt = np.load('{}/{}'.format(gt_dir, groundtruth[i]))
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

    # Get list of test dates from file
    with open('{}/test_dates.txt'.format(gt_dir), 'r') as file:
        text = file.read()
    substrings = ['sample' + part.strip() for part in text.split('sample') if part.strip()]
    doys = [s.replace('sample_', '') for s in substrings]

    # --- Getting Precision-Recall for varying decision threshold using method "at least one overlap"
    # Running for all thresholds
    thr = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    thr_results = []
    for t in thr:
        pr = pr_generation(np.array([groundtruth_arrays]), np.array([prediction_arrays]),
                                              t, district_masks)
        thr_results.append(pr)
    df = pd.DataFrame(thr_results)

    # Save to csv
    df.to_csv('{}/{}/{}_analysis_results.csv'.format(results_dir, run, run))
    # ----------------------------

    # --- Get Precision-Recall for pct cov metric
    # Running for all decision thresholds
    pct_cov = 0.2
    thr = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1]
    thr_results_pct = []
    for t in thr:
        pr_pct = pr_generation_threshold_pct_cov(np.array([groundtruth_arrays]), np.array([prediction_arrays]),
                                                 t, district_masks, pct_cov)
        thr_results_pct.append(pr_pct)
    df_pct = pd.DataFrame(thr_results_pct)

    df_pct.to_csv('{}/{}/{}_analysis_pctcov_{}_results.csv'.format(results_dir, run, run, pct_cov))


    # ---  Get list of gt and pred landslide districts for future plotting
    gt_landslide_districts, pred_landslide_districts = landslide_record_gen(np.array([groundtruth_arrays]),
                                                                            np.array([prediction_arrays]), 0.1,
                                                                            district_masks)
    gt_landslide_districts_dict = {}
    pred_landslide_districts_dict = {}
    for d in range(len(doys)):
        gt_landslide_districts_dict[doys[d]] = gt_landslide_districts[d]
        pred_landslide_districts_dict[doys[d]] = pred_landslide_districts[d]

    # Save dictionaries to file
    with open('{}/{}/{}_groundtruth_landsliding_districts.pkl'.format(results_dir, run, run), 'wb') as fp:
        pickle.dump(gt_landslide_districts_dict, fp)
    with open('{}/{}/{}_prediction_landsliding_districts.pkl'.format(results_dir,run, run), 'wb') as fp:
        pickle.dump(pred_landslide_districts_dict, fp)
