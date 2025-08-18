"""
Further analysis on overfit experiments to test model

Psuedo code:
Per sample:
Per district in groundtruth, check how many (if any) pixels are overlapped by the prediction via:
matching_ones = np.sum((pred == 1) & (label == 1))
Turn this into a percent coverage value based on the total number of pixels in the district
plot this and see what it looks like from a geojson map where the colours represent the percent covered by the prediction
Look at different thresholds for getting the final prediction from
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torch import tensor
from torchmetrics.classification import BinaryJaccardIndex



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

"""
def pr_generation_coverage(y_true, y_pred, threshold, d_masks):
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
    y_pred = (y_pred >= threshold)*1

    total_landslides = 0
    no_pred_count = 0
    for i in range(len(y_pred)):
        non_landslide_districts = d_masks.copy()  # copy of landslides dict to manipulate
        dummy_pred = np.copy(y_pred)  # copy of y_pred to manipulate

        # Get what districts are in label to compare
        landslide_districts = []
        landslides = np.where(y_true == 1)


        import ipdb
        ipdb.set_trace()

        points = []
        for k in range(len(landslides[0])):
            points.append([landslides[0][k], landslides[1][k]])
        for district in d_masks:
            if all(item in points for item in d_masks[district]):
                landslide_districts.append(d_masks[district])
                non_landslide_districts.pop(district)

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

    if threshold == 0.1:
        plt.plot(f1_list)
        plt.title('F1 Score timeseries for threshold 0.1')
        plt.xlabel('Days after April 1 2023')
        plt.ylabel('F1-Score')
        plt.show()

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
    
"""

def intersection_calc(groundtruth, prediction, district_masks):
    """
    Get the IoU for the Districts
    """
    landslide_districts = []
    landslide_district_names = []
    non_landslide_districts = []
    non_landslide_district_names = []
    metric = BinaryJaccardIndex()
    gt_torch = torch.from_numpy(groundtruth)

    threshold_prediction = (prediction >= 0.1)*1

    for district in district_masks.keys():
        district_torch = torch.from_numpy(district_masks[district])
        if metric(district_torch, gt_torch) > 0:
            landslide_districts.append(district_masks[district])
            landslide_district_names.append(district)
        else:
            non_landslide_districts.append(district_masks[district])
            non_landslide_district_names.append(district)

    tp_count = 0
    fn_count = 0
    # Let's get the true positives and false negatives
    print('Calculating True Positives...')
    for i in range(len(landslide_districts)):
        predicted_in_district = threshold_prediction[landslide_districts[i] == 1]
        total_gt_pixels = np.sum(landslide_districts[i])
        coverage = np.sum(predicted_in_district == 1) / total_gt_pixels
        print('Coverage for landslide', landslide_district_names[i], ':', coverage)
        if coverage >= 0.5:
            tp_count += 1
        else:
            fn_count +=1

    # Let's get false positives
    fp_count = 0
    print('Calculating False Positives...')
    for i in range(len(non_landslide_districts)):
        predicted_in_district = threshold_prediction[non_landslide_districts[i] == 1]
        total_gt_pixels = np.sum(non_landslide_districts[i])
        coverage = np.sum(predicted_in_district == 1) / total_gt_pixels
        print('Coverage for landslide', non_landslide_district_names[i], ':', coverage)
        if coverage >= 0.5:
            fp_count += 1

    precision = tp_count / (tp_count + fp_count)
    recall = tp_count / (tp_count + fn_count)
    f1 = 2 * (precision * recall) / (precision + recall)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F1:', f1)



if __name__ == '__main__':
    # Hardcoding things for now

    root_dir = '/Volumes/PRO-G40/landslides/Nepal_Landslides_Forecasting_Project/Monsoon2024_Prep'
    district_masks = generate_district_masks('{}/District_Labels.tif'.format(root_dir))

    # Load predictions from overfit dir; run is cerulean-sky-1973
    overfit_dir = '/Volumes/PRO-G40/landslides/Nepal_Landslides_Forecasting_Project/Monsoon2024_Prep/overfit_dir'
    # Load in last epoch prediction
    pred = np.load('{}/cerulean-sky-1973_pred_999.npy'.format(overfit_dir))[0,0,:,:]
    gt = np.load('{}/label_2017-06-29.npy'.format(overfit_dir))

    #pr_generation_coverage(gt, pred, 0.1, district_masks)

    intersection_calc(gt, pred, district_masks)
















