"""
Custom metrics to calculate precision and recall
if at least one pixel is overlapping district where
landslide occurred
"""

import numpy as np
import torch


def precision_recall_threshold(y_true, y_pred, threshold, d_masks):
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

    #threshold_value = torch.Tensor([threshold])
    # Convert y_pred to 0s and 1s based on threshold
    y_pred_t = (y_pred > float(threshold)).float()

    # Convert y_pred, y_true to numpy arrays to be able to do some pythonic
    # manipulation
    y_pred = torch.Tensor.numpy(y_pred_t)
    y_true = torch.Tensor.numpy(y_true)

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

    if total_landslides > true_positives:
        false_negatives = total_landslides - true_positives

    if false_positives == 0 and true_positives == 0:
        precision_ratio = 0
        recall_ratio = 0
    elif false_negatives == 0 and true_positives == 0:
        precision_ratio = 0
        recall_ratio = 0
    else:
        precision_ratio = true_positives / (true_positives + false_positives)
        recall_ratio = true_positives / (true_positives + false_negatives)


    return precision_ratio, recall_ratio


def precision_and_recall_threshold_pct_cov(y_true, y_pred, threshold, d_masks, pct_cov=0.2):
    '''
    Custom Precision-Recall metric.
    Computes the precision over the batch using the threshold_value indicated
    :param: y_true: label
    :param: y_pred: model prediction
    :param: d_masks: dictionary of districts
    :param: pct_cov: pct coverage of pixels in district to label as "landslide"
    '''

    #threshold_value = torch.Tensor([threshold])
    # Convert y_pred to 0s and 1s based on threshold
    y_pred_t = (y_pred > float(threshold)).float()

    # Convert y_pred, y_true to numpy arrays to be able to do some pythonic
    # manipulation
    y_pred = torch.Tensor.numpy(y_pred_t)
    y_true = torch.Tensor.numpy(y_true)

    total_landslides = 0
    f1_score = []
    precision_list = []
    recall_list = []
    true_positive_list = []
    false_positive_list = []
    false_negative_list = []
    total_landslides_list = []
    for i in range(len(y_pred)):
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
            overlap_thr = int(pct_cov * nonlandslide_size)      # 20% overlap threshold
            for point in non_landslide_districts[district]:
                if dummy_pred[point[0], point[1]] == 1:
                    overlap_count += 1
            if overlap_count >= overlap_thr:
                fp_count += 1
        false_positives = fp_count

        total_landslides_list.append(total_landslides)

        if total_landslides > true_positives:
            false_negatives = total_landslides - true_positives

        true_positive_list.append(true_positives)
        false_positive_list.append(false_positives)
        false_negative_list.append(false_negatives)

    for tp, fp, fn in zip(true_positive_list, false_positive_list, false_negative_list):
        precision = tp / (tp + fp) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn) if (tp + fn) != 0 else 0
        precision_list.append(precision)
        recall_list.append(recall)
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) != 0 else 0
        f1_score.append(f1)

    precision_ratio = np.mean(precision_list)
    recall_ratio = np.mean(recall_list)

    return precision_ratio, recall_ratio



"""
def recall_threshold(y_true, y_pred, threshold, d_masks):
    '''
    Recall metric. Computes the recall over the batch using
    the threshold_value indicated
    Needs to return type <class 'tensorflow.python.framework.ops.Tensor'>
    '''
    true_positives = 0
    false_negatives = 0
    threshold_value = torch.Tensor([threshold])
    # Convert y_pred to 0s and 1s based on threshold
    y_pred_t = (y_pred > threshold_value).float()

    # Convert y_pred, y_true to numpy arrays to be able to do some pythonic
    # manipulation
    y_pred = torch.Tensor.numpy(y_pred_t)
    y_true = torch.Tensor.numpy(y_true)

    # Compare to true prediction with masks
    total_landslides = 0
    for i in range(len(y_pred)):
        # Get what districts are in label
        district_pixels = []
        landslides = np.where(y_true[i, 0, :, :] == 1)
        points = []
        for k in range(len(landslides[0])):
            points.append([landslides[0][k], landslides[1][k]])
        for district in d_masks:
            if all(item in points for item in d_masks[district]):
                district_pixels.append(d_masks[district])

        total_overlap = 0
        for j in range(len(district_pixels)):
            # iterate through the list of points containing the landslide aka true_location
            true_location = district_pixels[j]
            overlap = 0
            for m in range(len(true_location)):
                if y_pred[i, 0, true_location[m][0], true_location[m][1]] == 1:
                    overlap += 1
            # check if at least one pixel overlapped district
            if overlap > 0:
                total_overlap += 1

        true_positives += total_overlap
        total_landslides += len(district_pixels)

    if total_landslides > true_positives:
        false_negatives = total_landslides - true_positives

    if true_positives == 0 and false_negatives == 0:
        recall_ratio = 0
    else:
        recall_ratio = true_positives / (true_positives + false_negatives)
    return recall_ratio
"""