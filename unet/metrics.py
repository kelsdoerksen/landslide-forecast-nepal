"""
Custom metrics to calculate precision and recall
if at least one pixel is overlapping district where
landslide occurred
"""

import numpy as np
import torch
import copy
from sklearn.metrics import precision_score, recall_score, f1_score


def get_binary_label(labels, district_masks):
    """
    Retreive the binary labels (0,1) for landslide from the labels to do linear classification
    To do: once I have access to the internet and can look it up, go through the labels and then return
    in the shape (B, H, W) as is fed in
    """
    print('Getting binary labels for district')
    district_labels = []
    for i in range(labels.shape[0]):
        binary_labels = []
        for d in sorted(district_masks.keys()):
            print(d)
            mask_torch = torch.from_numpy(district_masks[d]).to(labels.device)  # shape (H, W)
            masked = labels[i,0,:,:] * mask_torch
            district_sum = masked.sum()
            if district_sum > 0:
                # Need to append this as a torch value that do the torch.stack part
                binary_labels.append(torch.tensor(1, device=labels.device))
            else:
                binary_labels.append(torch.tensor(0, device=labels.device))

        district_labels.append(binary_labels)

    total_labels = [torch.stack(inner_list) for inner_list in district_labels]
    # want to return shape (B, District_count)

    return torch.stack(total_labels)


def district_embedding_pooling(embeddings, district_masks):
    """
    Calculate the pooled embeddings for each district and return to make final classification
    embedding is size (B, C, H, W)
    Per each district, I will get a feature space of 32 embedding values that I use in final classifier to make the
    binary prediction of Y/N landslide
    """

    def masked_avg_pool(emb, mask):
        """
        Average pooling over masked district region
        Want to return the shape (B, C, H, W) same as the original embedding shape
        """
        mask_torch = torch.from_numpy(mask) # shape (H, W)
        masked = emb * mask_torch
        total = mask.sum()  # Number of pixels in the region
        if total == 0:
            return torch.zeros(emb.shape[0], device=emb.device)
        return masked.sum(dim=(1, 2)) / total   # this will return size 32 which represents the embedding from each channel

    # apply function and get pooled avg of embeddings over the region
    # Need to iterate through the batch correctly and then make back into shape B, C, H, W -> use torch.stack to do this
    district_embedding_list = []
    for i in range(embeddings.shape[0]):
        pooled_emb = []
        for district in sorted(district_masks.keys()):
            pooled_emb.append(masked_avg_pool(embeddings[i,:,:,:], district_masks[district]))
        pooled_emb = torch.stack(pooled_emb)
        district_embedding_list.append(pooled_emb)

    return torch.stack(district_embedding_list)


def binary_classification_precision_recall(threshold, logits, labels):
    """
    Take in the logits and labels and return the precision and recall
    """
    # logits to predictions
    probs = torch.sigmoid(logits)

    # apply threshold
    preds = (probs >= threshold).int()

    precision_list = []
    recall_list = []
    f1_list = []
    for i in range(preds.shape[0]):
        district_prediction = preds[i,:,0]
        label = labels[i,:]
        precision = precision_score(label.cpu(), district_prediction.cpu())
        recall = recall_score(label.cpu(), district_prediction.cpu())
        f1 = f1_score(label.cpu(), district_prediction.cpu())
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    return precision_list, recall_list, f1_list


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
    y_pred = y_pred_t.cpu().numpy()
    y_true = y_true.cpu().numpy()

    total_landslides = 0
    for i in range(len(y_pred)):
        non_landslide_districts = copy.deepcopy(d_masks)  # copy of landslides dict to manipulate
        dummy_pred = np.copy(y_pred[i, 0, :, :])  # copy of y_pred to manipulate
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

    if total_landslides >= true_positives:
        false_negatives = total_landslides - true_positives
    else:
        raise ValueError("More TPs than actual landslides – logic error?")


    if false_positives == 0 and true_positives == 0:
        precision_ratio = 0
        recall_ratio = 0
    elif false_negatives == 0 and true_positives == 0:
        precision_ratio = 0
        recall_ratio = 0
    else:
        precision_ratio = true_positives / (true_positives + false_positives)
        recall_ratio = true_positives / (true_positives + false_negatives)


    return precision_ratio, recall_ratio, true_positives, false_positives, false_negatives


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
    y_pred = y_pred_t.cpu().numpy()
    y_true = y_true.cpu().numpy()

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

        non_landslide_districts = copy.deepcopy(d_masks)  # copy of landslides dict to manipulate
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

        if total_landslides >= true_positives:
            false_negatives = total_landslides - true_positives
        else:
            raise ValueError("More TPs than actual landslides – logic error?")

        true_positive_list.append(true_positives)
        false_positive_list.append(false_positives)
        false_negative_list.append(false_negatives)

    tp_total = sum(true_positive_list)
    fp_total = sum(false_positive_list)
    fn_total = sum(false_negative_list)

    try:
        precision_ratio = tp_total / (tp_total + fp_total)
    except ZeroDivisionError as e:
        precision_ratio = 0
    try:
        recall_ratio = tp_total / (tp_total + fn_total)
    except ZeroDivisionError as e:
        recall_ratio = 0

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