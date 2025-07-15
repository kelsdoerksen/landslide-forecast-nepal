"""
Script to generate CM as maps over Nepal, so we can look at
TP, FP, TN, FN for the varying disctricts from the UNet model
"""

import geopandas as gpd
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description='CM Maps per run')
    parser.add_argument('--results_dir', help='Results directory',
                        required=True)

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    results_dir = args.results_dir

    nepal_gdf = gpd.read_file('/Volumes/PRO-G40/landslides/Nepal_Landslides_Forecasting_Project/'
                              'Nature_Comms/Nepal_District_Boundaries.geojson')

    nepal_gdf['DISTRICT'] = nepal_gdf['DISTRICT'].replace('Rukum East', 'Rukum_E')
    nepal_gdf['DISTRICT'] = nepal_gdf['DISTRICT'].replace('Parasi', 'Nawalparasi_W')
    nepal_gdf['DISTRICT'] = nepal_gdf['DISTRICT'].replace('Rukum West', 'Rukum_W')
    nepal_gdf['DISTRICT'] = nepal_gdf['DISTRICT'].replace('Nawalpur', 'Nawalparasi_E')


    # Provided full list of districts
    all_districts_list = nepal_gdf['DISTRICT']

    # Prediction and groundtruth dictionaries
    # Load in predictions and groundtruth locations from results -> hardcoding results dir for now
    pred_file = ('/Volumes/PRO-G40/landslides/Nepal_Landslides_Forecasting_Project/Monsoon2024_Prep/Results/GPMv07/'
                 '{}/{}_prediction_landsliding_districts.pkl'.format(results_dir, results_dir))

    gt_file = ('/Volumes/PRO-G40/landslides/Nepal_Landslides_Forecasting_Project/Monsoon2024_Prep/Results/GPMv07/'
               '{}/{}_groundtruth_landsliding_districts.pkl'.format(results_dir, results_dir))

    with open(pred_file, 'rb') as fp:
        prediction = pickle.load(fp)

    with open(gt_file, 'rb') as fp:
        groundtruth = pickle.load(fp)

    # Initialize count dictionary
    results = {district: {'TP': 0, 'FP': 0, 'FN': 0, 'TN': 0} for district in all_districts_list}

    # Get all unique dates
    all_dates = set(prediction) | set(groundtruth)

    # Loop through dates
    for date in all_dates:
        pred_set = set(prediction.get(date, []))
        gt_set = set(groundtruth.get(date, []))

        for district in all_districts_list:
            in_pred = district in pred_set
            in_gt = district in gt_set

            if in_pred and in_gt:
                results[district]['TP'] += 1
            elif in_pred and not in_gt:
                results[district]['FP'] += 1
            elif not in_pred and in_gt:
                results[district]['FN'] += 1
            else:  # not in_pred and not in_gt
                results[district]['TN'] += 1

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(results, orient='index').reset_index()
    df.columns = ['DISTRICT', 'true_positive_count', 'false_positive_count', 'false_negative_count', 'true_negative_count']

    combined = nepal_gdf.merge(df, on='DISTRICT')

    # Adding rates
    combined['fpr'] = combined['false_positive_count'] / (
                combined['false_positive_count'] + combined['true_negative_count'])
    combined['fnr'] = combined['false_negative_count'] / (
                combined['false_negative_count'] + combined['true_positive_count'])
    combined['tpr'] = combined['true_positive_count'] / (
            combined['true_positive_count'] + combined['false_negative_count'])
    combined['tnr'] = combined['true_negative_count'] / (
            combined['true_negative_count'] + combined['false_positive_count'])

    # Adding F1
    combined['f1'] = 2 * combined['true_positive_count'] / (2 * combined['true_positive_count'] +
                                                            combined['false_positive_count'] +
                                                            combined['false_negative_count'])
    combined = combined.fillna(0)

    # Plotting CM ----
    # Get max val
    max_vals = {'fp': max(combined['false_positive_count']),
                'fn': max(combined['false_negative_count']),
                'tp': max(combined['true_positive_count']),
                'tn': max(combined['true_negative_count'])}
    #vmax = int(max_vals[max(max_vals, key=max_vals.get)])

    # Plotting TP, FP, FN, TN Counts
    '''
    fig, ax = plt.subplots(2, 2)
    tp = combined.plot(ax=ax[0,0],column='true_positive_count', edgecolor='black', cmap='Reds', vmin=0, vmax=max_vals['tp'])
    ax[0,0].set_title('True Positives')
    fig.colorbar(tp.collections[0], ax=ax[0,0], label='Count')

    fp = combined.plot(ax=ax[0,1],column='false_positive_count', edgecolor='black', cmap='Blues', vmin=0, vmax=max_vals['fp'])
    ax[0,1].set_title('False Positives')
    fig.colorbar(fp.collections[0], ax=ax[0,1], label='Count')

    fn = combined.plot(ax=ax[1,0],column='false_negative_count', edgecolor='black', cmap='Greens', vmin=0, vmax=max_vals['fn'])
    ax[1,0].set_title('False Negatives')
    fig.colorbar(fn.collections[0], ax=ax[1,0], label='Count')

    tn = combined.plot(ax=ax[1,1],column='true_negative_count', edgecolor='black', cmap='Oranges', vmin=0, vmax=max_vals['tn'])
    ax[1,1].set_title('True Negatives')
    fig.colorbar(tn.collections[0], ax=ax[1,1], label='Count')

    #plt.show()
    plt.close()

    # Plotting Rates
    fig, ax = plt.subplots(2, 2)
    tp = combined.plot(ax=ax[0, 0], column='tpr', edgecolor='black', linewidth=1, cmap='Reds', vmin=0, vmax=1)
    ax[0, 0].set_title('True Positive Rate')
    fig.colorbar(tp.collections[0], ax=ax[0, 0], label='%')

    fp = combined.plot(ax=ax[0, 1], column='fpr', edgecolor='black', linewidth=1, cmap='Blues', vmin=0, vmax=1)
    ax[0, 1].set_title('False Positive Rate')
    fig.colorbar(fp.collections[0], ax=ax[0, 1], label='%')

    fn = combined.plot(ax=ax[1, 0], column='fnr', edgecolor='black', linewidth=1, cmap='Greens', vmin=0, vmax=1)
    ax[1, 0].set_title('False Negative Rate')
    fig.colorbar(fn.collections[0], ax=ax[1, 0], label='%')

    tn = combined.plot(ax=ax[1, 1], column='tnr', edgecolor='black', linewidth=1, cmap='Oranges', vmin=0, vmax=1)
    ax[1, 1].set_title('True Negative Rate')
    fig.colorbar(tn.collections[0], ax=ax[1, 1], label='%')

    #plt.show()
    plt.close()

    # Plotting F1
    fig, ax = plt.subplots()
    f1 = combined.plot(column='f1', edgecolor='black', linewidth=1, cmap='Reds', vmin=0, vmax=1)
    ax.set_title('F1')
    fig.colorbar(f1.collections[0], label='F1 Score')
    plt.show()
    '''

    combined.to_csv('/Volumes/PRO-G40/landslides/Nepal_Landslides_Forecasting_Project/Monsoon2024_Prep/Results/GPMv07/'
                    '{}/confusion_matrix_{}.csv'.format(results_dir, results_dir))

    # Aggregating by location
    def f1_per_province(df):
        """
        Spatially aggregating F1 score of model
        """
        f1_per_province = []
        for i in range(1,8):
            subset = df[df['PROVINCE'] == i]
            f1_per_province.append(np.mean(subset['f1']))

    f1_prov = f1_per_province(combined)


