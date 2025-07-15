"""
Aggregating CM metrics for final plotting
"""

import geopandas as gpd
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description='CM maps')
    parser.add_argument('--run_type', help='Run type for model setup to aggregate runs from',
                        required=True)

    return parser.parse_args()

runs = {'unet_mini_dice_2023': ['UKMO_ensemble_0_1m2yy0ry', 'UKMO_ensemble_0_4hs7ghh8', 'UKMO_ensemble_0_dw67nyyy',
                                'UKMO_ensemble_0_ixn4w8lf']}
root_dir = '/Volumes/PRO-G40/landslides/Nepal_Landslides_Forecasting_Project/Monsoon2024_Prep/Results/GPMv07/'

if __name__ == '__main__':
    args = get_args()
    run = args.run_type

    run_list = runs[run]

    df_list = []
    for r in run_list:
        df = pd.read_csv('{}/{}/confusion_matrix_{}.csv'.format(root_dir, r, r))
        df_info = df[['true_positive_count', 'false_positive_count', 'false_negative_count', 'true_negative_count',
                      'fpr', 'fnr', 'tpr', 'tnr', 'f1']]
        df_list.append(df_info)

    # Stack and get mean
    array = np.stack([df.values for df in df_list])
    mean_array = array.mean(axis=0)
    mean_df = pd.DataFrame(mean_array, columns=df_list[0].columns)
    mean_df = mean_df.astype({'true_positive_count': int, 'false_positive_count': int, 'false_negative_count': int,
                              'true_negative_count': int})

    # Add back District column
    mean_df['DISTRICT'] = df['DISTRICT']
    mean_df['PROVINCE'] = df['PROVINCE']

    mean_df.to_csv('/Users/kelseydoerksen/Desktop/Nature_Comms_Analysis_Plotting/{}_cm.csv'.format(run))

    # Load in Nepal gdf
    nepal_gdf = gpd.read_file('/Volumes/PRO-G40/landslides/Nepal_Landslides_Forecasting_Project/'
                              'Nature_Comms/Nepal_District_Boundaries.geojson')
    nepal_gdf['DISTRICT'] = nepal_gdf['DISTRICT'].replace('Rukum East', 'Rukum_E')
    nepal_gdf['DISTRICT'] = nepal_gdf['DISTRICT'].replace('Parasi', 'Nawalparasi_W')
    nepal_gdf['DISTRICT'] = nepal_gdf['DISTRICT'].replace('Rukum West', 'Rukum_W')
    nepal_gdf['DISTRICT'] = nepal_gdf['DISTRICT'].replace('Nawalpur', 'Nawalparasi_E')

    # Load in my df
    combined = nepal_gdf.merge(mean_df, on='DISTRICT')


    # Plotting ----
    # Get max val
    max_vals = {'fp': max(combined['false_positive_count']),
                'fn': max(combined['false_negative_count']),
                'tp': max(combined['true_positive_count']),
                'tn': max(combined['true_negative_count'])}
    # vmax = int(max_vals[max(max_vals, key=max_vals.get)])

    # Plotting TP, FP, FN, TN Counts
    fig, ax = plt.subplots(2, 4, layout='compressed')
    # Row 1 ----
    tp = combined.plot(ax=ax[0,0],column='true_positive_count', edgecolor='black', cmap='Reds', vmin=0, vmax=max_vals['tp'])
    ax[0,0].set_title('True Positives')
    fig.colorbar(tp.collections[0], ax=ax[0,0], label='Count')

    fp = combined.plot(ax=ax[0,1],column='false_positive_count', edgecolor='black', cmap='Blues', vmin=0, vmax=max_vals['fp'])
    ax[0,1].set_title('False Positives')
    ax[0,1].set_xticks([])
    ax[0,1].set_yticks([])
    fig.colorbar(fp.collections[0], ax=ax[0,1], label='Count')

    tpr = combined.plot(ax=ax[0, 2], column='tpr', edgecolor='black', linewidth=1, cmap='Reds', vmin=0, vmax=1)
    ax[0, 2].set_title('True Positive Rate')
    ax[0,2].set_xticks([])
    ax[0,2].set_yticks([])
    fig.colorbar(tpr.collections[0], ax=ax[0, 2], label='%')

    fpr = combined.plot(ax=ax[0, 3], column='fpr', edgecolor='black', linewidth=1, cmap='Blues', vmin=0, vmax=1)
    ax[0, 3].set_title('False Positive Rate')
    ax[0, 3].set_xticks([])
    ax[0, 3].set_yticks([])
    fig.colorbar(fpr.collections[0], ax=ax[0, 3], label='%')

    # Row 2 ----
    tn = combined.plot(ax=ax[1, 0], column='true_negative_count', edgecolor='black', cmap='Oranges', vmin=0,
                       vmax=max_vals['tn'])
    ax[1, 0].set_title('True Negatives')
    ax[1, 0].set_xticks([])
    ax[1, 0].set_yticks([])
    fig.colorbar(tn.collections[0], ax=ax[1, 0], label='Count')

    fn = combined.plot(ax=ax[1,1],column='false_negative_count', edgecolor='black', cmap='Greens', vmin=0, vmax=max_vals['fn'])
    ax[1,1].set_title('False Negatives')
    ax[1, 1].set_xticks([])
    ax[1, 1].set_yticks([])
    fig.colorbar(fn.collections[0], ax=ax[1,1], label='Count')

    fnr = combined.plot(ax=ax[1, 2], column='fnr', edgecolor='black', linewidth=1, cmap='Greens', vmin=0, vmax=1)
    ax[1, 2].set_title('False Negative Rate')
    ax[1, 2].set_xticks([])
    ax[1, 2].set_yticks([])
    fig.colorbar(fnr.collections[0], ax=ax[1, 2], label='%')

    f1 = combined.plot(ax=ax[1, 3], column='f1', edgecolor='black', linewidth=1, cmap='Reds', vmin=0, vmax=1)
    ax[1,3].set_title('F1')
    ax[1,3].set_xticks([])
    ax[1, 3].set_yticks([])
    fig.colorbar(f1.collections[0],  ax=ax[1,3], label='F1 Score')

    plt.show()



