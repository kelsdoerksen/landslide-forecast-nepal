"""
Script to generate CM as maps over Nepal, so we can look at
TP, FP, TN, FN for the varying disctricts from the UNet model
"""

import geopandas as gpd
import pickle
import matplotlib.pyplot as plt
import pandas as pd

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
             'KMA_ensemble_0_1hi2kon5/KMA_ensemble_0_1hi2kon5_prediction_landsliding_districts.pkl')

gt_file = ('/Volumes/PRO-G40/landslides/Nepal_Landslides_Forecasting_Project/Monsoon2024_Prep/Results/GPMv07/'
           'KMA_ensemble_0_1hi2kon5/KMA_ensemble_0_1hi2kon5_groundtruth_landsliding_districts.pkl')

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

# Plotting CM ----
# Get max val
max_vals = {'fp': max(combined['false_positive_count']),
            'fn': max(combined['false_negative_count']),
            'tp': max(combined['true_positive_count']),
            'tn': max(combined['true_negative_count'])}
#vmax = int(max_vals[max(max_vals, key=max_vals.get)])

fig, ax = plt.subplots(2, 2, figsize=(11, 11))
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

plt.show()



