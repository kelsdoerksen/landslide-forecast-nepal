"""
Generate maps that are geojson so they look a bit nicer for the figures
"""

import geopandas as gpd
import pickle
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


# Load in geojson
nepal_gdf = gpd.read_file('/Volumes/PRO-G40/landslides/Nepal_Landslides_Forecasting_Project/'
                          'Nature_Comms/Nepal_District_Boundaries.geojson')

# Load in predictions and groundtruth locations from results -> hardcoding results dir for now
pred_file = ('/Volumes/PRO-G40/landslides/Nepal_Landslides_Forecasting_Project/Monsoon2024_Prep/'
             'Results/GPMv07/UKMO_ensemble_0_z726vu1d/UKMO_ensemble_0_z726vu1d_prediction_landsliding_districts.pkl')

gt_file = ('/Volumes/PRO-G40/landslides/Nepal_Landslides_Forecasting_Project/Monsoon2024_Prep/'
             'Results/GPMv07/UKMO_ensemble_0_z726vu1d/UKMO_ensemble_0_z726vu1d_groundtruth_landsliding_districts.pkl')

with open(pred_file, 'rb') as fp:
    pred_data = pickle.load(fp)

with open(gt_file, 'rb') as fp:
    gt_data = pickle.load(fp)

# Adding things to the nepal gdf
nepal_gdf['predicted_landslide'] = 0
nepal_gdf['groundtruth_landslide'] = 0

# Rename so it matches how I have it in my data generation
nepal_gdf['DISTRICT'] = nepal_gdf['DISTRICT'].replace('Rukum East', 'Rukum_E')
nepal_gdf['DISTRICT'] = nepal_gdf['DISTRICT'].replace('Parasi', 'Nawalparasi_W')
nepal_gdf['DISTRICT'] = nepal_gdf['DISTRICT'].replace('Rukum West', 'Rukum_W')
nepal_gdf['DISTRICT'] = nepal_gdf['DISTRICT'].replace('Nawalpur', 'Nawalparasi_E')

cmap_landslide = mpl.colors.LinearSegmentedColormap.from_list("", ["black", "white"])
cmap_total = mpl.colors.LinearSegmentedColormap.from_list("", ["black", "white", "yellow"])

for k in pred_data.keys():
    gdf_copy = nepal_gdf.copy()
    pred_name_list = pred_data[k]
    gt_name_list = gt_data[k]
    gdf_copy['predicted_landslide'] = gdf_copy.apply(lambda row: row['predicted_landslide'] + 1 if row['DISTRICT'] in pred_name_list  else row['predicted_landslide'], axis=1)
    gdf_copy['groundtruth_landslide'] = gdf_copy.apply(
        lambda row: row['groundtruth_landslide'] + 1 if row['DISTRICT'] in gt_name_list else row['groundtruth_landslide'],
        axis=1)
    gdf_copy['overlap_areas'] = gdf_copy['predicted_landslide']+gdf_copy['groundtruth_landslide']
    gdf_copy['overlap_areas'] = gdf_copy['overlap_areas'].apply(lambda x: 0 if x < 2 else x)

    gdf_copy['missed_landslides'] = gdf_copy['groundtruth_landslide']-gdf_copy['predicted_landslide']
    gdf_copy['missed_landslides'] = gdf_copy['missed_landslides'].apply(lambda x: np.nan if x <1 else x)


    fig, ax = plt.subplots(1, 3, figsize=(11, 3))
    gdf_copy.plot(ax=ax[0],column='predicted_landslide', edgecolor='black', cmap=cmap_landslide)
    ax[0].set_title('Threshold Model Output')

    gdf_copy.plot(ax=ax[1],column='groundtruth_landslide', edgecolor='black', cmap=cmap_landslide)
    ax[1].set_title('Groundtruth landsliding Districts')

    gdf_copy.plot(ax=ax[2],column='overlap_areas', edgecolor='black', cmap=cmap_total)
    if len(gdf_copy.dropna()) > 0:
        gdf_copy.dropna().plot(ax=ax[2], column='missed_landslides', color='blue')
    ax[2].set_title('Overlapping Landsliding Districts')

    plt.suptitle('{} + 13 days outlook'.format(k))

    plt.savefig('/Volumes/PRO-G40/landslides/Nepal_Landslides_Forecasting_Project/Monsoon2024_Prep/'
                'Results/GPMv07/UKMO_ensemble_0_z726vu1d/{}_subplots.png'.format(k))

    plt.close()

