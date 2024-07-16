"""
Generating maps from the test predictions
Quick notes on this:
Can use the District mask to code the locations of the
landslide/non-landslides. Would like to get for each time period
in the test set, a map of the probability of landslide as output by the model,
then make this into a nice gif
I have all the sampels for the districts over the same time period, so basically what I can
do is sort the dataframe by date, generate a map for each date, then make a gif
from all of those figures
Work with the District_
"""

from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Generating Confusion Matrix')
    parser.add_argument('--run', help='Wandb run name')
    parser.add_argument('--forecast_model', help='Precipitation Forecast Model used')
    parser.add_argument('--ens_member', help='Precipitation Forecast Model Ens member used')
    parser.add_argument('--hindcast_model', help='Precipitation Hindcast Model used')
    return parser.parse_args()


district_dict = {
    'Bhojpur': 1, 'Dhankuta': 2, 'Ilam': 3, 'Jhapa': 4, 'Khotang': 5, 'Morang': 6, 'Okhaldhunga': 7,
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
    'Doti': 75, 'Kailali': 76, 'Kanchanpur': 77
}


def generate_prediction_map(df, nepal_mask, save_dir):
    '''
    Generates a landslide prediction map per date of interest based on
    ML model output
    '''

    # Get list of unique dates in dataframe
    date_list = df['date'].unique()
    sorted_dates = sorted(date_list)

    # Get list of unique districts in dataframe
    district_list = df['district'].unique()

    if not os.path.exists('{}/prediction'.format(save_dir)):
        os.mkdir('{}/prediction'.format(save_dir))

    # Iterate through date list and make nepal array
    for date in sorted_dates:
        # subset date
        df_subset = df[df['date'] == date]
        nepal_arr = np.array(nepal_mask)
        nepal_arr[nepal_arr == 0] = np.nan
        for district in district_list:
            # subset district
            df_dist = df_subset[df_subset['district'] == district]
            pred_val = df_dist['model soft predictions']
            district_val = district_dict[district]
            nepal_arr[nepal_arr == district_val] = pred_val

        daterange_start = datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)
        daterange_end = daterange_start + timedelta(days=14)

        plt.imshow(nepal_arr, cmap='gist_heat')
        plt.clim(0, 1)
        cbar = plt.colorbar()
        cbar.set_label('ML Classifier Predicted Probability', labelpad=15)

        plt.tight_layout()
        plt.title('{}-{}'.format(daterange_start.strftime("%Y-%m-%d"),
                                                       daterange_end.strftime("%Y-%m-%d")))
        plt.savefig('{}/prediction/{}_prediction_gist_heat.png'.format(save_dir, date))
        plt.close()


def generate_precipitation_map(precip_df, nepal_mask, save_dir, ens_model):
    """
    Generate the max precipitation over the same period so I can compare these two side by side
    """
    # Get list of unique dates in dataframe
    date_list = precip_df['date'].unique()
    sorted_dates = sorted(date_list)

    # Get list of unique districts in dataframe
    district_list = precip_df['district'].unique()

    # Get limits of precip for the plotting
    max_precip = precip_df['{}_ens_1_precip_total_cumulative_precipitation'.format(ens_model)].max()
    mean_precip = precip_df['{}_ens_1_precip_total_cumulative_precipitation'.format(ens_model)].mean()

    if not os.path.exists('{}/precipitation'.format(save_dir)):
        os.mkdir('{}/precipitation'.format(save_dir))

    # Iterate through date list and make nepal array
    for date in sorted_dates:
        # subset date
        df_subset = precip_df[precip_df['date'] == date]
        nepal_arr = np.array(nepal_mask)
        nepal_arr[nepal_arr == 0] = np.nan
        for district in district_list:
            # subset district
            df_dist = df_subset[df_subset['district'] == district]
            total_precip = df_dist['{}_ens_1_precip_total_cumulative_precipitation'.format(ens_model)]
            district_val = district_dict[district]
            nepal_arr[nepal_arr == district_val] = total_precip

        daterange_start = datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)
        daterange_end = daterange_start + timedelta(days=14)

        plt.imshow(nepal_arr, cmap='Blues')
        plt.clim(250000, max_precip)
        cbar = plt.colorbar()
        cbar.set_label('Total cumulative precipitation (mm)', labelpad=15)
        plt.tight_layout()
        plt.title('{}-{}'.format(daterange_start.strftime("%Y-%m-%d"),
                                 daterange_end.strftime("%Y-%m-%d")))
        plt.savefig('{}/precipitation/{}_precipitation_Blues.png'.format(save_dir, date))
        plt.close()


if __name__ == '__main__':
    args = get_args()
    run_dir = args.run
    forecast_model = args.forecast_model
    ens_number = args.ens_member
    hindcast_model =args.hindcast_model

    root_dir = '/Users/kelseydoerksen/Desktop/Nepal_Landslides_Forecasting_Project/Monsoon2024_Prep'
    nepal_im = Image.open('{}/District_Labels.tif'.format(root_dir))
    feature_df = pd.read_csv('{}/LabelledData_{}/{}/ensemble_{}/2023_windowsize14_district.csv'.format(root_dir,
                                                                                                       hindcast_model,
                                                                                                       forecast_model,
                                                                                                       ens_number))
    results_dir = '{}/Results/{}'.format(root_dir, run_dir)
    prediction_df = pd.read_csv('{}/predictions_and_groundtruth.csv'.format(results_dir))
    # Generate maps
    generate_prediction_map(prediction_df, nepal_im, results_dir)
    generate_precipitation_map(feature_df, nepal_im, results_dir, forecast_model)



