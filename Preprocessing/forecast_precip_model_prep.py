"""
Script to process the precipitation data from ensemble forecast
to be used by Random Forest model
"""

import numpy as np
import os
import pandas as pd
from PIL import Image
import re


district_dict = {
    'Bhojpur': 1, 'Dhankuta': 2, 'Ilam': 3, 'Jhapa': 4, 'Khotang': 5, 'Morang': 6, 'Okhaldhunga': 7,
    'Panchthar': 8, 'Sankhuwasabha': 9, 'Solukhumbu': 10, 'Sunsari': 11, 'Taplejung': 12, 'Terhathum': 13,
    'Udayapur': 14, 'Bara': 15, 'Dhanusha': 16, 'Mahottari': 17, 'Parsa': 18, 'Rautahat': 19, 'Saptari': 20,
    'Sarlahi': 21, 'Siraha': 22, 'Bhaktapur': 23, 'Chitawan': 24, 'Dhading': 25, 'Dolakha': 26,
    'Kabhrepalanchok': 27, 'Kathmandu': 28, 'Lalitpur': 29, 'Makawanpur': 30, 'Nuwakot': 31, 'Ramechhap': 32,
    'Rasuwa': 33, 'Sindhuli': 34, 'Sindhupalchok': 35, 'Baglung': 36, 'Gorkha': 37, 'Kaski': 38, 'Lamjung': 39,
    'Manang': 40, 'Mustang': 41, 'Myagdi': 42, 'Nawalparasi_E': 43, 'Parbat': 44, 'Syangja': 45, 'Tanahu': 46,
    'Arghakhanchi': 47, 'Banke': 48, 'Bardiya': 49, 'Dang': 50, 'Gulmi': 51, 'Kapilbastu': 52, 'Palpa': 53,
    'Nawalparasi_W': 54, 'Pyuthan': 55, 'Rolpa': 56, 'Rukum East': 57, 'Rupandehi': 58, 'Dailekh': 59, 'Dolpa': 60,
    'Humla': 61, 'Jajarkot': 62, 'Jumla': 63, 'Kalikot': 64, 'Mugu': 65, 'Rukum West': 66, 'Salyan': 67,
    'Surkhet': 68, 'Achham': 78, 'Baitadi': 70, 'Bajhang': 71, 'Bajura': 72, 'Dadeldhura': 73, 'Darchula': 74,
    'Doti': 75, 'Kailali': 76, 'Kanchanpur': 77
}

def convert_npy_to_precip(nepal_im, data_dir, precip_source, results_dir):
    """
    Precipitation npy to df per district of average daily precipitation
    """
    for key in district_dict:
        print('Processing for district: {}'.format(key))
        nepal_arr = np.array(nepal_im)
        nepal_arr[nepal_arr != district_dict[key]] = np.nan

        date_list = []
        avg_precip_list = []
        for f in os.listdir(data_dir):
            if f == '.DS_Store':
                continue
            if precip_source == 'GPM':
                result = re.search('GPM_(.*).npy', f)
            else:
                result = re.search('doy_(.*).npy', f)
            try:
                date_str = result.group(1)
            except AttributeError:
                print('Missing precip for date {}, skipping'.format(date_str))
                continue
            print('Processing for date: {}'.format(date_str))
            try:
                precip_arr = np.load('{}/{}'.format(data_dir, f))
                precip_dist = precip_arr * nepal_arr
                avg_precip = np.nanmean(precip_dist)
                date_list.append(date_str)
                avg_precip_list.append(avg_precip)
            except OSError:
                print('Missing precip for date {}, skipping'.format(date_str))
                continue

        df = pd.DataFrame()
        df['doy'] = date_list
        df['avg_precip'] = avg_precip_list
        df_sorted = df.sort_values(by=['doy'])
        df_sorted['District'] = key
        df_sorted = df_sorted.reset_index()

        if not precip_source == 'GPM':
            savename = 'subseasonal_precipitation'
        else:
            savename = 'GPM'

        df_sorted.to_csv('{}/{}_{}.csv'.format(results_dir, key, savename))


# Load in the things we need
root_dir = '/Users/kelseydoerksen/Desktop/Nepal_Landslides_Forecasting_Project/Monsoon2024_Prep'
forecast_model = 'ecmwf'
ensemble_member = '1'

if forecast_model == 'ecmwf':
    subseasonal_precip_dir = '{}/PrecipitationModel_Forecast_Data/Subseasonal/{}'.format(
        root_dir, forecast_model)
    savedir = '{}/PrecipitationModel_Forecast_Data/Subseasonal/{}/DistrictLevel'.format(
        root_dir, forecast_model, ensemble_member)
else:
    subseasonal_precip_dir = '{}/PrecipitationModel_Forecast_Data/Subseasonal/{}/ensemble_member_{}'.format(
        root_dir, forecast_model, ensemble_member)
    savedir = '{}/PrecipitationModel_Forecast_Data/Subseasonal/{}/DistrictLevel/ensemble_member_{}'.format(
        root_dir, forecast_model, ensemble_member)

if not os.path.exists(subseasonal_precip_dir):
    os.makedirs(subseasonal_precip_dir)


nepal_tiff = Image.open('{}/District_Labels.tif'.format(root_dir))
convert_npy_to_precip(nepal_tiff, subseasonal_precip_dir, forecast_model, savedir)

'''
if forecast_model == 'GPM':
    subseasonal_precip_dir = '{}/GPM_Mean_Pixelwise'.format(root_dir)
    savedir = '{}/GPM_Mean_Pixelwise/Subseasonal/DistrictLevel'.format(root_dir)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    nepal_tiff = Image.open('{}/District_Labels.tif'.format(root_dir))
    convert_npy_to_precip(nepal_tiff, subseasonal_precip_dir, forecast_model, savedir)
else:
'''