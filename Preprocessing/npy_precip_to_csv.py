"""
Script to process the precipitation data from ensemble forecast
to be used by Random Forest model
"""

import numpy as np
import os
import pandas as pd
from PIL import Image


root_dir = '/Volumes/PRO-G40/landslides/Nepal_Landslides_Forecasting_Project/Monsoon2024_Prep/PrecipitationModel_Forecast_Data'

district_dict = {
    'Bhojpur': 1, 'Dhankuta': 2, 'Ilam': 3, 'Jhapa': 4, 'Khotang': 5, 'Morang': 6, 'Okhaldhunga': 7,
    'Panchthar': 8, 'Sankhuwasabha': 9, 'Solukhumbu': 10, 'Sunsari': 11, 'Taplejung': 12, 'Terhathum': 13,
    'Udayapur': 14, 'Bara': 15, 'Dhanusha': 16, 'Mahottari': 17, 'Parsa': 18, 'Rautahat': 19, 'Saptari': 20,
    'Sarlahi': 21, 'Siraha': 22, 'Bhaktapur': 23, 'Chitawan': 24, 'Dhading': 25, 'Dolakha': 26,
    'Kabhrepalanchok': 27, 'Kathmandu': 28, 'Lalitpur': 29, 'Makawanpur': 30, 'Nuwakot': 31, 'Ramechhap': 32,
    'Rasuwa': 33, 'Sindhuli': 34, 'Sindhupalchok': 35, 'Baglung': 36, 'Gorkha': 37, 'Kaski': 38, 'Lamjung': 39,
    'Manang': 40, 'Mustang': 41, 'Myagdi': 42, 'Nawalpur': 43, 'Parbat': 44, 'Syangja': 45, 'Tanahu': 46,
    'Arghakhanchi': 47, 'Banke': 48, 'Bardiya': 49, 'Dang': 50, 'Gulmi': 51, 'Kapilbastu': 52, 'Palpa': 53,
    'Parasi': 54, 'Pyuthan': 55, 'Rolpa': 56, 'Rukum East': 57, 'Rupandehi': 58, 'Dailekh': 59, 'Dolpa': 60,
    'Humla': 61, 'Jajarkot': 62, 'Jumla': 63, 'Kalikot': 64, 'Mugu': 65, 'Rukum West': 66, 'Salyan': 67,
    'Surkhet': 68, 'Achham': 78, 'Baitadi': 70, 'Bajhang': 71, 'Bajura': 72, 'Dadeldhura': 73, 'Darchula': 74,
    'Doti': 75, 'Kailali': 76, 'Kanchanpur': 77
}

nepal_im = Image.open('/Volumes/PRO-G40/landslides/Nepal_Landslides_Forecasting_Project/Monsoon2024_Prep/District_Labels.tif')
nepal_arr = np.array(nepal_im)


def convert_npy_to_precip(precip_dir, forecast, ens_number):
    for key in district_dict:
        print('Processing for district: {}'.format(key))
        nepal_arr = np.array(nepal_im)
        nepal_arr[nepal_arr != district_dict[key]] = np.nan
        date_list = []
        avg_precip_list = []
        dist_list = []
        for f in os.listdir(precip_dir):
            if f == '.DS_Store':
                continue
            date_str = f[31:41] # update this
            print('Processing for date: {}'.format(date_str))
            precip_arr = np.load('{}/{}'.format(precip_dir, f))
            precip_dist = precip_arr * nepal_arr
            avg_precip = np.nanmean(precip_dist)
            date_list.append(date_str)
            avg_precip_list.append(avg_precip)
            dist_list.append(key)

        df = pd.DataFrame()
        df['doy'] = date_list
        df['avg_precip'] = avg_precip_list
        df['District'] = dist_list
        date_list = sorted(date_list)
        df_sorted = df.sort_values(by=['doy'])
        df_sorted.to_csv('{}/Subseasonal/{}/DistrictLevel/ensemble_member_{}/{}_subseasonal_precipitation.csv'.
                         format(root_dir, forecast, ens_number, key), index=False)


forecast = 'UKMO'
ens_members = [0, 1, 2, 3]
for ens_number in ens_members:
    precip_dir = '{}/Subseasonal/{}/ensemble_member_{}'.format(root_dir, forecast, ens_number)
    convert_npy_to_precip(precip_dir, forecast, ens_number)


#'precipitation_forecast_id0_doy_2019-12-31.npy'