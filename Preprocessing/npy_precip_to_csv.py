"""
Script to process the precipitation data from ensemble forecast
to be used by Random Forest model

Reads the per-day forecast .npy files in
Subseasonal/{model}/ensemble_member_{n}/ and writes one
{district}_subseasonal_precipitation.csv per district to that member's
DistrictLevel/ folder.

Averaging uses a true 0/1 district mask (avg_precip = mean(precip over the
district's pixels)), matching forecast_precip_model_prep.py and
hindcast_precip_model_prep.py. This replaces the previous behaviour that did
`precip_arr * nepal_arr`, which scaled avg_precip by the district's label ID
(District_Labels.tif stores IDs 1-78, not a 0/1 mask).
"""

import numpy as np
import os
import re
import pandas as pd
from PIL import Image

PROJECT_DIR = os.environ.get('LANDSLIDE_PROJECT_DIR', 'data')  # see README > Data


root_dir = PROJECT_DIR + '/Monsoon2024_Prep/PrecipitationModel_Forecast_Data'

# District names match the Bipad-portal spellings used downstream by
# generate_samples_hindcast_forecast.py (which reads these CSVs by name).
district_dict = {
    'Bhojpur': 1, 'Dhankuta': 2, 'Ilam': 3, 'Jhapa': 4, 'Khotang': 5, 'Morang': 6, 'Okhaldhunga': 7,
    'Panchthar': 8, 'Sankhuwasabha': 9, 'Solukhumbu': 10, 'Sunsari': 11, 'Taplejung': 12, 'Terhathum': 13,
    'Udayapur': 14, 'Bara': 15, 'Dhanusha': 16, 'Mahottari': 17, 'Parsa': 18, 'Rautahat': 19, 'Saptari': 20,
    'Sarlahi': 21, 'Siraha': 22, 'Bhaktapur': 23, 'Chitawan': 24, 'Dhading': 25, 'Dolakha': 26,
    'Kabhrepalanchok': 27, 'Kathmandu': 28, 'Lalitpur': 29, 'Makawanpur': 30, 'Nuwakot': 31, 'Ramechhap': 32,
    'Rasuwa': 33, 'Sindhuli': 34, 'Sindhupalchok': 35, 'Baglung': 36, 'Gorkha': 37, 'Kaski': 38, 'Lamjung': 39,
    'Manang': 40, 'Mustang': 41, 'Myagdi': 42, 'Nawalparasi_E': 43, 'Parbat': 44, 'Syangja': 45, 'Tanahu': 46,
    'Arghakhanchi': 47, 'Banke': 48, 'Bardiya': 49, 'Dang': 50, 'Gulmi': 51, 'Kapilbastu': 52, 'Palpa': 53,
    'Nawalparasi_W': 54, 'Pyuthan': 55, 'Rolpa': 56, 'Rukum_E': 57, 'Rupandehi': 58, 'Dailekh': 59, 'Dolpa': 60,
    'Humla': 61, 'Jajarkot': 62, 'Jumla': 63, 'Kalikot': 64, 'Mugu': 65, 'Rukum_W': 66, 'Salyan': 67,
    'Surkhet': 68, 'Achham': 78, 'Baitadi': 70, 'Bajhang': 71, 'Bajura': 72, 'Dadeldhura': 73, 'Darchula': 74,
    'Doti': 75, 'Kailali': 76, 'Kanchanpur': 77
}

nepal_im = Image.open(PROJECT_DIR + '/Monsoon2024_Prep/District_Labels.tif')
nepal_arr = np.array(nepal_im)

# Boolean mask per district (True where the pixel belongs to that district).
district_masks = {key: (nepal_arr == district_dict[key]) for key in district_dict}

# Files are named like 'precipitation_forecast_id0_doy_2019-12-31.npy'
date_re = re.compile(r'doy_(\d{4}-\d{2}-\d{2})')


def convert_npy_to_precip(precip_dir, forecast, ens_number):
    save_dir = '{}/Subseasonal/{}/DistrictLevel/ensemble_member_{}'.format(root_dir, forecast, ens_number)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    grid_shape = nepal_arr.shape
    series = {key: [] for key in district_dict}

    for f in sorted(os.listdir(precip_dir)):
        if not f.endswith('.npy'):
            continue
        result = date_re.search(f)
        if not result:
            print('Could not parse date from {}, skipping'.format(f))
            continue
        date_str = result.group(1)
        print('Processing for date: {}'.format(date_str))
        precip_arr = np.load('{}/{}'.format(precip_dir, f))
        if precip_arr.shape != grid_shape:
            print('Grid mismatch on {} ({}), skipping'.format(f, precip_arr.shape))
            continue
        for key, mask in district_masks.items():
            avg_precip = float(np.nanmean(precip_arr[mask])) if mask.any() else np.nan
            series[key].append((date_str, avg_precip))

    for key in district_dict:
        print('Writing CSV for district: {}'.format(key))
        df = pd.DataFrame(series[key], columns=['doy', 'avg_precip'])
        df['District'] = key
        df = df.drop_duplicates(subset=['doy'], keep='last')
        df_sorted = df.sort_values(by=['doy']).reset_index(drop=True)
        df_sorted.to_csv('{}/{}_subseasonal_precipitation.csv'.format(save_dir, key), index=False)


forecast = 'UKMO'
ens_members = [0, 1, 2, 3]
for ens_number in ens_members:
    precip_dir = '{}/Subseasonal/{}/ensemble_member_{}'.format(root_dir, forecast, ens_number)
    convert_npy_to_precip(precip_dir, forecast, ens_number)
