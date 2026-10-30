"""
Script to process the observed (hindcast) precipitation .npy files into the
per-district daily average precipitation CSVs used as LOOKBACK features by the
classifier's sample generator (generate_samples_hindcast_forecast.py).

Reads the per-day .npy files in {source}_Mean_Pixelwise/ (e.g.
GPMv07_2011-04-01.npy), masks per district with District_Labels.tif, and writes
{district}_{source}.csv to {source}_Mean_Pixelwise/Subseasonal/DistrictLevel/.

Averaging uses a true 0/1 district mask (avg_precip = mean(precip over the
district's pixels)), matching the corrected forecast pipeline in
forecast_precip_model_prep.py. This replaces the previous behaviour that scaled
avg_precip by the district's label ID (District_Labels.tif stores IDs 1-78).
The whole series is rebuilt from npy so all dates stay on the same scale.
"""

import numpy as np
import os
import re
import pandas as pd
from PIL import Image

PROJECT_DIR = os.environ.get('LANDSLIDE_PROJECT_DIR', 'data')  # see README > Data


root_dir = PROJECT_DIR + '/Monsoon2024_Prep'

# Hindcast sources to (re)build. npy files live directly in {source}_Mean_Pixelwise/
# named like '{source}_YYYY-MM-DD.npy'; output goes to that folder's Subseasonal/DistrictLevel.
SOURCES = ['GPMv07']

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


def build_district_masks(nepal_arr):
    """Boolean mask per district (True where the pixel belongs to that district)."""
    return {key: (nepal_arr == district_dict[key]) for key in district_dict}


def read_npy_series(npy_dir, source, district_masks):
    """
    Read the per-day .npy files ('{source}_YYYY-MM-DD.npy') and return a dict:
        district -> list of (date_str, avg_precip)   with avg_precip = mean(precip over district).
    """
    series = {key: [] for key in district_dict}
    date_re = re.compile(r'(\d{4}-\d{2}-\d{2})')
    grid_shape = next(iter(district_masks.values())).shape

    files = sorted(f for f in os.listdir(npy_dir)
                   if f.startswith('{}_'.format(source)) and f.endswith('.npy'))
    for f in files:
        m = date_re.search(f)
        if not m:
            continue
        date_str = m.group(1)
        precip_arr = np.load('{}/{}'.format(npy_dir, f))
        if precip_arr.shape != grid_shape:
            print('  Grid mismatch on {} ({}), skipping'.format(f, precip_arr.shape))
            continue
        for key, mask in district_masks.items():
            avg = float(np.nanmean(precip_arr[mask])) if mask.any() else np.nan
            series[key].append((date_str, avg))
    print('  Read {} npy files'.format(len(files)))
    return series


def write_district_csv(save_dir, district, source, rows):
    """Write one district's rebuilt hindcast series. Columns: doy, avg_precip, District."""
    df = pd.DataFrame(rows, columns=['doy', 'avg_precip'])
    df['District'] = district
    df['doy'] = df['doy'].astype(str)
    df = df.drop_duplicates(subset=['doy'], keep='last')
    df = df.sort_values(by='doy').reset_index(drop=True)
    df.to_csv('{}/{}_{}.csv'.format(save_dir, district, source), index=False)


if __name__ == "__main__":
    nepal_arr = np.array(Image.open('{}/District_Labels.tif'.format(root_dir)))
    district_masks = build_district_masks(nepal_arr)

    for source in SOURCES:
        npy_dir = '{}/{}_Mean_Pixelwise'.format(root_dir, source)
        save_dir = '{}/Subseasonal/DistrictLevel'.format(npy_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print('\n=== {} -> {} ==='.format(source, save_dir))
        if not os.path.isdir(npy_dir):
            print('  npy dir not found, skipping: {}'.format(npy_dir))
            continue

        series = read_npy_series(npy_dir, source, district_masks)
        for district in district_dict:
            write_district_csv(save_dir, district, source, series[district])
        example = series[next(iter(district_dict))]
        print('  Wrote {} districts, {} rows each (pre-dedupe).'.format(len(district_dict), len(example)))

    print('\nDone.')
