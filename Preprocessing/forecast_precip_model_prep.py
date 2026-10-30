"""
Script to process the precipitation forecast data from the ensemble models
into the per-district daily average precipitation CSVs consumed by the
classifier's sample generator.

Each district's subseasonal CSV is rebuilt end-to-end from its source data:
  - historical dates come from the per-day .npy files in ensemble_member_0
  - new dates come from the daily forecast netCDF products (tp[time, lat, lon])

Both sources are averaged with a true 0/1 district mask, i.e.
  avg_precip = mean(precip over the district's pixels).

NOTE: this corrects the previous behaviour, which multiplied precip by the
district's *label ID* (District_Labels.tif stores district IDs 1-78, not a
0/1 mask), leaving avg_precip scaled by the district ID. Because the whole
series is regenerated here, history and new data stay on the same (correct)
scale. Only ensemble_member_0 is regenerated; UKMO ensemble_member_1-3 remain
on the old scale.

LATITUDE FIX (2026-08-28)
-------------------------
Every forecast netCDF in this project has an ASCENDING latitude axis (row 0 =
25.05N, row 59 = 30.95N), while District_Labels.tif is NORTH-UP (row 0 = 30.95N).
This script used to index the netCDF by array position against the mask, so each
district's forecast rainfall was taken from its mirror image across 28.00N -
monsoon rain landed on the Himalayan rain shadow and was withheld from the wet
southeast. The .npy history has the same orientation, because nc_to_npy.py saves
`ds.sel(time=...).tp.values[i]` straight to disk with no flip.

`--flip_lat` (default True) applies np.flipud so the northernmost netCDF row
lines up with the northernmost tiff row. `--flip_lat False` reproduces the old,
mirrored behaviour byte for byte, which is what the models in
2026_Monsoon_Testing/deploy_model* were trained on - keep it available so those
remain reproducible.

Longitude was never affected: both axes run west to east. Files narrower than the
tiff (the 2026 ECMWF pull is 99 columns, not 100) are padded with NaN on the east
edge; that column contains no district pixels, and place_on_grid asserts as much.
"""

import argparse
import numpy as np
import os
import re
import pandas as pd
import xarray as xr
from PIL import Image

PROJECT_DIR = os.environ.get('LANDSLIDE_PROJECT_DIR', 'data')  # see README > Data


GRID_SHAPE = (60, 100)                                   # District_Labels.tif
GRID_LON0, GRID_LAT_TOP, GRID_RES = 79.05, 30.95, 0.1    # tiff is north-up


def get_args():
    p = argparse.ArgumentParser(description='Rebuild per-district forecast precipitation CSVs')
    p.add_argument('--flip_lat', default='True',
                   help='True (default) aligns the ascending netCDF latitude axis with the '
                        'north-up district tiff. False reproduces the old mirrored behaviour.')
    p.add_argument('--out_suffix', default='',
                   help="Appended to the DistrictLevel output dir, e.g. '_latfix'. Empty "
                        'overwrites the existing CSVs.')
    return p.parse_args()


root_dir = PROJECT_DIR + '/Monsoon2024_Prep'
forecast_data_dir = '{}/PrecipitationModel_Forecast_Data'.format(root_dir)

# Per model (folder name, ensemble member): historical npy dir + new netCDF products.
# ecmwf -> Subseasonal/ecmwf/... ; ukmo -> Subseasonal/UKMO/... (single member -> ensemble_member_0)
JOBS = [
    {
        'model': 'ecmwf',
        'ens': 0,
        'nc_files': [
            'Nepal_ecmwf_tp_Apr_Oct_2025_GPMfinal.nc',
            # Apr-June supersedes the earlier Apr-May pull; listed last so its
            # values win the doy de-dupe in write_district_csv.
            'Nepal_ecmwf_tp_Apr_May_2026_GPMfinal.nc',
            'Nepal_ecmwf_tp_Apr_June_2026_GPMfinal.nc',
            # Season-long pull, 2026-05-15 to 2026-08-24. Listed last so it wins
            # the doy de-dupe where it overlaps Apr-June.
            'Nepal_ecmwf_tp_May_Aug_2026_GPMfinal.nc',
            # Late-August extension, 2026-08-15 to 2026-09-12, downloaded 2026-09-03.
            # Lives beside the operational model rather than in the forecast data
            # folder, so it is given as a path relative to root_dir.
            '2026_Monsoon_Testing_Operations/Nepal_ecmwf_tp_Aug_2026_updated_GPMfinal.nc',
        ],
    },
    {
        'model': 'UKMO',
        'ens': 0,
        'nc_files': [
            'Nepal_ukmo_tp_Apr_Oct_2025_GPMfinal.nc',
            'Nepal_ukmo_tp_Apr_May_2026_GPMfinal.nc',
        ],
    },
]

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


def district_avg(precip_arr, district_masks):
    """Return {district: mean(precip over that district's pixels)} for one 2D precip field."""
    return {key: float(np.nanmean(precip_arr[mask])) if mask.any() else np.nan
            for key, mask in district_masks.items()}


def place_on_grid(field, lons, lats, flip_lat, any_district):
    """
    Put one 2D forecast field onto the District_Labels.tif grid, padding with NaN.

    flip_lat=True flips the ascending netCDF latitude axis so its northernmost row
    meets the tiff's northernmost row. flip_lat=False keeps the old positional
    behaviour, which mirrors the field across 28.00N.

    lons/lats may be None for .npy history, which carries no coordinates; those
    arrays are raw netCDF slices (see nc_to_npy.py) and are assumed to start at
    the grid origin.
    """
    g = np.full(GRID_SHAPE, np.nan, dtype=float)
    f = np.asarray(field, dtype=float)
    c0 = 0 if lons is None else int(round((float(lons[0]) - GRID_LON0) / GRID_RES))
    if flip_lat:
        f = np.flipud(f)
        r0 = 0 if lats is None else int(round((GRID_LAT_TOP - float(max(lats))) / GRID_RES))
    else:
        r0 = 0
    g[r0:r0 + f.shape[0], c0:c0 + f.shape[1]] = f
    missing = any_district & np.isnan(g)
    if missing.any():
        raise ValueError('{} district pixels fall outside the forecast grid - the file '
                         'does not cover all of Nepal.'.format(int(missing.sum())))
    return g


def read_npy_history(npy_dir, district_masks, flip_lat, any_district):
    """
    Read the per-day historical .npy files and return a dict:
        district -> list of (date_str, avg_precip).
    Files are named like 'precipitation_forecast_id0_doy_YYYY-MM-DD.npy'.
    """
    series = {key: [] for key in district_dict}
    if not os.path.isdir(npy_dir):
        print('  No npy history dir, skipping: {}'.format(npy_dir))
        return series

    date_re = re.compile(r'doy_(\d{4}-\d{2}-\d{2})')
    files = sorted(f for f in os.listdir(npy_dir) if f.endswith('.npy'))
    for f in files:
        m = date_re.search(f)
        if not m:
            continue
        date_str = m.group(1)
        precip_arr = np.squeeze(np.load('{}/{}'.format(npy_dir, f)))
        if precip_arr.shape != next(iter(district_masks.values())).shape:
            print('  Grid mismatch on {} ({}), skipping'.format(f, precip_arr.shape))
            continue
        precip_arr = place_on_grid(precip_arr, None, None, flip_lat, any_district)
        for key, avg in district_avg(precip_arr, district_masks).items():
            series[key].append((date_str, avg))
    print('  Read {} historical npy files'.format(len(files)))
    return series


def read_nc_forecast(nc_path, district_masks, flip_lat, any_district):
    """
    Read a forecast netCDF (tp[time, lat, lon]) and return a dict:
        district -> list of (date_str, avg_precip).
    """
    series = {key: [] for key in district_dict}
    ds = xr.open_dataset(nc_path)
    tp_vals = ds['tp'].values
    lats, lons = ds.lat.values, ds.lon.values
    date_strs = [str(np.datetime64(t, 'D')) for t in ds.time.values]
    ds.close()

    for ti, date_str in enumerate(date_strs):
        grid = place_on_grid(np.squeeze(tp_vals[ti]), lons, lats, flip_lat, any_district)
        for key, avg in district_avg(grid, district_masks).items():
            series[key].append((date_str, avg))
    print('  Read {} dates from {}'.format(len(date_strs), os.path.basename(nc_path)))
    return series


def write_district_csv(save_dir, district, rows):
    """Write one district's rebuilt series. Columns: doy, avg_precip, District."""
    df = pd.DataFrame(rows, columns=['doy', 'avg_precip'])
    df['District'] = district
    df['doy'] = df['doy'].astype(str)
    df = df.drop_duplicates(subset=['doy'], keep='last')
    df = df.sort_values(by='doy').reset_index(drop=True)
    df.to_csv('{}/{}_subseasonal_precipitation.csv'.format(save_dir, district), index=False)


if __name__ == "__main__":
    args = get_args()
    flip_lat = str(args.flip_lat).lower() not in ('false', '0', 'no')
    print('Latitude flip: {}  (False reproduces the pre-2026-08-28 mirrored data)'.format(flip_lat))

    nepal_arr = np.array(Image.open('{}/District_Labels.tif'.format(root_dir)))
    district_masks = build_district_masks(nepal_arr)
    any_district = np.zeros(GRID_SHAPE, dtype=bool)
    for _m in district_masks.values():
        any_district |= _m

    for job in JOBS:
        model, ens = job['model'], job['ens']
        npy_dir = '{}/Subseasonal/{}/ensemble_member_{}'.format(forecast_data_dir, model, ens)
        save_dir = '{}/Subseasonal/{}/DistrictLevel{}/ensemble_member_{}'.format(
            forecast_data_dir, model, args.out_suffix, ens)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print('\n=== {} ensemble_member_{} -> {} ==='.format(model, ens, save_dir))

        # Historical .npy dates
        combined = read_npy_history(npy_dir, district_masks, flip_lat, any_district)

        # New forecast .nc dates
        for nc_name in job['nc_files']:
            # A bare filename lives in the forecast data folder; anything with a
            # separator is taken relative to root_dir (or used as-is if absolute).
            if os.path.isabs(nc_name):
                nc_path = nc_name
            elif '/' in nc_name:
                nc_path = '{}/{}'.format(root_dir, nc_name)
            else:
                nc_path = '{}/{}'.format(forecast_data_dir, nc_name)
            if not os.path.exists(nc_path):
                print('  netCDF not found, skipping: {}'.format(nc_path))
                continue
            nc_series = read_nc_forecast(nc_path, district_masks, flip_lat, any_district)
            for key in district_dict:
                combined[key].extend(nc_series[key])

        # Write one rebuilt CSV per district
        for district in district_dict:
            write_district_csv(save_dir, district, combined[district])
        example = combined[next(iter(district_dict))]
        print('  Wrote {} districts, {} rows each (pre-dedupe).'.format(len(district_dict), len(example)))

    print('\nDone.')
