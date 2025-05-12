"""
Preprocess the precipitation forecast data from ensemble model
"""

import numpy as np
import xarray as xr
import datetime
from datetime import datetime, date, timedelta
import os

# --- Specify dir of data
# root dir
root_dir = '/Volumes/PRO-G40/landslides/Nepal_Landslides_Forecasting_Project/Monsoon2024_Prep/PrecipitationModel_Forecast_Data'

# Dictionary of forecast precipitation models
models = {
    'KMA': {'filename': 'Nepal_KMA_4ens_RT_Nov2016-Oct2023_GPM_final.nc',
            'ensemble_count': 4,
            'date_start': '2023-01-01',
            'date_end': '2023-11-14'},
    'NCEP': {'filename': 'Nepal_NCEP_16ens_RT_Jan2015-Oct2023_GMPfinal.nc',
             'ensemble_count': 16,
             'date_start': '2015-01-01',
             'date_end': '2023-11-14'},
    'UKMO': {'filename': "Nepal_UKMO_4ens_RT_Dec2015-Oct2023_GPMfinal.nc",
             'ensemble_count': 4,
             'date_start': '2015-12-01',
             'date_end': '2024-01-01'}
}

def daterange(date1, date2):
  date_list = []
  for n in range(int((date2 - date1).days)+1):
    dt = date1 + timedelta(n)
    date_list.append(dt)
  return date_list


def process_subseasonal(model_type):
    """
    Process subseasonal forecast data
    Save individual ensemble
    """
    precip_model = models[model_type]
    print('Running for forecast model: {}'.format(model_type))

    # Load precip model based on specified model type
    ds = xr.open_dataset('{}/{}'.format(root_dir, precip_model['filename']))
    start_date = datetime.strptime(precip_model['date_start'], '%Y-%m-%d')
    end_date = datetime.strptime(precip_model['date_end'], '%Y-%m-%d')
    date_list = daterange(start_date, end_date)

    missing_dates = []

    for doy in date_list:
        print('Processing for DOY: {}'.format(doy))
        for i in range(precip_model['ensemble_count']):
            start_date_for_lookahead = doy + timedelta(days=1)
            lookahead_date = start_date_for_lookahead+timedelta(days=14)
            try:
                precip_arr = ds.sel(time='{}'.format(lookahead_date)).tp.values[i]
            except (KeyError, IndexError):
                print('DOY {} is not found, skipping'.format(lookahead_date))
                missing_dates.append(lookahead_date)
                continue
            # save array
            directory = '{}/Subseasonal/{}/ensemble_member_{}'.format(root_dir, model_type, i)
            if not os.path.exists(directory):
                os.makedirs(directory)
            np.save('{}/precipitation_forecast_id{}_doy_{}.npy'.format(directory, i,
                                                                       str(start_date_for_lookahead)[0:10]), precip_arr)

    with open('{}/missing_dates_{}.txt'.format(root_dir, model_type), 'w') as fp:
        for item in missing_dates:
            # write each item on a new line
            fp.write("%s\n" % item)
        print('Done')

forecast_models = ['NCEP']
for fm in forecast_models:
    process_subseasonal(fm)

