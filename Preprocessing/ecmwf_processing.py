"""
Processing script for ecmwf forecast data
"""

import xarray as xr
import datetime
from datetime import datetime, timedelta
import os
import numpy as np

# Specify directory of nc file to process
root_dir = '/Volumes/PRO-G40/landslides/Nepal_Landslides_Forecasting_Project/Monsoon2024_Prep/2024_Season_Retro'

def daterange(date1, date2):
  date_list = []
  for n in range(int((date2 - date1).days)+1):
    dt = date1 + timedelta(n)
    date_list.append(dt)
  return date_list


def ecmwf_to_daily(precip_data):
    """
    Processes the pre-2024 ecmwf forecast data to a daily product
    taking the nearest neighbour as day of year
    """
    # Give a range to query, will give +5 days on either side in case of missing data
    start = datetime.strptime('2024-10-01', '%Y-%m-%d')
    end = datetime.strptime('2024-10-31', '%Y-%m-%d')

    dates = daterange(start, end)
    for doy in dates:
        print('Processing for DOY: {}'.format(doy))
        precip_val = None
        n = 0
        while precip_val is None:
            try:
                date_queried = doy
                date_queried = date_queried + timedelta(days=n)
                precip_val = precip_data.sel(time='{}'.format(date_queried)).tp.values
            except:
                n +=1
        # Specify save dir
        directory = '{}/PrecipitationModel_Forecast_Data/Subseasonal/ecmwf'.format(root_dir)
        if not os.path.exists(directory):
            os.makedirs(directory)

        np.save('{}/precipitation_forecast_ecmwf_doy_{}.npy'.format(directory, doy), precip_val)

# Specify precip data to process
precip = xr.open_dataset('/Volumes/PRO-G40/landslides/Nepal_Landslides_Forecasting_Project/Monsoon2024_Prep/'
                         '2024_Season_Retro/Nepal_ECMWF_2024_Oct_GPMfinal_2.nc')
ecmwf_to_daily(precip)