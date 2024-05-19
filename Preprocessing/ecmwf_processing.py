"""
Processing script for ecmwf forecast data
"""

import xarray as xr
import datetime
from datetime import datetime, timedelta
import os
import numpy as np

root_dir = '/Users/kelseydoerksen/Desktop/Nepal_Landslides_Forecasting_Project/Monsoon2024_Prep/' \
           'PrecipitationModel_Forecast_Data'

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
    start = datetime.strptime('2023-01-01', '%Y-%m-%d')
    end = datetime.strptime('2024-01-04', '%Y-%m-%d')

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
        directory = '{}/Subseasonal/ecmwf'.format(root_dir)
        if not os.path.exists(directory):
            os.makedirs(directory)

        np.save('{}/precipitation_forecast_ecmwf_doy_{}.npy'.format(directory, doy), precip_val)

precip = xr.open_dataset('/Users/kelseydoerksen/Desktop/Nepal_Landslides_Forecasting_Project/'
                         'Monsoon2024_Prep/2024_ecmwf_ukmo/Nepal_ECMWF_Jan2015-May2024_GPMfinal.nc')
ecmwf_to_daily(precip)