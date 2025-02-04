"""
Script to generate samples for image segmentation
"""

import numpy as np
from PIL import Image
from datetime import datetime, timedelta, date
import argparse
import os


def get_args():
    parser = argparse.ArgumentParser(description='Running ML Pipeline for Landslide Prediction')
    parser.add_argument('--root_dir', help='Root directory of data')
    parser.add_argument('--model', help='Ensemble Model to query precipitation data from. Currently supports KMA, NCEP,'
                                        'UKMO')
    parser.add_argument('--ens_member', help='Ensemble member to use precipitation data from.')
    parser.add_argument('--hindcast_source', help='Precipitation Hindcast data source. Must be one of GPM or GSMaP')
    parser.add_argument('--start_year', help='Year to start data processing')
    parser.add_argument('--end_year', help='Year to end data processing')

    return parser.parse_args()


def load_dem_arrays():
    """
    Load dem-based tifs, return np arrays
    """
    dem_im = Image.open('{}/Reprojected_MatchingExtent_DEM.tif'.format(dem_dir))
    aspect_im = Image.open('{}/Reprojected_MatchingExtent_Aspect.tif'.format(dem_dir))
    slope_im = Image.open('{}/Reprojected_MatchingExtent_Slope.tif'.format(dem_dir))

    return np.array(dem_im), np.array(aspect_im), np.array(slope_im)


def load_modis(doy):
    """
    Load modis array based on year
    """
    year = doy[0:4]
    if int(year) > 2023:
        query_year = '2023'
    else:
        query_year = year

    modis_arr = np.load('{}/MODIS_{}.npy'.format(MODIS_dir, query_year))
    return modis_arr


def load_precip_lookback(day, hindcast_data):
    '''
    Load 14-day precipitation lookback and generate
    mean, cumulative precipitation arrays for sample
    :param: doy: day of year
    :param: hindcast_data: precipitation data source
    '''
    format = '%Y-%m-%d'
    hindcast_dir = '{}/{}_Mean_Pixelwise'.format(root_dir, hindcast_data)
    day = str(day)
    date_dateformat = datetime.strptime(day, format)

    precip_list = []
    cumulative_precip = []
    for i in range(14):
        delta = date_dateformat.date() - timedelta(days=i)
        lookback = delta.strftime(format)
        print('running for lookback {}'.format(lookback))
        try:
            lookback_precip = np.load('{}/{}_{}.npy'.format(hindcast_dir, hindcast_data, lookback))
            if lookback_precip.shape != np.zeros((60,100)).shape:
                print('Missing precipitation for date {}, returning None'.format(lookback))
                return None
            else:
                precip_list.append(lookback_precip)
                cumulative_precip.append(lookback_precip * 24)
        except ValueError and FileNotFoundError:
            print('No {} available for lookback, skipping sample'.format(hindcast_data))
            return None

    precip_mean = np.array(precip_list).mean(axis=0)
    precip_list.append(precip_mean)
    precip_list.append(sum(cumulative_precip))
    return np.array(precip_list)


def load_precip_lookahead(day, ensemble_model, ensemble_num):
    '''
    Load precipitation forecast model data and generate
    mean, cumulative precipitation arrays for sample
    '''
    format = '%Y-%m-%d'
    day = str(day)
    date_dateformat = datetime.strptime(day, format)
    precip_list = []
    cumulative_precip = []
    for i in range(14):
        increment = i+1
        delta = date_dateformat.date() + timedelta(days=increment)
        lookahead = delta.strftime(format)
        print('Running for lookahead: {}'.format(lookahead))
        try:
            lookahead_precip = np.load('{}/Subseasonal/{}/ensemble_member_{}/precipitation_forecast_id{}_doy_{}.npy'.
                                       format(precip_dir, ensemble_model, ensemble_num, ensemble_num, lookahead))
            if lookahead_precip.shape != np.zeros((60,100)).shape:
                print('Missing precipitation for date {}, returning None'.format(lookahead))
                return None
            else:
                precip_list.append(lookahead_precip)
                cumulative_precip.append(lookahead_precip * 24)
        except ValueError and FileNotFoundError:
            print('Missing precipitation for date {}, returning None')
            return None

    precip_mean = np.array(precip_list).mean(axis=0)
    precip_list.append(precip_mean)
    precip_list.append(sum(cumulative_precip))
    return np.array(precip_list)


def generate_sample(doy, ens_model, ens_num, hindcast):
    """
    Generate sample for doy specified
    :param: doy: day of year to generate sample
    :param: ens_model: precipitation forecast model
    :param: ens_num: precipitation forecast model ensemble member
    :param: hindcast: precipitation hindcast data source
    """
    sample_list = []
    # Add dem, aspect, slope
    dem, aspect, slope = load_dem_arrays()

    # Get lookback precip
    lookback = load_precip_lookback(doy, hindcast)
    if lookback is None:
        print('Missing lookback data, cannot generate sample, skipping for doy: {}'.format(doy))
        return None

    # Get lookahead precip
    lookahead = load_precip_lookahead(doy, ens_model, ens_num)
    if lookahead is None:
        print('Missing lookahead data, cannot generate sample, skipping for doy: {}'.format(doy))
        return None

    # Get modis
    modis = load_modis(doy)

    try:
        # Append to sample list and return as array
        sample_array = np.array([dem, aspect, slope, modis, lookback[0,:,:], lookback[1,:,:], lookback[2,:,:],
                                 lookback[3,:,:], lookback[4,:,:], lookback[5,:,:], lookback[6,:,:], lookback[7,:,:],
                                 lookback[8,:,:], lookback[9,:,:], lookback[10,:,:], lookback[11,:,:], lookback[12,:,:],
                                 lookback[13,:,:], lookahead[0,:,:], lookahead[1,:,:], lookahead[2,:,:], lookahead[3,:,:],
                                 lookahead[4,:,:], lookahead[5,:,:], lookahead[6,:,:], lookahead[7,:,:], lookahead[8,:,:],
                                 lookahead[9,:,:], lookahead[10,:,:], lookahead[11,:,:], lookahead[12,:,:],
                                 lookahead[13,:,:]])
    except IndexError:
        return None
    # Save array
    save_dir = '{}/UNet_Samples_14Day_{}/{}/ensemble_{}'.format(root_dir, hindcast, ens_model, ens_num)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    np.save('{}/sample_{}.npy'.format(save_dir, doy),sample_array)


def daterange(date1, date2):
    dates = []
    for n in range(int((date2-date1).days)+1):
        dt = date1+timedelta(n)
        dates.append(dt.strftime('%Y-%m-%d'))
    return dates


if __name__ == "__main__":
    args = get_args()
    model = args.model
    ens_num = args.ens_member
    hindcast_source = args.hindcast_source
    start = args.start_year
    end = args.end_year
    root_dir = args.root_dir

    # Setting data directories to query from
    MODIS_dir = '{}/MODIS_Pixelwise'.format(root_dir)
    precip_dir = '{}/PrecipitationModel_Forecast_Data'.format(root_dir)
    dem_dir = '{}/Topography'.format(root_dir)

    print('Generating samples using forecast data from model: {}, member: {}'.format(model, ens_num))

    # Get list of dates
    sdate = date(int(start), 1, 1)
    edate = date(int(end), 12, 31)
    date_list = daterange(sdate, edate)

    # Generate samples
    for date in date_list:
        print('Generating sample for DOY: {}'.format(date))
        generate_sample(date, model, ens_num, hindcast_source)


