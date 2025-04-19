"""
Preprocess GEE GPM script, faster than
using the Javascript API
"""

import ee
import multiprocessing
from datetime import datetime, timedelta, date
import numpy as np
import geemap
import time
import argparse


# argparse arguments
# Argparse arguments for CLI
parser = argparse.ArgumentParser(description='Google Earth Engine Precipitation Processing',
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--gee_data",
                    help='GEE data to process')
parser.add_argument("--query_year",
                    help='Yeah to query data for')

root_dir = '/Volumes/PRO-G40/landslides/Nepal_Landslides_Forecasting_Project/Monsoon2024_Prep'

# Initialize ee
ee.Initialize(project='sudsaq-kelsey')
#ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')


def daterange(date1, date2):
    date_list = []
    for n in range(int((date2 - date1).days) + 1):
        dt = date1 + timedelta(n)
        date_list.append(dt.strftime("%Y-%m-%d"))
    return date_list

def getResult(index, point):
    """
    Handle HTTP requests to download
    GEE image
    :param: point: config information to extract data from
    """

    if point['gee_collection'] == 'JAXA/GPM_L3/GSMaP/v8/operational':
        band = 'hourlyPrecipRate'
        num_imgs = 24
        save_folder = 'GSMaP'

    if point['gee_collection'] == 'NASA/GPM_L3/IMERG_V06':
        band = 'precipitationCal'
        num_imgs = 48
        save_folder = 'GPM'

    if point['gee_collection'] == 'NASA/GPM_L3/IMERG_V07':
        band = 'precipitation'
        num_imgs = 48
        save_folder = 'GPMv07'

    # Define polygon over Nepal based on extent provided by Geo
    aoi = ee.Geometry.Polygon([[
        [79.05, 25.05],
        [88.95, 25.05],
        [88.95, 30.95],
        [79.05, 30.95]]], None, False)

    print('Processing for {}'.format(point['start_date']))

    # --- Query collection over year of interest ---
    collection = ee.ImageCollection(point['gee_collection'])\
        .filterDate('{}'.format(point['start_date']), '{}'.format(point['end_date']))

    img_list = collection.toList(num_imgs)
    arr_imgs_list = []
    for i in range(num_imgs):
        print('Running for image: {}'.format(i))
        try:
            img = ee.Image(img_list.get(i)).select([band])
            arr_img = geemap.ee_to_numpy(img, region=aoi)[:,:,0]
            arr_imgs_list.append(arr_img)
        except TypeError as e:
            continue

    daily_mean = np.mean(arr_imgs_list, axis=0)
    if point['gee_collection'] == 'JAXA/GPM_L3/GSMaP/v8/operational':
        save_folder = 'GSMaP'
    if point['gee_collection'] == 'NASA/GPM_L3/IMERG_V06':
        save_folder = 'GPMv06'
    if point['gee_collection'] == 'NASA/GPM_L3/IMERG_V07':
        save_folder = 'GPMv07'
    np.save('{}/{}_Mean_Pixelwise/{}_{}.npy'.format(root_dir, save_folder, save_folder, point['start_date']), daily_mean)


def getRequests(query_year, gee_data):
    """
    Generate list of GEE data to extract over Nepal
    Note coordinates are of form lon, lat
    :param: query_year: year to query data
    :param: gee_data: full name of GEE data collection
    """
    # --- Calculating  daily mean ---
    start_date = date(int(query_year), 4, 1)
    end_date = date(int(query_year) , 10, 30)

    # Get list of days over the query year
    dates_list = daterange(start_date, end_date)

    date_ranges = []
    for i in range(len(dates_list) - 1):
        date_ranges.append({'start_date': dates_list[i],
                            'end_date': dates_list[i+1],
                            'gee_collection': gee_data})

    return date_ranges


if __name__ == '__main__':
    args = parser.parse_args()
    year = int(args.query_year)
    collection_name = args.gee_data

    st = time.time()
    print('Start time: {}'.format(datetime.fromtimestamp(st).strftime('%Y-%m-%d %H:%M:%S')))

    items = getRequests(year, collection_name)

    pool = multiprocessing.Pool(15)
    pool.starmap(getResult, enumerate(items))
    pool.close()
    pool.join()

    et = time.time()
    # get the execution time
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')