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


# Initialize ee
ee.Initialize(opt_url='https://earthengine-highvolume.googleapis.com')


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

    # Define polygon over Nepal based on extent provided by Geo
    aoi = ee.Geometry.Polygon([[
        [79.05, 25.05],
        [88.95, 25.05],
        [88.95, 30.95],
        [79.05, 30.95]]], None, False)

    print('Processing for {}'.format(point['start_date']))

    # --- Query GPM collection over year of interest ---
    collection = ee.ImageCollection('NASA/GPM_L3/IMERG_V06')\
        .filterDate('{}'.format(point['start_date']), '{}'.format(point['end_date']))

    img_list = collection.toList(48)
    arr_imgs_list = []
    for i in range(48):
        print('Running for image: {}'.format(i))
        img = ee.Image(img_list.get(i)).select(['precipitationCal'])
        arr_img = geemap.ee_to_numpy(img, region=aoi)[:,:,0]
        arr_imgs_list.append(arr_img)

    daily_mean = np.mean(arr_imgs_list, axis=0)
    np.save('/Users/kelseydoerksen/Desktop/GPM_Mean_Pixelwise/2022/{}.npy'.format(point['start_date']), daily_mean)


def getRequests(query_year):
    """
    Generate list of GPM data to extract over Nepal
    Note coordinates are of form lon, lat
    """
    # --- Calculating  daily mean ---
    start_date = date(int(query_year), 10, 25)
    end_date = date(int(query_year), 10, 26)
    #end_date = date(int(query_year) + 1, 1, 1)

    # Get list of days over the query year
    dates_list = daterange(start_date, end_date)

    date_ranges = []
    for i in range(len(dates_list) - 1):
        date_ranges.append({'start_date': dates_list[i],
                            'end_date':dates_list[i+1]})

    return date_ranges


if __name__ == '__main__':
    st = time.time()
    print('Start time: {}'.format(datetime.fromtimestamp(st).strftime('%Y-%m-%d %H:%M:%S')))

    items = getRequests(2019)

    pool = multiprocessing.Pool(15)
    pool.starmap(getResult, enumerate(items))
    pool.close()
    pool.join()

    et = time.time()
    # get the execution time
    elapsed_time = et - st
    print('Execution time:', elapsed_time, 'seconds')