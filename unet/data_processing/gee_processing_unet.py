"""
Preprocess GEE MODIS data
"""

import ee
from datetime import datetime, timedelta, date
import numpy as np
import geemap
import time
import argparse


# argparse arguments
# Argparse arguments for CLI
parser = argparse.ArgumentParser(description='Google Earth Engine MODIS Processing',
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--year",
                    help='Year to query data for')

root_dir = '/Volumes/PRO-G40/landslides/Nepal_Landslides_Forecasting_Project/Monsoon2024_Prep/MODIS_Pixelwise'

# Initialize ee
ee.Initialize(project='sudsaq-kelsey',opt_url='https://earthengine-highvolume.googleapis.com')


def generate_data(year):
    # Define polygon over Nepal based on extent provided by Geo
    aoi = ee.Geometry.Polygon([[
        [79.05, 25.05],
        [88.95, 25.05],
        [88.95, 30.95],
        [79.05, 30.95]]], None, False)

    start = '{}-01-01'.format(year)
    end = '{}-01-01'.format(int(year)+1)

    collection = ee.ImageCollection('MODIS/061/MCD12Q1').filterDate(start,end)
    img_list = collection.toList(1)
    img = ee.Image(img_list.get(0)).select(['LC_Type1'])
    img = img.reproject(crs='EPSG:4326', scale=11132)
    arr_img = geemap.ee_to_numpy(img, region=aoi)[:, :, 0]

    np.save('{}/MODIS_{}.npy'.format(root_dir, year), arr_img)

args = parser.parse_args()
generate_data(args.year)