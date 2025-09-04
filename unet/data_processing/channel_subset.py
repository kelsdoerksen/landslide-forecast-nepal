"""
Subsetting the channels into a 10-channel sample
Channels to retain:
dem, aspect, slope, modis, lookback avg, lookback min, lookback max,
lookahead avg, lookahead min, lookahead max
Order of channels is:
np.array([dem, aspect, slope, modis, lookback[0,:,:], lookback[1,:,:], lookback[2,:,:],
                                 lookback[3,:,:], lookback[4,:,:], lookback[5,:,:], lookback[6,:,:], lookback[7,:,:],
                                 lookback[8,:,:], lookback[9,:,:], lookback[10,:,:], lookback[11,:,:], lookback[12,:,:],
                                 lookback[13,:,:], lookahead[0,:,:], lookahead[1,:,:], lookahead[2,:,:], lookahead[3,:,:],
                                 lookahead[4,:,:], lookahead[5,:,:], lookahead[6,:,:], lookahead[7,:,:], lookahead[8,:,:],
                                 lookahead[9,:,:], lookahead[10,:,:], lookahead[11,:,:], lookahead[12,:,:],
                                 lookahead[13,:,:]])
"""
from pickle import PicklingError, UnpicklingError

import numpy as np
import os


def aggregate_features(arr):
    """
    Aggregate array to get final shape
    """
    # static channels
    dem = arr[0]
    aspect = arr[1]
    slope = arr[2]
    modis = arr[3]

    # lookback and lookahead slices
    lookback = arr[4:17]
    lookahead = arr[17:31]

    # aggregate features
    max_lookback = np.max(lookback, axis=0)
    min_lookback = np.min(lookback, axis=0)
    avg_lookback = np.mean(lookback, axis=0)

    max_lookahead = np.max(lookahead, axis=0)
    min_lookahead = np.min(lookahead, axis=0)
    avg_lookahead = np.mean(lookahead, axis=0)

    # stack into new array
    agg_array = np.stack([
        dem, aspect, slope, modis,
        max_lookback, min_lookback, avg_lookback,
        max_lookahead, min_lookahead, avg_lookahead
    ], axis=0)

    return agg_array


if __name__ == "__main__":
    # Hardcoding sample dir for now
    sample_dir = ('/Volumes/PRO-G40/landslides/Nepal_Landslides_Forecasting_Project/Monsoon2024_Prep/'
                  'UNet_Samples_14Day_GPMv07/UKMO/ensemble_0')
    save_dir = ('/Volumes/PRO-G40/landslides/Nepal_Landslides_Forecasting_Project/Monsoon2024_Prep/'
                'UNet_Samples_14Day_GPMv07/UKMO/ensemble_0_agg')

    for f in os.listdir(sample_dir):
        try:
            arr = np.load(os.path.join(sample_dir, f), allow_pickle=True)
        except UnpicklingError:
            print('error with loading {}'.format(f))
            continue
        print('aggregating for {}'.format(f))
        agg_array = aggregate_features(arr)
        np.save(os.path.join(save_dir, f), agg_array)

