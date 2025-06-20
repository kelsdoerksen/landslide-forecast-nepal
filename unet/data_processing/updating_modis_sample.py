"""
Script to update the samples with the correct modis, need to run for every sample folder
"""


import numpy as np
import os
import re
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Fixing samples with corrected MODIS',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--forecast_model",
                        help='Forecast model')
    parser.add_argument("--ensemble",
                        help='Ensemble number')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    forecast = args.forecast_model
    num = args.ensemble


    # Paths (update these as needed)
    modis_dir = '/Volumes/PRO-G40/landslides/Nepal_Landslides_Forecasting_Project/Monsoon2024_Prep/MODIS_Pixelwise'
    samples_dir = ('/Volumes/PRO-G40/landslides/Nepal_Landslides_Forecasting_Project/Monsoon2024_Prep/'
                   'UNet_Samples_14Day_GPMv07/{}/ensemble_{}'.format(forecast, num))
    output_dir = samples_dir           # Can be the same or a different folder

    # Compile regex to extract date
    date_pattern = re.compile(r'sample_(\d{4})-(\d{2})-(\d{2})\.npy')

    # List all sample files
    sample_files = [f for f in os.listdir(samples_dir) if f.startswith('sample_') and f.endswith('.npy')]

    for file_name in sample_files:
        match = date_pattern.match(file_name)
        if not match:
            print(f"Skipping file with unmatched format: {file_name}")
            continue

        year, month, day = match.groups()
        if year == '2024':
            modis_filename = 'MODIS_2023.npy'
        else:
            modis_filename = f"MODIS_{year}.npy"
        modis_path = os.path.join(modis_dir, modis_filename)

        if not os.path.exists(modis_path):
            print(f"MODIS file for year {year} not found: {modis_path}")
            continue

        # Load MODIS and sample arrays
        modis_array = np.load(modis_path)  # shape (60, 100)
        sample_path = os.path.join(samples_dir, file_name)
        sample_array = np.load(sample_path)  # shape (32, 60, 100)

        if sample_array.shape != (32, 60, 100):
            print(f"Unexpected sample shape in {file_name}: {sample_array.shape}")
            continue
        if modis_array.shape != (60, 100):
            print(f"Unexpected MODIS shape in {modis_filename}: {modis_array.shape}")
            continue

        # Replace index 3
        sample_array[3, :, :] = modis_array

        # Save updated sample
        output_path = os.path.join(output_dir, file_name)
        np.save(output_path, sample_array)
        print(f"Updated {file_name} with MODIS {year}")

    print("✅ Done updating all samples.")