"""
Scrap processing scripts to do some extra stuff I didn't do in original setup
Will likely just incorporate all these functions into the generate samples
"""

import pandas as pd

def add_location_information(feature_df, centroids_df):
    '''
    Add location of centroid of District to feature
    '''

    # Get the count of unique districts in df
    unique_num = feature_df['district'].nunique()

    # Get the number of entries per district by dividing by length
    num_entries_each = len(feature_df)/unique_num
    centroids_repeated = centroids_df.loc[centroids_df.index.repeat(num_entries_each)]

    centroids_repeated = centroids_repeated[['DISTRICT', 'xcoord', 'ycoord']]
    centroids_repeated = centroids_repeated.reset_index()
    centroids_repeated = centroids_repeated.drop(columns=['index'])

    combined = pd.concat([centroids_repeated, feature_df], axis=1)
    combined = combined.drop(columns=['DISTRICT'])

    return combined

data_directory = '/Users/kelseydoerksen/Desktop/Nepal_Landslides_Forecasting_Project/Monsoon2024_Prep'
years = ['2016', '2017', '2018', '2019', '2020', '2021', '2022', '2023']

location_df = pd.read_csv('{}/District_Centroid_Locations.csv'.format(data_directory))
forecast_model = 'UKMO'
ensemble_num = '1'

for y in years:
    feat_df = pd.read_csv('{}/LabelledData/{}/ensemble_{}/{}_windowsize14_district.csv'.format(data_directory, forecast_model,
                                                                                   ensemble_num, y))
    combined_df = add_location_information(feat_df, location_df)
    combined_df.to_csv('{}/LabelledData/{}/ensemble_{}/{}_windowsize14_district.csv'.format(data_directory, forecast_model,
                                                                                   ensemble_num, y))
