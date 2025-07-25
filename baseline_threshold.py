"""
Script for a baseline threshold comparison of the 140mm rainfall threshold
If it rains more than 140mm in one day, label this as landslide
"""

import pandas as pd
from datetime import datetime, timedelta, date

districts = ['Achham', 'Arghakhanchi', 'Baglung', 'Baitadi', 'Bajhang', 'Bajura', 'Banke', 'Bardiya', 'Bhaktapur',
             'Bhojpur', 'Chitawan', 'Dadeldhura', 'Dailekh', 'Dang', 'Darchula', 'Dhading', 'Dhankuta', 'Dolakha',
             'Dolpa', 'Doti', 'Gorkha', 'Gulmi', 'Humla', 'Ilam', 'Jajarkot', 'Jhapa', 'Jumla', 'Kabhrepalanchok',
             'Kailali', 'Kalikot', 'Kanchanpur', 'Kapilbastu', 'Kaski', 'Kathmandu', 'Khotang', 'Lalitpur', 'Lamjung',
             'Mahottari', 'Makawanpur', 'Manang', 'Morang', 'Mugu', 'Mustang', 'Myagdi', 'Nawalparasi_E',
             'Nawalparasi_W', 'Nuwakot', 'Okhaldhunga', 'Palpa', 'Panchthar', 'Parbat', 'Parsa', 'Pyuthan',
             'Ramechhap', 'Rasuwa', 'Rolpa', 'Rukum_E', 'Rukum_W', 'Rupandehi', 'Salyan', 'Sankhuwasabha',
             'Sarlahi', 'Sindhuli', 'Sindhupalchok', 'Solukhumbu', 'Sunsari', 'Surkhet', 'Syangja', 'Tanahu',
             'Taplejung', 'Terhathum', 'Udayapur']

# Set data path
full_path = '/Users/kelseydoerksen/Desktop/Nepal_Landslides_Forecasting_Project/Monsoon2024_Prep'

# GPM data
GPM_folder = '{}/GPM_Mean_Pixelwise/Subseasonal/DistrictLevel/'.format(full_path)

# Landslide incidents
incidents_df = pd.read_csv(full_path + '/Wards_with_Bipad_events_one_to_many_landslides_only.csv')


def daterange(date1, date2):
  date_list = []
  for n in range(int((date2 - date1).days)+1):
    dt = date1 + timedelta(n)
    date_list.append(dt.strftime("%Y-%m-%d"))
  return date_list


def label_data(gpmdf, dates_list, district, landslide_events_df):
    """
    Label district 1, if threshold exceeded, 0 if not
    :param: gpmdf: dataframe of gpm data
    :param: dates_list: list of dates we are querying for
    :param: district: district of interest
    :param: landslide_events_df: dataframe of landslide events for true label
    """
    landslide_events = landslide_events_df[landslide_events_df.District_Proper == district]
    # Filter to remove any duplicates, can't differentiate between
    # two landslides in the same district on the same day
    landslide_events_filt = landslide_events.drop_duplicates(subset=['Date'], keep='first')
    landslide_events_list = landslide_events_filt['Date'].tolist()
    landslide_events_list_formatted = [datetime.strptime(sub, "%d/%m/%Y").strftime('%Y-%m-%d') for sub in
                                       landslide_events_list]

    # Generate list of labelled events for the AOI specified
    label_list = []
    for doy in dates_list:
        if doy in landslide_events_list_formatted:
            label_list.append(1)
        else:
            label_list.append(0)

    # Output to dataframe
    df_labelled_data = pd.DataFrame()
    df_labelled_data['date'] = dates_list
    df_labelled_data['district'] = district
    df_labelled_data['true_label'] = label_list
    df_labelled_data['theshold_label'] = gpmdf['total_precip'] >= 140
    df_labelled_data['theshold_label'] = df_labelled_data['theshold_label']*1
    df_labelled_data['24hour_gpm'] = gpmdf['total_precip']

    return df_labelled_data


def get_24hour_precip(gpm):
    """
    24-hour cumulative precipitation of gpm from avg rate
    """
    gpm['total_precip'] = gpm['avg_precip']*24
    return gpm


def run_generate_labelled_threshold_data(year, start_date, end_date):
    """
    Generate labelled threshold data based on if
    precipitation exceeds 140mm in 24 hours per district
    """
    sdate = start_date
    edate = end_date
    daily_dates = daterange(sdate, edate)
    all_data = pd.DataFrame()
    print('Running for year: {}'.format(year))
    for district in districts:
        print('Generating labels for district: {}'.format(district))
        gpm_df = pd.read_csv('{}/{}_GPM.csv'.format(GPM_folder, district))
        # Get 24 hour precipitation
        gpm_24 = get_24hour_precip(gpm_df)
        # Get labelled df
        df = label_data(gpm_24, daily_dates, district, incidents_df)
        all_data = all_data.append(df)

    all_data.to_csv('/Users/kelseydoerksen/Desktop/Nepal_Landslides_Forecasting_Project/Monsoon2024_Prep/'
                    'Baseline_Threshold/{}_GPM_Threshold.csv'.format(year))


run_generate_labelled_threshold_data('2016', date(2016,11,1), date(2016,12,31))
run_generate_labelled_threshold_data('2017', date(2017,1,1), date(2017,12,31))
run_generate_labelled_threshold_data('2018', date(2018,1,1), date(2018,12,31))
run_generate_labelled_threshold_data('2019', date(2019,1,1), date(2019,12,31))
run_generate_labelled_threshold_data('2020', date(2020,1,1), date(2020,12,31))
run_generate_labelled_threshold_data('2021', date(2021,1,1), date(2021,12,31))
run_generate_labelled_threshold_data('2022', date(2022,1,1), date(2022,12,31))
run_generate_labelled_threshold_data('2023', date(2023,1,1), date(2023,12,31))


