"""
Script for generating samples with hindcast and forecast information
"""
import pandas as pd
from datetime import datetime, date, timedelta
import time
import numpy as np
import argparse
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def get_args():
    parser = argparse.ArgumentParser(description='Generate Feature Space for District Landslide Prediction')
    parser.add_argument("--year", help='Specify year to generate feature space for. Range 2015-2021')
    parser.add_argument("--hindcast_source", help='Hindcast precipitation source. Must be one of GPM or GSMaP')
    parser.add_argument("--forecast_source", help='Forecast precipitation source. Must be one of UKMO, NCEP, KMA.')
    parser.add_argument('--ens_member', help='Ensemble member to use precipitation data from.')
    return parser.parse_args()

# Set data path
root_dir = '/scratch-ssd/kelsen'

# Read in incidents records for 2011-2023
incidents_df = pd.read_csv(root_dir + '/Wards_with_Bipad_events_one_to_many_landslides_only.csv')


# Grab District Names -> Harcoded to match Bipad Portal records
districts = ['Achham', 'Arghakhanchi', 'Baglung', 'Baitadi', 'Bajhang', 'Bajura', 'Banke', 'Bara', 'Bardiya',
             'Bhaktapur', 'Bhojpur', 'Chitawan', 'Dadeldhura', 'Dailekh', 'Dang', 'Darchula', 'Dhading',
             'Dhankuta', 'Dhanusha', 'Dolakha', 'Dolpa', 'Doti', 'Gorkha', 'Gulmi', 'Humla', 'Ilam',
             'Jajarkot', 'Jhapa', 'Jumla', 'Kabhrepalanchok', 'Kailali', 'Kalikot', 'Kanchanpur',
             'Kapilbastu', 'Kaski', 'Kathmandu', 'Khotang', 'Lalitpur', 'Lamjung', 'Mahottari',
             'Makawanpur', 'Manang', 'Morang', 'Mugu', 'Mustang', 'Myagdi', 'Nawalparasi_E',
             'Nawalparasi_W', 'Nuwakot', 'Okhaldhunga', 'Palpa', 'Panchthar', 'Parbat', 'Parsa',
             'Pyuthan', 'Ramechhap', 'Rasuwa', 'Rautahat', 'Rolpa', 'Rukum_E', 'Rukum_W',
             'Rupandehi', 'Salyan', 'Sankhuwasabha', 'Saptari', 'Sarlahi', 'Sindhuli',
             'Sindhupalchok', 'Siraha', 'Solukhumbu', 'Sunsari', 'Surkhet', 'Syangja', 'Tanahu',
             'Taplejung', 'Terhathum', 'Udayapur']

print('The number of landslides on the District-level from 2015-2023 in Nepal is {}'.format(len(incidents_df)))


def daterange(date1, date2):
  date_list = []
  for n in range(int ((date2 - date1).days)+1):
    dt = date1 + timedelta(n)
    date_list.append(dt.strftime("%Y-%m-%d"))
  return date_list


def generate_labelled_district(df, district, dates_list):
    '''
    Generates a dataframe for a district specified with labelled
    landslide occurrences 0,1 respectively
    '''
    landslide_events = df[df.District_Proper == district]
    # Filter to remove any duplicates, can't differentiate between
    # two landslides in the same district on the same day
    landslide_events_filt = landslide_events.drop_duplicates(subset=['Date'], keep='first')
    landslide_events_list = landslide_events_filt['Date'].tolist()
    landslide_events_list_formatted = [datetime.strptime(sub, "%d/%m/%Y").strftime('%Y-%m-%d') for sub in landslide_events_list]

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
    df_labelled_data['label'] = label_list

    return df_labelled_data


def add_geomorphic_features(geo_df, geo_string, district):
  '''
  Generate the geomorphic feature set per district
  '''
  df_filtered = geo_df[geo_df.DISTRICT == district]
  df_reset = df_filtered.reset_index(drop=True)

  # Drop irrelevant columns
  df_stats = df_reset.drop(['OBJECTID', 'PROVINCE', 'PR_NAME', 'DISTRICT'], axis=1)

  # Add suffix with respect to the feature to dataframe
  df_stats.columns = [geo_string + str(col) for col in df_stats.columns]

  return df_stats


def add_landcover_features(modis_df, day):
    '''
    Generate and add landcover features per district
    '''
    year = day[0:4]
    if int(year) > 2021:
        query_year = '2021'
    else:
        query_year = year


    df_lc = pd.DataFrame(columns=['lc_mode', 'lc_var'])
    df_lc['lc_mode'] = [modis_df[modis_df.date == '{}-01-01-12-00'.format(query_year)]['lc_mode'].iloc[0]]
    df_lc['lc_var'] = [modis_df[modis_df.date == '{}-01-01-12-00'.format(query_year)]['lc_var'].iloc[0]]
    return df_lc


def add_precip_lookahead(day, precip_forecast_df, window_size, model_name, ensemble_num):
    '''
    Add precipitation lookahead from model forecast data
    '''
    format = '%Y-%m-%d'
    day = str(day)
    date_dateformat = datetime.strptime(day, format)
    precip_dict = {}
    cumulative_precip = []
    for i in range(window_size):
        increment = i+1
        delta = date_dateformat.date() + timedelta(days=increment)
        lookahead = delta.strftime(format)
        print('running for lookahead {}'.format(lookahead))
        try:
            lookahead_precip = precip_forecast_df.loc[precip_forecast_df['doy'] == str(lookahead), 'avg_precip'].item()
            precip_dict['{}_ens_{}_precip_rate_tplus_{}'.format(model_name, ensemble_num, increment)] = lookahead_precip
            cumulative_precip.append(lookahead_precip * 24)
        except ValueError:
            print('Missing precipitation for date {}, setting as NaN to skip later'.format(lookahead))
            precip_dict['{}_ens_{}_precip_rate_tplus_{}'.format(model_name, ensemble_num, increment)] = np.nan
            cumulative_precip.append(lookahead_precip * 24)

    precip_mean = np.mean(list(precip_dict.values()))
    precip_max = max(precip_dict.values())
    precip_min = min(precip_dict.values())
    precip_dict['{}_ens_{}_precip_mean_precip_rate'.format(model_name, ensemble_num)] = precip_mean
    precip_dict['{}_ens_{}_precip_max_precip_rate'.format(model_name, ensemble_num)] = precip_max
    precip_dict['{}_ens_{}_precip_min_precip_rate'.format(model_name, ensemble_num)] = precip_min
    precip_dict['{}_ens_{}_precip_total_cumulative_precipitation'.format(model_name, ensemble_num)]\
        = sum(cumulative_precip)

    return pd.DataFrame([precip_dict])


def generate_precip_lookback(day, precip, window_size, operational_mode=False):
    '''
    Generate Precipitation lookback features given day0, precipitation df
    :param: operaiotnal_mode: refers to operational-use context, querying from tminus days bwack
    '''
    format = '%Y-%m-%d'
    day = str(day)
    date_dateformat = datetime.strptime(day, format)
    precip_dict = {}
    cumulative_precip = []
    for i in range(window_size):
        delta = date_dateformat.date() - timedelta(days=i)
        lookback = delta.strftime(format)
        print('running for lookback {}'.format(lookback))
        lookback_precip = precip.loc[precip['doy'] == str(lookback),
                               'avg_precip'].item()
        precip_dict['precip_rate_tminus_{}'.format(i)] = lookback_precip
        cumulative_precip.append(lookback_precip * 24)

    # check if operational context, if so let's drop the first two lookback days
    if operational_mode:
        entries_to_remove = ('precip_rate_tminus_0', 'precip_rate_tminus_1')
        for k in entries_to_remove:
            precip_dict.pop(k, None)
        cumulative_precip = cumulative_precip[2:]

    precip_mean = np.mean(list(precip_dict.values()))
    precip_max = max(precip_dict.values())
    precip_min = min(precip_dict.values())
    precip_dict['gpm_mean_precip_rate'] = precip_mean
    precip_dict['gpm_max_precip_rate'] = precip_max
    precip_dict['gpm_min_precip_rate'] = precip_min
    precip_dict['gpm_total_cumulative_precipitation'] = sum(cumulative_precip)

    return pd.DataFrame([precip_dict])


def generate_landslide_label(label_df, day, window_size):
    '''
    Generate landslide label based on lookahead window size
    if landslide occurred over the next t+window size days
    '''
    format = '%Y-%m-%d'
    start_date = str(day)
    date_dateformat = datetime.strptime(start_date, format)
    labels = []
    for i in range(window_size):
        increment = i+1
        delta = date_dateformat.date() + timedelta(days=increment)
        lookahead = delta.strftime(format)
        df_filtered = label_df[label_df.date == lookahead]
        day_label = df_filtered['label'].values
        labels.append(day_label[0])

    final_label = sum(labels)
    if final_label > 0:
        landslide_label = 1
    else:
        landslide_label = 0

    label_dict = {'label': landslide_label}

    return pd.DataFrame([label_dict])


def generate_labelled_xdays_sample(filepath, label_df, district, day, window_size,
                                   precip_lookback, precip_lookahead, forecast_name, ensemble_number):
    '''
    For each unique date in date range of study, create feature space
    '''

    # Grab GPM Precipitation features
    df_gpm_precip = generate_precip_lookback(day, precip_lookback, window_size)

    # Grab geomorphic features
    dem = pd.read_csv(filepath +
                       '/DEM-Derived_District/dem_stats_all_district.csv')
    aspect = pd.read_csv(filepath +
                          '/DEM-Derived_District/aspect_stats_all_district.csv')
    slope = pd.read_csv(filepath +
                         '/DEM-Derived_District/slope_stats_all_district.csv')

    dem_df = add_geomorphic_features(dem, 'dem', district)
    slope_df = add_geomorphic_features(slope, 'slope', district)
    aspect_df = add_geomorphic_features(aspect, 'aspect', district)

    # Add landcover features -> only available 2001-2021 for MODIS
    modis = pd.read_csv(filepath + '/MODIS_District_2015-2021/{}_Modis_2015-2021.csv'.format(district))
    modis_df = add_landcover_features(modis, day)

    # Add precip lookahead features
    df_precip_forecast = add_precip_lookahead(day, precip_lookahead, window_size, forecast_name, ensemble_number)

    # Generate landslide label
    label = generate_landslide_label(label_df, day, window_size)

    labelled_df = df_gpm_precip.join(dem_df).join(aspect_df).join(slope_df).join(modis_df).join(df_precip_forecast).\
        join(label)
    # add date and distric location
    labelled_df['date'] = day
    labelled_df['district'] = district

    return labelled_df


def check_if_date_valid(query_date, precipitation_df, direction):
    """
    Function to check if the date queried
    has lookahead and lookback data, returns
    True or False
    """
    format = '%Y-%m-%d'
    day = str(query_date)
    date_dateformat = datetime.strptime(day, format)

    for i in range(14):
        if direction == 'lookback':
            delta = date_dateformat.date() - timedelta(days=i)
        else:
            delta = date_dateformat.date() + timedelta(days=i)
        timestep = delta.strftime(format)
        print('running for lookback {}'.format(timestep))
        if timestep not in precipitation_df['doy'].values:
            return False

    return True


def run_generate_data(root_dir, year, start_date, end_date, window_size, forecast_model, forecast_ensemble,
                      hindcast_data):
    '''
    :param year: year of data gathered
    :param start_date: format date(YYYY,MM,DD)
    :param end_date: format date(YYYY,MM,DD)
    of data availability for GPM and forecast data
    :return: csv file of labelled data
    '''
    sdate = start_date
    edate = end_date
    daily_dates = daterange(sdate, edate)
    window_size = window_size
    print('start date is {}'.format(sdate))
    print('end date is {}'.format(edate))
    print('window size is {}'.format(window_size))
    # Call function to generate labelled CSV files
    all_data = pd.DataFrame()
    total_districts = len(districts)
    count = total_districts
    start_time = time.time()
    for district in districts:
        print('Generating samples for district {}'.format(district))
        # Generate labels
        labelled_df = generate_labelled_district(incidents_df, district, daily_dates)
        # Grab Precipitation data for LOOKBACK

        precip_lookback = pd.read_csv('{}/{}_Mean_Pixelwise/Subseasonal/DistrictLevel/{}_{}.csv'.format(root_dir,
                                                                                                         hindcast_data,
                                                                                                         district,
                                                                                                        hindcast_data))

        # Grab Precipitation data for Forecast LOOKAHEAD
        precip_forecast = pd.read_csv('{}/PrecipitationModel_Forecast_Data/Subseasonal/{}/DistrictLevel/'
                                      'ensemble_member_{}/{}_subseasonal_precipitation.csv'.
                                      format(root_dir, forecast_model,  forecast_ensemble, district))

        for i in range(len(daily_dates) - window_size):
            if not check_if_date_valid(daily_dates[i], precip_lookback, 'lookback'):
                print('We dont have all the lookback information needed, skipping sample for doy: {}'.
                      format(daily_dates[i]))
                continue
            if not check_if_date_valid(daily_dates[i], precip_forecast, 'lookahead'):
                print('We dont have all the lookahead information needed, skipping sample for doy: {}'.
                      format(daily_dates[i]))
                continue

            # check dates for lookahead and lookback precip
            x_day_labelled_df = generate_labelled_xdays_sample(root_dir, labelled_df,
                                                                   district, daily_dates[i],
                                                                   window_size, precip_lookback,
                                                                   precip_forecast, forecast_model, forecast_ensemble)
            all_data = all_data.append(x_day_labelled_df)
        count = count - 1
        print('%.2f percent complete' % (100 * (1 - count / total_districts)))
        print('{} districts left to generate samples for'.format(count))
    # Saving data for future use
    if forecast_model == 'ecmwf':
        all_data.to_csv('{}/LabelledData_{}/{}/{}_windowsize{}_district.csv'.format(root_dir, hindcast_data,
                                                                                    forecast_model, year, window_size))
    else:
        all_data.to_csv('{}/LabelledData_{}/{}/ensemble_{}/{}_windowsize{}_district.csv'.format(root_dir, hindcast_data,
                                                                                                forecast_model,
                                                                                                forecast_ensemble,
                                                                                                year, window_size))
    print('The program took {}s to run'.format(time.time()-start_time))


if __name__ == "__main__":
    args = get_args()
    year = args.year
    hindcast = args.hindcast_source
    forecast = args.forecast_source
    ens_num = args.ens_member

    run_generate_data(root_dir, year, date(int(year), 1, 1), date(int(year)+1, 1, 14), 14, forecast, ens_num, hindcast)

