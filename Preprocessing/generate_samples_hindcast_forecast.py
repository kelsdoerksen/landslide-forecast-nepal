"""
Script for generating samples with hindcast and forecast information
"""
import pandas as pd
from datetime import datetime, date, timedelta
import time
import numpy as np
import argparse
import warnings
import os

PROJECT_DIR = os.environ.get('LANDSLIDE_PROJECT_DIR', 'data')  # see README > Data
warnings.simplefilter(action='ignore', category=FutureWarning)

def get_args():
    parser = argparse.ArgumentParser(description='Generate Feature Space for District Landslide Prediction')
    parser.add_argument("--year", help='Specify year to generate feature space for.')
    parser.add_argument("--hindcast_source", help='Hindcast precipitation source. Must be one of GPM or GSMaP')
    parser.add_argument("--forecast_source", help='Forecast precipitation source. Must be one of UKMO, NCEP, KMA.')
    parser.add_argument('--ens_member', help='Ensemble member to use precipitation data from.')
    parser.add_argument('--forecast_district_suffix', default='',
                        help="Suffix on the forecast DistrictLevel input dir, e.g. '_latfix'.")
    parser.add_argument('--labelled_suffix', default='',
                        help="Suffix on the LabelledData output dir, e.g. '_latfix'.")
    parser.add_argument('--modis_dir', default='MODIS_District_2011-2023',
                        help="Land-cover folder under root_dir. Files inside are named "
                             "<district>_Modis_<tag>.csv where <tag> is whatever follows "
                             "'MODIS_District_' in this name.")
    parser.add_argument('--modis_file_suffix', default='_Modis_2011-2023.csv',
                        help="Filename suffix inside --modis_dir. The 2011-2023 folder uses "
                             "'_Modis_2011-2023.csv'; the 2011-2026 folder uses '_Modis.csv'.")
    parser.add_argument('--modis_max_year', default='2023',
                        help='Latest land-cover year available. Sample years after this are '
                             'clamped to it. Raise it when a folder with newer layers is used.')
    return parser.parse_args()

# Set data path
root_dir = PROJECT_DIR + '/Monsoon2024_Prep'
#root_dir = '$LANDSLIDE_PROJECT_DIR/Monsoon2024_Prep/2024_Season_Retro/'

# Read in incidents records
#incidents_df = pd.read_csv('{}/Wards_with_Bipad_events_one_to_many_landslides_only.csv'.format(root_dir))
#incidents_df = pd.read_csv(root_dir + 'incidents_April_October_2024.csv')
# Combined, de-duplicated landslide labels 2016-2026 (built by build_incident_labels.py:
# merges the 3 Bipad downloads, ISO dates, canonical district names, Hazard==Landslide).
incidents_df = pd.read_csv('{}/combined_landslide_incidents_2016-2026.csv'.format(root_dir))

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

print('The number of landslides on the District-level in Nepal is {}'.format(len(incidents_df)))

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
    landslide_events = df[df.District == district]
    # Filter to remove any duplicates, can't differentiate between
    # two landslides in the same district on the same day
    landslide_events = landslide_events.rename(columns={'Incident on': 'Date'})
    landslide_events_filt = landslide_events.drop_duplicates(subset=['Date'], keep='first')
    landslide_events_list = landslide_events_filt['Date'].tolist()
    landslide_events_list_formatted = [datetime.strptime(sub, "%Y-%m-%d").strftime('%Y-%m-%d') for sub in landslide_events_list] # 2025-2026 Bipad list (ISO dates)
    #landslide_events_list_formatted = [datetime.strptime(sub, "%m/%d/%y").strftime('%Y-%m-%d') for sub in landslide_events_list] # 2024 list
    #landslide_events_list_formatted = [datetime.strptime(sub, "%d/%m/%Y").strftime('%Y-%m-%d') for sub in landslide_events_list] # 2016-2023 list
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


# The 2011-2026 land-cover export names four districts the way the shapefile does,
# not the way the rest of this pipeline does. Same mapping the notebook's map cell
# uses (Nawalparasi was split into Nawalpur in the east and Parasi in the west).
MODIS_NAME_ALIASES = {
    'Nawalparasi_E': 'Nawalpur',
    'Nawalparasi_W': 'Parasi',
    'Rukum_E': 'Rukum East',
    'Rukum_W': 'Rukum West',
}


def read_modis(root_dir, modis_dir, district, file_suffix):
    '''Read one district's land-cover CSV, accepting either naming convention.'''
    for name in (district, MODIS_NAME_ALIASES.get(district)):
        if name is None:
            continue
        path = '{}/{}/{}{}'.format(root_dir, modis_dir, name, file_suffix)
        if os.path.exists(path):
            return pd.read_csv(path)
    raise FileNotFoundError('No land-cover CSV for {} in {}/{} (tried suffix {})'.format(
        district, root_dir, modis_dir, file_suffix))


def add_landcover_features(modis_df, day, max_year='2023'):
    '''
    Generate and add landcover features per district

    Sample years beyond the last available land-cover layer are clamped to it.
    max_year must match the folder actually being read - training and inference
    have to agree on which layer a given year gets, or lc_mode/lc_var mean
    different things at fit time and at score time.
    '''
    year = day[0:4]
    if int(year) > int(max_year):
        query_year = str(max_year)
    else:
        query_year = year


    df_lc = pd.DataFrame(columns=['lc_mode', 'lc_var'])
    df_lc['lc_mode'] = [modis_df[modis_df.date == '{}-01-01-12-00'.format(query_year)]['lc_mode'].iloc[0]]
    df_lc['lc_var'] = [modis_df[modis_df.date == '{}-01-01-12-00'.format(query_year)]['lc_var'].iloc[0]]
    return df_lc


def add_precip_lookahead(day, precip_forecast, window_size, model_name, ensemble_num):
    '''
    Add precipitation lookahead from model forecast data

    precip_forecast is a {doy: avg_precip} dict (see to_series). This used to scan
    the whole dataframe per lookup, which dominated the runtime; the values are
    identical.
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
        lookahead_precip = precip_forecast[lookahead]
        precip_dict['{}_ens_{}_precip_rate_tplus_{}'.format(model_name, ensemble_num, increment)] = lookahead_precip
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


def generate_precip_lookback(day, precip, window_size, lookback_name):
    '''
    Generate Precipitation lookback features given day0, precipitation dict
    :param: precip: {doy: avg_precip} dict (see to_series)
    '''
    format = '%Y-%m-%d'
    day = str(day)
    date_dateformat = datetime.strptime(day, format)
    precip_dict = {}
    cumulative_precip = []
    for i in range(window_size):
        delta = date_dateformat.date() - timedelta(days=i)
        lookback = delta.strftime(format)
        lookback_precip = precip[lookback]
        precip_dict['precip_rate_tminus_{}'.format(i)] = lookback_precip
        cumulative_precip.append(lookback_precip * 24)

    precip_mean = np.mean(list(precip_dict.values()))
    precip_max = max(precip_dict.values())
    precip_min = min(precip_dict.values())
    precip_dict['{}_mean_precip_rate'.format(lookback_name)] = precip_mean
    precip_dict['{}_max_precip_rate'.format(lookback_name)] = precip_max
    precip_dict['{}_min_precip_rate'.format(lookback_name)] = precip_min
    precip_dict['{}_total_cumulative_precipitation'.format(lookback_name)] = sum(cumulative_precip)

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
                                   precip_lookback, precip_lookahead, forecast_name, ensemble_number,
                                   hindcast_name, static, modis_max_year='2023'):
    '''
    For each unique date in date range of study, create feature space

    static is (dem_df, aspect_df, slope_df, modis) for this district, read once by
    the caller. These four CSVs used to be re-read from disk for every single
    district-day, which was most of the runtime; the values are identical.
    '''

    # Grab Lookback
    df_lookback_precip = generate_precip_lookback(day, precip_lookback, window_size, hindcast_name)

    dem_df, aspect_df, slope_df, modis = static

    # Add landcover features (depends on the day's year, so still per-day)
    modis_df = add_landcover_features(modis, day, modis_max_year)

    # Add precip lookahead features
    df_precip_forecast = add_precip_lookahead(day, precip_lookahead, window_size, forecast_name, ensemble_number)

    # Generate landslide label
    label = generate_landslide_label(label_df, day, window_size)

    labelled_df = df_lookback_precip.join(dem_df).join(aspect_df).join(slope_df).join(modis_df).join(df_precip_forecast).\
        join(label)
    # add date and district location
    labelled_df['date'] = day
    labelled_df['district'] = district

    return labelled_df


def to_series(df):
    """{doy: avg_precip} for one district. Asserts one row per day."""
    doy = df['doy'].astype(str)
    assert not doy.duplicated().any(), 'Duplicate doy rows in a precipitation CSV'
    return dict(zip(doy, df['avg_precip'].astype(float)))


def check_if_date_valid(query_date, precip_days, direction):
    """
    Function to check if the date queried
    has lookahead and lookback data, returns
    True or False

    precip_days is a set of available doy strings.
    """
    format = '%Y-%m-%d'
    day = str(query_date)
    date_dateformat = datetime.strptime(day, format)

    for i in range(14):
        if direction == 'lookback':
            delta = date_dateformat.date() - timedelta(days=i)
        else:
            # add_precip_lookahead reads day+1 .. day+14, so validate that exact window
            delta = date_dateformat.date() + timedelta(days=i + 1)
        if delta.strftime(format) not in precip_days:
            return False

    return True


def run_generate_data(root_dir, year, start_date, end_date, window_size, forecast_model, forecast_ensemble,
                      hindcast_data, forecast_district_suffix='', labelled_suffix='',
                      modis_dir='MODIS_District_2011-2023',
                      modis_file_suffix='_Modis_2011-2023.csv', modis_max_year='2023'):
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
    all_frames = []
    total_districts = len(districts)
    count = total_districts
    start_time = time.time()

    # Read the geomorphic tables once for the whole run, not once per district-day.
    dem_all = pd.read_csv(root_dir + '/DEM-Derived_District/dem_stats_all_district.csv')
    aspect_all = pd.read_csv(root_dir + '/DEM-Derived_District/aspect_stats_all_district.csv')
    slope_all = pd.read_csv(root_dir + '/DEM-Derived_District/slope_stats_all_district.csv')

    forecast_dir = ('{}/PrecipitationModel_Forecast_Data/Subseasonal/{}/DistrictLevel{}/'
                    'ensemble_member_{}'.format(root_dir, forecast_model, forecast_district_suffix,
                                                forecast_ensemble))
    print('forecast lookahead source: {}'.format(forecast_dir))
    print('land cover: {} (years after {} clamped to {})'.format(modis_dir, modis_max_year, modis_max_year))

    for district in districts:
        print('Generating samples for district {}'.format(district))
        # Generate labels
        labelled_df = generate_labelled_district(incidents_df, district, daily_dates)

        # Grab Precipitation data for LOOKBACK
        precip_lookback_df = pd.read_csv('{}/{}_Mean_Pixelwise/Subseasonal/DistrictLevel/{}_{}.csv'.format(root_dir,
                                                                                                         hindcast_data,
                                                                                                         district,
                                                                                                        hindcast_data))

        # Grab Precipitation data for Forecast LOOKAHEAD
        precip_forecast_df = pd.read_csv('{}/{}_subseasonal_precipitation.csv'.format(forecast_dir, district))

        precip_lookback = to_series(precip_lookback_df)
        precip_forecast = to_series(precip_forecast_df)
        lookback_days = set(precip_lookback)
        lookahead_days = set(precip_forecast)

        static = (add_geomorphic_features(dem_all, 'dem', district),
                  add_geomorphic_features(aspect_all, 'aspect', district),
                  add_geomorphic_features(slope_all, 'slope', district),
                  read_modis(root_dir, modis_dir, district, modis_file_suffix))

        for i in range(len(daily_dates) - window_size):
            if not check_if_date_valid(daily_dates[i], lookback_days, 'lookback'):
                continue
            if not check_if_date_valid(daily_dates[i], lookahead_days, 'lookahead'):
                continue

            # check dates for lookahead and lookback precip
            x_day_labelled_df = generate_labelled_xdays_sample(root_dir, labelled_df, district, daily_dates[i],
                                                               window_size, precip_lookback,
                                                               precip_forecast, forecast_model, forecast_ensemble,
                                                               hindcast_data, static, modis_max_year)
            all_frames.append(x_day_labelled_df)
        count = count - 1
        print('%.2f percent complete' % (100 * (1 - count / total_districts)))
    # Saving data for future use
    save_dir = '{}/LabelledData_{}{}/{}/ensemble_{}'.format(root_dir, hindcast_data, labelled_suffix,
                                                            forecast_model, forecast_ensemble)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    all_data = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    all_data.to_csv('{}/{}_windowsize{}_district.csv'.format(save_dir, year, window_size))
    print('The program took {}s to run'.format(time.time()-start_time))


if __name__ == "__main__":
    args = get_args()
    year = args.year
    hindcast = args.hindcast_source
    forecast = args.forecast_source
    ens_num = args.ens_member

    #run_generate_data(root_dir, year, date(int(year), 1, 1), date(int(year)+1, 1, 14), 14, forecast, ens_num, hindcast)
    run_generate_data(root_dir, year, date(int(year), 1, 1), date(int(year), 12, 31), 14, forecast, ens_num, hindcast,
                      forecast_district_suffix=args.forecast_district_suffix,
                      labelled_suffix=args.labelled_suffix,
                      modis_dir=args.modis_dir, modis_file_suffix=args.modis_file_suffix,
                      modis_max_year=args.modis_max_year)
