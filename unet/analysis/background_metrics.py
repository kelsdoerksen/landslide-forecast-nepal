"""
Purpose of this script is to create some nice metrics
about the landslide records themselves for the journal paper
"""

import pandas as pd
from datetime import timedelta
from datetime import datetime

# Loading in the landslide records from Bipad portal
df = pd.read_csv('/Volumes/PRO-G40/landslides/Nepal_Landslides_Forecasting_Project/Nature_Comms/'
                 'BipadPortal_Downloaded_08-04-2025.csv')

# Dictionary of monsoon start and end dates from https://www.dhm.gov.np/uploads/dhm/climateService/monsoon_onset_and_withdrawal_dates_english2.pdf
monsoon_dates = {
    2011: {'start': '2011-06-15', 'end': '2011-10-07'},
    2012: {'start': '2012-06-16', 'end': '2012-09-28'},
    2013: {'start': '2013-06-14', 'end': '2013-10-19'},
    2014: {'start': '2014-06-20', 'end': '2014-10-07'},
    2015: {'start': '2015-06-13', 'end': '2015-10-03'},
    2016: {'start': '2016-06-15', 'end': '2016-10-12'},
    2017: {'start': '2017-06-12', 'end': '2017-10-16'},
    2018: {'start': '2018-06-08', 'end': '2018-10-05'},
    2019: {'start': '2019-06-20', 'end': '2019-10-12'},
    2020: {'start': '2020-06-12', 'end': '2020-10-16'},
    2021: {'start': '2021-06-11', 'end': '2021-10-11'},
    2022: {'start': '2022-06-05', 'end': '2022-10-16'},
    2023: {'start': '2023-06-14', 'end': '2023-10-15'},
    2024: {'start': '2024-06-10', 'end': '2024-10-12'}
}

# Setting district dictionary all values to 0 to populate them with the landslide count for further plotting
district_dict = {'Bhojpur': 0, 'Dhankuta': 0, 'Ilam': 0, 'Jhapa': 0, 'Khotang': 0, 'Morang': 0, 'Okhaldhunga': 0,
    'Panchthar': 0, 'Sankhuwasabha': 0, 'Solukhumbu': 0, 'Sunsari': 0, 'Taplejung': 0, 'Terhathum': 0,
    'Udayapur': 0, 'Bara': 0, 'Dhanusha': 0, 'Mahottari': 0, 'Parsa': 0, 'Rautahat': 0, 'Saptari': 0,
    'Sarlahi': 0, 'Siraha': 0, 'Bhaktapur': 0, 'Chitwan': 0, 'Dhading': 0, 'Dolakha': 0,
    'Kabhrepalanchok': 0, 'Kathmandu': 0, 'Lalitpur': 0, 'Makawanpur': 0, 'Nuwakot': 0, 'Ramechhap': 0,
    'Rasuwa': 0, 'Sindhuli': 0, 'Sindhupalchok': 0, 'Baglung': 0, 'Gorkha': 0, 'Kaski': 0, 'Lamjung': 0,
    'Manang': 0, 'Mustang': 0, 'Myagdi': 0, 'Nawalparasi West': 0, 'Parbat': 0, 'Syangja': 0, 'Tanahu': 0,
    'Arghakhanchi': 0, 'Banke': 0, 'Bardiya': 0, 'Dang': 0, 'Gulmi': 0, 'Kapilbastu': 0, 'Palpa': 0,
    'Nawalparasi East': 0, 'Pyuthan': 0, 'Rolpa': 0, 'Rukum East': 0, 'Rupandehi': 0, 'Dailekh': 0, 'Dolpa': 0,
    'Humla': 0, 'Jajarkot': 0, 'Jumla': 0, 'Kalikot': 0, 'Mugu': 0, 'Rukum West': 0, 'Salyan': 0,
    'Surkhet': 0, 'Achham': 0, 'Baitadi': 0, 'Bajhang': 0, 'Bajura': 0, 'Dadeldhura': 0, 'Darchula': 0,
    'Doti': 0, 'Kailali': 0, 'Kanchanpur': 0}

results_dict = {}

def daterange(date1, date2):
    date_list = []
    format_str = '%Y-%m-%d'
    date1_datetime_obj = datetime.strptime(date1, format_str)
    date2_datetime_obj = datetime.strptime(date2, format_str)

    for n in range(int((date2_datetime_obj - date1_datetime_obj).days) + 1):
        dt = date1_datetime_obj + timedelta(n)
        date_list.append(dt.strftime("%Y-%m-%d"))
    return date_list

months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
df['monsoon'] = 0
landslide_timeseries_dict = {}

for y in monsoon_dates.keys():
    monsoon_date_list = daterange(monsoon_dates[y]['start'], monsoon_dates[y]['end'])
    year_list = daterange('{}-01-01'.format(y), '{}-12-31'.format(y))
    df.loc[df['Incident on'].isin(monsoon_date_list), 'monsoon'] = 1
    total_incident_count = len(df.loc[df['Incident on'].isin(year_list), 'monsoon'])
    monsoon_incident_count = len(df.loc[df['Incident on'].isin(monsoon_date_list), 'monsoon'])

    print('Total landslide incidents in {} is: {}'.format(y, total_incident_count))
    print('# of landslide incidents during {} monsoon season is: {}'.format(y, monsoon_incident_count))
    print('-------------------')
    time_series_counts = []
    for m in months:
        subset_df = df[df['Incident on'].str.contains('{}-{}'.format(y, m))]
        time_series_counts.append(int(len(subset_df)))
    landslide_timeseries_dict['{}'.format(y)] = time_series_counts

    # Subset to get count per District for future plotting in QGIS
    year_dict = district_dict.copy()
    df_y = df[df['Incident on'].str.contains('{}'.format(y))]
    counts = df_y.District.value_counts()
    province_counts = df_y.Province.value_counts()

    # Generate for the districts
    for k in counts.keys():
        year_dict['{}'.format(k)] = int(counts[k])
    year_df = pd.DataFrame(list(year_dict.items()), columns=["District", "Landslide_count"])
    year_df.to_csv('/Volumes/PRO-G40/landslides/Nepal_Landslides_Forecasting_Project/Nature_Comms/{}_LandslideCount_Per_District.csv'.format(y))

    # Generate for the provinces
    year_province_dict = {}
    for k in province_counts.keys():
        year_province_dict['{}'.format(k)] = int(province_counts[k])
    year_province_df = pd.DataFrame(list(year_province_dict.items()), columns=["Province", "Landslide_count"])
    year_province_df.to_csv('/Volumes/PRO-G40/landslides/Nepal_Landslides_Forecasting_Project/Nature_Comms/{}_LandslideCount_Per_Province.csv'.format(y))

landslides_timeseries_df = pd.DataFrame(landslide_timeseries_dict)
#landslides_timeseries_df.to_csv('/Volumes/PRO-G40/landslides/Nepal_Landslides_Forecasting_Project/Nature_Comms/landslide_timeseries_per_month.csv')
#df.to_csv('/Volumes/PRO-G40/landslides/Nepal_Landslides_Forecasting_Project/Nature_Comms/BipadPortal_Downloaded_08-04-2025_MonsoonLabel.csv')

results_dict['monsoon_landslides'] = df.monsoon.sum()
results_dict['monsoon_landslides_percentage'] = df.monsoon.sum()/len(df)

province_metrics = pd.DataFrame()
district_metrics = pd.DataFrame()


# Province occurrences
df.Province.value_counts()

# Percentages
df.Province.value_counts()/len(df)

# District occurrences
df.District.value_counts()

# Percentages
df.District.value_counts()/len(df)