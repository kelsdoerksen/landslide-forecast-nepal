"""
Script to generate landslide labels as np arrays
"""

import numpy as np
from PIL import Image
from datetime import datetime, timedelta, date
import pandas as pd

# Setting data directories to query from
#root_dir = '/Volumes/PRO-G40/landslides/Nepal_Landslides_Forecasting_Project/Monsoon2024_Prep'
root_dir = '/Volumes/PRO-G40/landslides/Nepal_Landslides_Forecasting_Project/Monsoon2024_Prep/2024_Season_Retro'

# Loading landslide records and Nepal District Array
#landslide_records = pd.read_csv('{}/Wards_with_Bipad_events_one_to_many_landslides_only.csv'.format(root_dir))
#landslide_records = pd.read_csv('{}/incidents_April_October_2024.csv'.format(root_dir))
landslide_records = pd.read_csv('{}/incidents_2024_Downloaded_14-04-2025.csv'.format(root_dir))
nepal_im = Image.open('{}/District_Labels.tif'.format(root_dir))


district_dict = {
    'Bhojpur': 1, 'Dhankuta': 2, 'Ilam': 3, 'Jhapa': 4, 'Khotang': 5, 'Morang': 6, 'Okhaldhunga': 7,
    'Panchthar': 8, 'Sankhuwasabha': 9, 'Solukhumbu': 10, 'Sunsari': 11, 'Taplejung': 12, 'Terhathum': 13,
    'Udayapur': 14, 'Bara': 15, 'Dhanusha': 16, 'Mahottari': 17, 'Parsa': 18, 'Rautahat': 19, 'Saptari': 20,
    'Sarlahi': 21, 'Siraha': 22, 'Bhaktapur': 23, 'Chitawan': 24, 'Dhading': 25, 'Dolakha': 26,
    'Kabhrepalanchok': 27, 'Kathmandu': 28, 'Lalitpur': 29, 'Makawanpur': 30, 'Nuwakot': 31, 'Ramechhap': 32,
    'Rasuwa': 33, 'Sindhuli': 34, 'Sindhupalchok': 35, 'Baglung': 36, 'Gorkha': 37, 'Kaski': 38, 'Lamjung': 39,
    'Manang': 40, 'Mustang': 41, 'Myagdi': 42, 'Nawalparasi_W': 43, 'Parbat': 44, 'Syangja': 45, 'Tanahu': 46,
    'Arghakhanchi': 47, 'Banke': 48, 'Bardiya': 49, 'Dang': 50, 'Gulmi': 51, 'Kapilbastu': 52, 'Palpa': 53,
    'Nawalparasi_E': 54, 'Pyuthan': 55, 'Rolpa': 56, 'Rukum_E': 57, 'Rupandehi': 58, 'Dailekh': 59, 'Dolpa': 60,
    'Humla': 61, 'Jajarkot': 62, 'Jumla': 63, 'Kalikot': 64, 'Mugu': 65, 'Rukum_W': 66, 'Salyan': 67,
    'Surkhet': 68, 'Achham': 78, 'Baitadi': 70, 'Bajhang': 71, 'Bajura': 72, 'Dadeldhura': 73, 'Darchula': 74,
    'Doti': 75, 'Kailali': 76, 'Kanchanpur': 77
}


def daterange(date1, date2):
  date_list = []
  for n in range(int ((date2 - date1).days)+1):
    dt = date1 + timedelta(n)
    date_list.append(dt.strftime("%Y-%m-%d"))
  return date_list


def generate_daily_labels(doy, landslide_df):
    day = str(doy)
    #day_formatted = datetime.strptime(day, "%Y-%m-%d").strftime("%d/%m/%Y")
    #dt_formatted = datetime.strptime(day_formatted, "%d/%m/%Y")

    day_formatted = datetime.strptime(day, "%Y-%m-%d").strftime("%d/%m/%y")
    dt_formatted = datetime.strptime(day_formatted, "%d/%m/%y")

    landslide_list = []
    for i in range(14):
        nepal_arr = np.array(nepal_im)
        increment = i + 1
        delta = dt_formatted.date() + timedelta(days=increment)
        #lookahead = delta.strftime("%d/%m/%Y")
        lookahead = delta.strftime("%m/%d/%y")
        # Removing 0 at the beginning as this is not in the 2024 monsoon records
        if lookahead[0] == '0':
            lookahead = lookahead[1:]
        landslide_subset = landslide_df[landslide_df['Incident on'] == lookahead]
        if len(landslide_subset) == 0:
            landslide_list.append(np.zeros((60, 100)))
        else:
            for district in list(district_dict.keys()):
                district_val = district_dict[district]
                if len(landslide_subset[landslide_subset['District'] == district]) == 0:
                    nepal_arr[nepal_arr == district_val] = 0
            nepal_arr[nepal_arr>=1] = 1
            landslide_list.append(nepal_arr)

    combined_label = np.array(landslide_list).sum(axis=0)
    combined_label[combined_label>=1] = 1
    np.save('{}/Binary_Landslide_Labels_14day/{}.npy'.format(root_dir, day), combined_label)

    return combined_label


date_list = daterange(date(2024,4,13), date(2024,10,31))
for d in date_list:
    print('Generating label for doy: {}'.format(d))
    generate_daily_labels(d, landslide_records)







