import numpy as np
from PIL import Image


def generate_district_masks(file_name):
    '''
    Create the masks for each district from Nepal raster
    '''
    # Load in Nepal District file
    im = Image.open('{}'.format(file_name))
    array = np.array(im)

    # Create dict to match pixel values
    district_dict = {'Bhojpur': 1, 'Dhankuta': 2, 'Ilam': 3, 'Jhapa': 4, 'Khotang': 5, 'Morang': 6, 'Okhaldhunga': 7,
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
    'Doti': 75, 'Kailali': 76, 'Kanchanpur': 77}

    new_dict = {}
    for k, v in district_dict.items():
        # Only include District of interest
        new_array = array.copy()
        new_array[new_array != v] = 0
        # Bound between 0 and 1
        filtered_array = new_array.copy()
        # Landslide class set to 1
        filtered_array[filtered_array == v] = 1
        new_dict[k] = filtered_array

    return new_dict


def landslide_record_gen(y_true, d_masks):
    '''
    Record the landslide predicted by model and true landslide locations for further plotting in future
    :param: y_true: label
    :param: d_masks: dictionary of districts
    '''

    true_landsliding_districts = []
    predicted_landsliding_districts = []
    groundtruth_landslides = []
    non_landslide_districts = d_masks.copy()  # copy of landslides dict to manipulate
    # Get what districts are in label
    district_pixels = []
    landslides = np.where(y_true == 1)
    points_list = []
    gt_landslide_dict = {}
    for k in range(len(landslides[0])):
        points_list.append([landslides[0][k], landslides[1][k]])
    for district in d_masks:
        if all(item in points_list for item in d_masks[district]):
            district_pixels.append(d_masks[district])
            non_landslide_districts.pop(district)
            groundtruth_landslides.append(district)
            gt_landslide_dict[district] = d_masks[district]

        true_landsliding_districts.append(groundtruth_landslides)

    return true_landsliding_districts


root_dir = '/Volumes/PRO-G40/landslides/Nepal_Landslides_Forecasting_Project/Monsoon2024_Prep'

# Generating district masks to use for the precision, recall
district_masks = generate_district_masks('{}/District_Labels.tif'.format(root_dir))
for district in district_masks:
    points = []
    for i in range(len(district_masks[district][0])):
        points.append([district_masks[district][0][i], district_masks[district][1][i]])
    district_masks[district] = points

# load array
arr = np.load('{}/Binary_Landslide_Labels_14day/label_2023-09-21.npy'.format(root_dir))

# get gt districts
gt_landslide_districts = landslide_record_gen(arr, district_masks)

print(gt_landslide_districts)














