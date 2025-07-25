from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
import copy
import argparse
from datetime import datetime, timedelta


def get_args():
    parser = argparse.ArgumentParser(description='Generating Confusion Matrix')
    parser.add_argument('--run', default=None, help='Wandb run name')
    parser.add_argument('--root_dir', help='Root directory of data')
    parser.add_argument('--test_year', help='Test directory')
    return parser.parse_args()


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


def generate_confusion_matrix(df, threshold=0.2):
    """
    Calculates confusion matrix
    """
    # Setting based on threshold
    df['model soft predictions'] = df['model soft predictions'] >= threshold
    y_test = df['groundtruth']
    y_pred = df['model soft predictions']*1

    CM = confusion_matrix(y_test, y_pred)

    return CM


def generate_plot(CM, start_date, end_date, threshold, save_dir, save_date):
    """
    Generate cfm plot
    """
    TN = CM[0][0]
    FN = CM[1][0]
    TP = CM[1][1]
    FP = CM[0][1]

    FP_Rate = FP / (FP + TN)
    TP_Rate = TP / (TP + FN)
    FN_Rate = FN / (FN + TP)
    TN_Rate = TN / (TN + FP)
    CM_rate = np.array([[TP_Rate, FP_Rate], [FN_Rate, TN_Rate]])

    group_percentages = [TP_Rate, FP_Rate, FN_Rate, TN_Rate]
    group_vals = [TP, FP, FN, TN]
    group_names = ['True Pos', 'False Pos', 'False Neg', 'True Neg']
    labels = ['{} \n {} \n {}'.format(v1, format(v2, ".2f"), v3) for v1, v2, v3 in zip(group_names, group_percentages,
                                                                                       group_vals)]
    labels = np.asarray(labels).reshape(2, 2)

    sn.heatmap(CM_rate, annot=labels, cmap='Reds', vmin=0, vmax=1, fmt='')
    plt.title('{}-{} at Decision Threshold: {}'.format(start_date.strftime("%Y-%m-%d"),
                                                                        end_date.strftime("%Y-%m-%d"), threshold))
    plt.savefig('{}/CM/{}_confusion_matrix'.format(save_dir, save_date))
    plt.close()


def generate_tp_rate_map(df, nepal_mask, save_dir, decision_threshold=0.2):
    """
    Generate map of TP rate over monsoon season
    """
    district_list = df['district'].unique()
    nepal_arr = np.array(nepal_mask)
    FP_arr = copy.deepcopy(nepal_arr)
    nepal_arr[nepal_arr == 0] = np.nan
    FP_arr[FP_arr ==0] = np.nan

    tp_df = pd.DataFrame()
    tp_dist = []
    tp_count = []
    landslide_count = []

    for district in district_list:
        print('running for district: {}'.format(district))
        # subset district
        df_dist = df[df['district'] == district]
        labels = df_dist['groundtruth']
        preds = df_dist['model soft predictions']
        threshold_preds = (preds >= decision_threshold) * 1
        CM = confusion_matrix(labels, threshold_preds)
        TN = CM[0][0]
        try:
            FN = CM[1][0]
        except IndexError as e:
            FN = 0
        try:
            TP = CM[1][1]
        except IndexError as e:
            TP = 0
        try:
            FP = CM[0][1]
        except IndexError as e:
            FP = 0

        FP_Rate = FP / (FP + TN)
        if (TP+FN) == 0:
            TP_Rate = 0
        else:
            TP_Rate = TP / (TP + FN)

        tp_dist.append(district)
        tp_count.append(TP)
        landslide_count.append(sum(labels))

        district_val = district_dict[district]
        nepal_arr[nepal_arr == district_val] = TP_Rate
        FP_arr[FP_arr == district_val] = FP_Rate

    tp_df['District'] = tp_dist
    tp_df['TP Count over Monsoon'] = tp_count
    tp_df['Landslide Count over Monsoon Moving Window'] = landslide_count

    tp_df.to_csv('{}/tp_count_df.csv'.format(save_dir))
    plt.imshow(nepal_arr, cmap='gist_heat')
    cbar = plt.colorbar()
    plt.clim(0, 1)
    cbar.set_label('ML Classifier True Positive Rate', labelpad=15)
    plt.tight_layout()
    plt.savefig('{}/tp_rate_dist.png'.format(save_dir))
    plt.close()

    # Plot FP Rate
    plt.imshow(FP_arr, cmap='gist_heat')
    cbar = plt.colorbar()
    plt.clim(0,1)
    cbar.set_label('ML Classifier False Positive Rate', labelpad=15)
    plt.tight_layout()
    plt.savefig('{}/fp_rate_dist.png'.format(save_dir))
    plt.close()


def generate_monsoon_plots(results_df, year):
    """
    Generate Monsoon Season CM plots
    """
    # Get list of unique dates in dataframe
    date_list = results_df['date'].unique()
    sorted_dates = sorted(date_list)
    apr = sorted_dates.index("{}-04-01".format(year))
    may = sorted_dates.index("{}-05-01".format(year))
    june = sorted_dates.index("{}-06-01".format(year))
    july = sorted_dates.index("{}-07-01".format(year))
    aug = sorted_dates.index("{}-08-01".format(year))
    sept = sorted_dates.index("{}-09-01".format(year))
    # Iterate through date list and make confusion matrix plot
    total_tp = []
    total_fp = []
    total_fn = []
    total_tn = []

    for date in sorted_dates:
        # subset date
        df_subset = results_df[results_df['date'] == date]
        start = datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)
        end = start + timedelta(days=14)
        cm = generate_confusion_matrix(df_subset, 0.2)
        total_tp.append(cm[1][1])
        total_fp.append(cm[0][1])
        total_fn.append(cm[1][0])
        total_tn.append(cm[0][0])
        generate_plot(cm, start, end, 0.2, results_dir, date)

    fig, ax = plt.subplots()
    plt.plot(sorted_dates, total_tp, label='True Positive')
    plt.plot(sorted_dates, total_fp, label='False Positive')
    plt.plot(sorted_dates, total_fn, label='False Negative')
    ax.set_xticks([sorted_dates[apr], sorted_dates[may], sorted_dates[june], sorted_dates[july], sorted_dates[aug],
                   sorted_dates[sept]])
    ax.tick_params(axis='x', labelsize=8)
    plt.xlabel('Date')
    plt.ylabel('Count')
    plt.legend()
    plt.title('True Positive, False Positive, and False Negative Count for UKMO Ensemble Member 1')
    plt.show()


if __name__ == '__main__':
    args = get_args()
    run_dir = args.run
    root_dir = args.root_dir
    year = args.test_year

    nepal_im = Image.open('{}/District_Labels.tif'.format(root_dir))

    if not run_dir:
        results_dir = '{}/FullSeason_Results'.format(root_dir)
        results_df = pd.read_csv('{}/predictions_and_groundtruth_trainsource_gpm.csv'.format(results_dir))
    else:
        results_dir = '{}/Results/{}'.format(root_dir, run_dir)
        results_df = pd.read_csv('{}/predictions_and_groundtruth.csv'.format(results_dir))


    if not os.path.exists('{}/CM'.format(results_dir)):
        os.mkdir('{}/CM'.format(results_dir))

    # Get overall TP rate per Dist
    generate_tp_rate_map(results_df, nepal_im, results_dir)
    generate_monsoon_plots(results_df, year)