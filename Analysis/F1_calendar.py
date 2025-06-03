"""
From the test set predictions, generate a calendar
of F1 scores for Nepal from the test set
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from datetime import datetime, timedelta
import os
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='Generating Confusion Matrix')
    parser.add_argument('--run', default=None, help='Wandb run name')
    parser.add_argument('--root_dir', help='Root Directory of data')
    parser.add_argument('--test_year', help='Year of test')
    return parser.parse_args()


def f1_score_gen(df, decision_threshold):
    """
    Get f1_score based on decision threshold
    :param: df: dataframe of predictions and labels
    :param: decision_threshold: user-specified decision threshold for classification
    """
    labels = df['groundtruth']
    preds = df['model soft predictions']
    threshold_preds = (preds >= decision_threshold)*1

    f1 = f1_score(labels, threshold_preds, zero_division=0)
    return f1


def generate_f1_fig(df, year, save_dir):
    """
    Generate f1 score figure
    """
    # Get list of unique dates in dataframe
    date_list = df['date'].unique()
    sorted_dates = sorted(date_list)

    f1_all = f1_score_gen(df, 0.2)
    if not os.path.exists('{}/F1'.format(save_dir)):
        os.mkdir('{}/F1'.format(save_dir))

    # Iterate through date list and make nepal array
    f1_scores = []
    start_dates = []
    end_dates = []

    #apr = sorted_dates.index("{}-04-01".format(year))
    may = sorted_dates.index("{}-05-01".format(year))
    june = sorted_dates.index("{}-06-01".format(year))
    july = sorted_dates.index("{}-07-01".format(year))
    aug = sorted_dates.index("{}-08-01".format(year))
    sept = sorted_dates.index("{}-09-01".format(year))

    for date in sorted_dates:
        # subset date
        df_subset = df[df['date'] == date]
        start = datetime.strptime(date, '%Y-%m-%d') + timedelta(days=1)
        start_dates.append(start)
        end = start + timedelta(days=14)
        end_dates.append(end)
        f1 = f1_score_gen(df_subset, 0.2)
        f1_scores.append(f1)

    df = pd.DataFrame()
    df['f1'] = f1_scores
    df['doy'] = sorted_dates
    df.to_csv('{}/f1_timeseries.csv'.format(save_dir))

    '''

    for i in range(len(f1_scores)):
        fig, ax = plt.subplots()
        plt.plot(sorted_dates, f1_scores)
        plt.scatter(sorted_dates[i], f1_scores[i],  marker='o', color='red')
        ax.set_xticks([sorted_dates[may], sorted_dates[june], sorted_dates[july], sorted_dates[aug],
                       sorted_dates[sept]])
        ax.tick_params(axis='x', labelsize=8)
        plt.title('{}-{}'.format(start_dates[i].strftime("%Y-%m-%d"), end_dates[i].strftime("%Y-%m-%d")))
        plt.xlabel('Date')
        plt.ylabel('F1 Score')
        plt.savefig('{}/F1/{}_F1.png'.format(save_dir, sorted_dates[i]))
        plt.close()
    '''


if __name__ == '__main__':
    args = get_args()
    run_dir = args.run
    root_dir = args.root_dir
    year = args.test_year

    if not run_dir:
        results = '{}/FullSeason_Results'.format(root_dir)
        prediction_df = pd.read_csv('{}/predictions_and_groundtruth_trainsource_gpm.csv'.format(results))
    else:
        results = '{}/{}'.format(root_dir, run_dir)
        prediction_df = pd.read_csv('{}/predictions_and_groundtruth.csv'.format(results))

    generate_f1_fig(prediction_df, year, results)



