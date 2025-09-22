"""
From the test set predictions, generate a calendar
of F1 scores for Nepal from the test set
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import datetime
import os
import argparse
import re
import numpy as np


def get_args():
    parser = argparse.ArgumentParser(description='Generating Confusion Matrix')
    parser.add_argument('--run', default=None, help='Wandb run name')
    parser.add_argument('--root_dir', help='Root Directory of data')
    parser.add_argument('--test_year', help='Year of test')
    return parser.parse_args()

def daterange(date1, date2):
  date_list = []
  for n in range(int ((date2 - date1).days)+1):
    dt = date1 + datetime.timedelta(n)
    date_list.append(dt.strftime("%Y-%m-%d"))
  return date_list


def landslide_count_per_day(landslide_df):
    """
    Get a count of landslides per day and use this in plotting
    Using the landslide records bipad_records_2016-2024_july172025download.csv
    """
    # subset the landslide to only keep one per day per district
    df_unique = landslide_df.drop_duplicates(subset=['Incident on', 'District'], keep="first")
    landslide_counts = df_unique.groupby('Incident on').size().reset_index(name='landslide_count')

    return landslide_counts


def f1_score_gen_binary(df, decision_threshold):
    """
    Get f1_score based on decision threshold
    :param: df: dataframe of predictions and labels
    :param: decision_threshold: user-specified decision threshold for classification
    """
    labels = df['groundtruth']
    preds = df['model soft predictions']
    threshold_preds = (preds >= decision_threshold)*1

    f1 = f1_score(labels, threshold_preds, average='binary', zero_division=np.nan)
    return f1

def f1_score_gen_macro(df, decision_threshold):
    """
    Get f1_score based on decision threshold
    :param: df: dataframe of predictions and labels
    :param: decision_threshold: user-specified decision threshold for classification
    """
    labels = df['groundtruth']
    preds = df['model soft predictions']
    threshold_preds = (preds >= decision_threshold)*1

    f1 = f1_score(labels, threshold_preds, average='macro', zero_division=np.nan)
    return f1


def generate_f1_fig(df, year, save_dir):
    """
    Generate f1 score figure
    df: dataframe of predictions and labels
    save_dir: directory to save f1 score figure
    landslide_counts: number of landslides per day from the bipad portal for plotting
    """
    # Get list of dates that we should have data for, that way we can have nans for missing data
    sdate = datetime.date(int(year), 4, 1)
    edate = datetime.date(int(year), 10, 31)

    date_list = daterange(sdate, edate)
    sorted_dates = sorted(date_list)

    thr = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    for t in thr:
        f1_all = f1_score_gen_binary(df, t)

        # Iterate through date list and make nepal array
        f1_scores_binary = []
        start_dates = []
        end_dates = []
        f1_scores_macro = []

        for date in sorted_dates:
            # subset date
            df_subset = df[df['date'] == date]
            if df_subset.empty:
                f1_scores_binary.append(np.nan)
                f1_scores_macro.append(np.nan)
            else:
                start = datetime.datetime.strptime(date, '%Y-%m-%d') + datetime.timedelta(days=1)
                start_dates.append(start)
                end = start + datetime.timedelta(days=14)
                end_dates.append(end)
                f1_binary = f1_score_gen_binary(df_subset, t)
                f1_scores_binary.append(f1_binary)
                f1_macro = f1_score_gen_macro(df_subset, t)
                f1_scores_macro.append(f1_macro)

        df_f1 = pd.DataFrame()
        df_f1['f1_binary'] = f1_scores_binary
        df_f1['f1_macro'] = f1_scores_macro
        df_f1['doy'] = sorted_dates
        df_f1.to_csv('{}/f1_binary_timeseries_{}_thr{}.csv'.format(save_dir, year, t))

    # do this for the best threshold
    with open("{}/model_testing_results.txt".format(save_dir), "r") as f:
        text = f.read()

    # find the number after threshold so we can get daily f1 from this
    match = re.search(r"threshold\s+([0-9.]+)", text)
    if match:
        best_threshold = float(match.group(1))

    f1_all = f1_score_gen_binary(df, best_threshold)

    # Iterate through date list and make nepal array
    f1_score_best_binary = []
    f1_score_best_macro = []
    start_dates = []
    end_dates = []

    for date in sorted_dates:
        # subset date
        df_subset = df[df['date'] == date]
        start = datetime.datetime.strptime(date, '%Y-%m-%d') + datetime.timedelta(days=1)
        start_dates.append(start)
        end = start + datetime.timedelta(days=14)
        end_dates.append(end)
        f1_binary = f1_score_gen_binary(df_subset, best_threshold)
        f1_score_best_binary.append(f1_binary)
        f1_macro = f1_score_gen_macro(df_subset, best_threshold)
        f1_score_best_macro.append(f1_macro)

    df_f1_best = pd.DataFrame()
    df_f1_best['f1_binary'] = f1_score_best_binary
    df_f1_best['f1_macro'] = f1_score_best_macro
    df_f1_best['doy'] = sorted_dates
    df_f1_best.to_csv('{}/f1_timeseries_{}_best_threshold.csv'.format(save_dir, year, t))


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
    return df_f1_best


def plot_best_f1(f1_df, landslide_counts, year, save_dir):
    """
    Plot the best f1 time series with a bar graph of the landslide counts behind
    """
    landslide_counts = landslide_counts.rename(columns={'Incident on': 'doy'})
    combined_df = pd.merge(f1_df, landslide_counts, on='doy', how='left')

    combined_df = combined_df.sort_values(by=['doy'], ascending=True)
    sorted_dates = combined_df['doy'].tolist()

    apr = sorted_dates.index("{}-04-01".format(year))
    may = sorted_dates.index("{}-05-01".format(year))
    june = sorted_dates.index("{}-06-01".format(year))
    july = sorted_dates.index("{}-07-01".format(year))
    aug = sorted_dates.index("{}-08-01".format(year))
    sept = sorted_dates.index("{}-09-01".format(year))
    oct = sorted_dates.index("{}-10-01".format(year))
    x_ticks = [sorted_dates[apr], sorted_dates[may], sorted_dates[june], sorted_dates[july], sorted_dates[aug],
               sorted_dates[sept], sorted_dates[oct]]


    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot the F1 scores (primary y-axis)
    ax1.plot(combined_df["doy"], combined_df["f1_binary"], label="F1 (binary)", color="lightblue")
    ax1.plot(combined_df["doy"], combined_df["f1_macro"], label="F1 (macro)", color="coral")
    ax1.set_xlabel("Day of Year")
    ax1.set_xticks(x_ticks)
    ax1.set_ylabel("F1 Score")
    ax1.set_ylim(0, 1)

    # Create a secondary y-axis for the counts of landslides
    ax2 = ax1.twinx()
    ax2.bar(combined_df["doy"], combined_df["landslide_count"],
            color="gray", alpha=0.3, width=1, label="Landslide count")
    ax2.set_ylabel("Landslide Count")

    # Add legends (combine both axes)
    lines, labels = ax1.get_legend_handles_labels()
    bars, bar_labels = ax2.get_legend_handles_labels()
    ax1.legend(lines + bars, labels + bar_labels, loc="upper left")


    plt.title("Daily F1 Scores for 14-day forecast with Landslide Counts")
    plt.savefig('{}/f1_timeseries_{}_best_threshold.png'.format(save_dir, year))


if __name__ == '__main__':
    args = get_args()
    run_dir = args.run
    root_dir = args.root_dir
    year = args.test_year

    # Hard coding because this is what I will always use
    landslide_records = pd.read_csv('/Volumes/PRO-G40/landslides/Nepal_Landslides_Forecasting_Project/'
                                    'Monsoon2024_Prep/bipad_records_2016-2024_july172025download.csv')
    landslide_counts = landslide_count_per_day(landslide_records)

    if not run_dir:
        results = '{}/FullSeason_Results'.format(root_dir)
        prediction_df = pd.read_csv('{}/predictions_and_groundtruth_trainsource_gpm.csv'.format(results))
    else:
        results = '{}/{}'.format(root_dir, run_dir)
        prediction_df = pd.read_csv('{}/predictions_and_groundtruth.csv'.format(results))

    f1_best = generate_f1_fig(prediction_df, year, results)

    plot_best_f1(f1_best, landslide_counts, year, results)



