import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Plotting for journal paper')
    parser.add_argument('--filepath', help='File path to run plotting for',
                        required=True)
    parser.add_argument('--plot', help='Plot type to run',
                        required=True)
    parser.add_argument('--test_year', help='Test year',
                        required=False,
                        default=None)

    return parser.parse_args()

def f1_timeseries(df, test_year):
    """
    Plot F1 over time
    """
    date_list = df['DOY'].tolist()

    # Get ticks of certain periods
    apr = date_list.index("{}-04-01".format(test_year))
    may = date_list.index("{}-05-01".format(test_year))
    june = date_list.index("{}-06-01".format(test_year))
    july = date_list.index("{}-07-01".format(test_year))
    aug = date_list.index("{}-08-01".format(test_year))
    sept = date_list.index("{}-09-01".format(test_year))
    oct = date_list.index("{}-10-01".format(test_year))
    end = date_list.index("{}-10-31".format(test_year))

    # Check if

    # Plotting
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(df['DOY'], df['Avg F1'], label='F1-Score', color='green')
    ax.fill_between(df['DOY'], df['Lower F1'], df['Upper F1'], color='green', alpha=0.3, label='Variance')

    ax.set_xlabel('Date')
    ax.set_ylabel('F1 Score')
    ax.set_title('UKMO-0 Train, ECMWF Test F1 Timeseries for {} Monsoon Season'.format(test_year))
    ax.set_xticks([date_list[apr], date_list[may], date_list[june], date_list[july], date_list[aug],
                   date_list[sept], date_list[sept], date_list[oct], date_list[-1]])
    ax.set_xticklabels([date_list[apr], date_list[may], date_list[june], date_list[july], date_list[aug],
                        date_list[sept], date_list[sept], date_list[oct], date_list[-1]], rotation=30, ha='right')
    ax.legend()
    ax.set_ylim(0.5, 1.0)
    plt.tight_layout()
    plt.show()
    plt.savefig('{}.png'.format(file[-4]), dpi=300, bbox_inches='tight')


def temporal_cv_pr(df):

    # First let's plot the precision and recall and how they vary with threshold per year
    years = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024]
    fig1, ax1 = plt.subplots(figsize=(7.5, 4))

    # Plotting Precision
    for y in years:
        df_filt = df[df['Year'] == y]
        upper = df_filt['Avg Precision'] + df_filt['Var Precision']
        lower = df_filt['Avg Precision'] - df_filt['Var Precision']
        ax1.plot(df_filt['Threshold'], df_filt['Avg Precision'], label='Precision Test Year {}'.format(y))
        ax1.fill_between(df_filt['Threshold'], lower, upper, alpha=0.3)

    ax1.legend(fontsize="x-small")
    ax1.set_xlabel('Decision Threshold')
    ax1.set_ylabel('Precision')
    ax1.set_title('Precision for varying decision threshold for held-out test year')
    plt.tight_layout()
    plt.show()

    # Plotting Recall
    fig2, ax2 = plt.subplots(figsize=(7.5, 4))

    for y in years:
        df_filt = df[df['Year'] == y]
        upper = df_filt['Avg Recall'] + df_filt['Var Recall']
        lower = df_filt['Avg Recall'] - df_filt['Var Recall']
        ax2.plot(df_filt['Threshold'], df_filt['Avg Recall'], label='Recall Test Year {}'.format(y))
        ax2.fill_between(df_filt['Threshold'], lower, upper, alpha=0.3)

    ax2.legend(fontsize="x-small")
    ax2.set_xlabel('Decision Threshold')
    ax2.set_ylabel('Recall')
    ax2.set_title('Recall for varying decision threshold for held-out test year')
    plt.tight_layout()
    plt.show()

def temporal_cv_fnr(df):

    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.errorbar(df['Year'], df['FNR'], yerr=df['FNR var'], fmt='o', color='blue', ecolor='lightblue', capsize=4,
                label='FNR')
    ax.set_ylabel('False Negative Rate', color='blue')
    ax.tick_params(axis='y')

    ax2 = ax.twinx()
    ax2.errorbar(df['Year'], df['FPR'], yerr=df['FPR var'], fmt='o', color='orange', ecolor='lightyellow', capsize=4,
                label='FPR')
    ax2.tick_params(axis='y')
    ax2.set_ylabel('False Positive Rate', color='orange')

    plt.title('False Negative Rate and False Positive Rate for held-out test year')
    plt.xlabel('Held-out Test Year')
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    args = get_args()
    file = args.filepath
    test_year = args.test_year
    plot_type = args.plot

    # Read in file
    df = pd.read_csv(file)

    # Run analysis based on specified
    if plot_type == 'f1_timeseries':
        f1_timeseries(df, test_year)

    if plot_type == 'temporal-cv-pr':
        temporal_cv_pr(df)

    if plot_type == 'temporal-cv-fnr':
        temporal_cv_fnr(df)














