"""
Aggregating F1 score from runs and plotting with count of landslides
"""

import pandas as pd
import matplotlib.pyplot as plt
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Generating Confusion Matrix')
    parser.add_argument('--save_dir', help='Save directory')
    parser.add_argument('--test_year', help='Year of test')
    return parser.parse_args()



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
    ax1.plot(combined_df["doy"], combined_df["eng"], label="F1 eng", color="lightblue")
    ax1.plot(combined_df["doy"], combined_df["10-embed"], label="F1 10-embed", color="darkorange")
    ax1.plot(combined_df["doy"], combined_df["32-embed"], label="F1 32-embed", color="forestgreen")
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
    save_dir = args.save_dir
    year = args.test_year

    f1_df = pd.read_csv('{}/{}_summary.csv'.format(save_dir, year))
    landslide_counts = pd.read_csv('{}/landslide_counts_dataframe.csv'.format(save_dir))
    plot_best_f1(f1_df, landslide_counts, year, save_dir)



