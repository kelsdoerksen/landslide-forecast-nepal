"""
From the test set predictions, generate a calendar
of F1 scores for Nepal from the test set
for the various prediction thresholds
"""

import pandas as pd
import matplotlib.pyplot as plt
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Generating Confusion Matrix')
    parser.add_argument('--run_dir', help='Wandb run name')
    parser.add_argument('--threshold', help='Prediction threshold')
    return parser.parse_args()


def generate_f1_fig(date_list, f1, save_dir):
    """
    Generate f1 score figure
    """
    apr = date_list.index("2023-04-01")
    may = date_list.index("2023-05-01")
    june = date_list.index("2023-06-01")
    july = date_list.index("2023-07-01")
    aug = date_list.index("2023-08-01")
    sept = date_list.index("2023-09-01")

    for i in range(len(date_list)):
        fig, ax = plt.subplots()
        plt.plot(date_list, f1)
        plt.scatter(date_list[i], f1[i],  marker='o', color='red')
        ax.set_xticks([date_list[apr], date_list[may], date_list[june], date_list[july], date_list[aug],
                       date_list[sept], date_list[-1]])
        ax.tick_params(axis='x', labelsize=8)
        plt.title('{}-{}'.format(date_list[i], date_list[i]))
        plt.xlabel('Date')
        plt.ylabel('F1 Score')
        plt.savefig('{}/F1/{}_F1.png'.format(save_dir, date_list[i]))
        plt.close()


if __name__ == '__main__':
    args = get_args()
    run_dir = args.run_dir
    threshold = args.threshold

    pr_df = pd.read_csv('{}/precision_recall_2023_results_threshold_{}.csv'.format(run_dir, threshold))

    # Get list of unique dates in dataframe
    date_list = pr_df['date'].tolist()
    f1 = pr_df['F1'].tolist()

    generate_f1_fig(date_list, f1, run_dir)