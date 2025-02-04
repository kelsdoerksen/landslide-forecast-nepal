"""
Script to calculate the results
of the model for the 2024 monsoon season
"""

import pandas as pd
import os
import matplotlib.pyplot as plt

root_dir = '/Volumes/PRO-G40/landslides/Nepal_Landslides_Forecasting_Project/Monsoon2024_Prep/2024_Season_Retro/Predictions_vs_BiPad'
save_dir = '/Volumes/PRO-G40/landslides/Nepal_Landslides_Forecasting_Project/Monsoon2024_Prep/2024_Season_Retro/'


def generate_f1_tpr_fig(df, save_dir):
    """
    Generate f1, tpr score figure
    """
    '''
    for i in range(len(df)):
        fig, ax = plt.subplots()
        plt.plot(df['date'], df['F1'])
        plt.scatter(df['date'][i], df['F1'][i],  marker='o', color='red')
        plt.text(0.1, 0.69, 'TP: {}'.format(df['TP'][i]), size=10)
        plt.text(0.1, 0.65, 'FN: {}'.format(df['FN'][i]), size=10)
        plt.text(0.1, 0.61, 'FP: {}'.format(df['FP'][i]), size=10)
        ax.tick_params(axis='x', labelsize=6)
        plt.xticks(rotation=45)
        plt.title('{} Forecast Score'.format(df['date'][i]))
        plt.xlabel('Date')
        plt.ylabel('F1 Score')
        plt.savefig('{}/F1/{}_F1.png'.format(save_dir, df['date'][i]))
        plt.close()
    '''

    for i in range(len(df)):
        fig, ax = plt.subplots()
        plt.plot(df['date'], df['TPR'])
        plt.scatter(df['date'][i], df['TPR'][i],  marker='o', color='red')
        plt.text(0.1, 0.6, 'TP: {}'.format(df['TP'][i]), size=10)
        plt.text(0.1, 0.56, 'FN: {}'.format(df['FN'][i]), size=10)
        plt.text(0.1, 0.51, 'FP: {}'.format(df['FP'][i]), size=10)
        ax.tick_params(axis='x', labelsize=6)
        plt.xticks(rotation=45)
        plt.title('{} Forecast Score'.format(df['date'][i]))
        plt.xlabel('Date')
        plt.ylabel('TPR')
        plt.savefig('{}/TPR/{}_TPR.png'.format(save_dir, df['date'][i]))
        plt.close()


def get_metrics(df):
    """
    Get metics from df of results
    """
    results_dict = {}

    predictions = df['Districts Predicted']
    predictions = [x for x in predictions if str(x) != 'nan']
    groundtruth = df['Districts with Landslide']
    groundtruth = [x for x in groundtruth if str(x) != 'nan']

    TP = len(set(predictions) & set(groundtruth))
    FP = len([x for x in predictions if x not in groundtruth])
    FN = len([x for x in groundtruth if x not in predictions])
    TPR = TP / (TP + FN)
    F1 = TP / (TP + (0.5*(FP+FN)))

    results_dict['TP'] = TP
    results_dict['FP'] = FP
    results_dict['FN'] = FN
    results_dict['TPR'] = TPR
    results_dict['F1'] = F1

    return results_dict

# Load in file names
fns = os.listdir(root_dir)
# Get list of dates
dates = [ele.replace('Prediction_vs_BiPad_', '') for ele in fns]
dates = [ele.replace('.csv', '') for ele in dates]
dates.sort()

results_list = []
for d in range(len(dates)):
    data = pd.read_csv('{}/{}'.format(root_dir, fns[d]))
    results = get_metrics(data)
    results['date'] = dates[d]
    results_list.append(results)

results_df = pd.DataFrame(results_list)
generate_f1_tpr_fig(results_df, save_dir)
results_df.to_csv('{}/monsoon_2024_results.csv'.format(root_dir))
