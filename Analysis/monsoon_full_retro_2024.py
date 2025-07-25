"""
Full retro of monsoon season performance using pre-trained model
"""

import pickle
import argparse

import ipdb
import pandas as pd
from sklearn.utils import shuffle

def get_args():
    parser = argparse.ArgumentParser(description='Running for 2024 Monsoon season')
    parser.add_argument('--source', help='Preciptation source for hindcast data. gpm or gsmap')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    source = args.source

    root_dir = '/Volumes/PRO-G40/landslides/Nepal_Landslides_Forecasting_Project/Monsoon2024_Prep/2024_Season_Retro'
    results_dir = '{}/Results'.format(root_dir)

    # Load pre-trained model
    if source == 'gpm':
        model = pickle.load(open('{}/rf_model_gpm.pkl'.format(root_dir), 'rb'))
        test_data = pd.read_csv('{}/LabelledData_GSMaP/ecmwf/2024_windowsize14_district.csv'.format(root_dir))
        test_data.columns = test_data.columns.str.replace('GSMaP', 'gpm')
    else:
        model = pickle.load(open('{}/rf_model_gsmap.pkl'.format(root_dir), 'rb'))
        test_data = pd.read_csv('{}/LabelledData_GSMaP/ecmwf/2024_windowsize14_district.csv'.format(root_dir))

    # Rename columns according to what our model was trained on
    test_data.columns = test_data.columns.str.replace('ecmwf_ens_0', 'UKMO_ens_1')
    test_data = test_data.drop(columns='Unnamed: 0')

    # Preserve date and location information for output
    info_cols = ['date', 'district']
    test_info = test_data[info_cols]

    df_test = shuffle(test_data)

    Xtest = df_test.drop(columns=['date', 'district', 'label'])
    ytest = df_test['label']

    print('Evaluating model...')
    probs = model.predict_proba(Xtest)

    # Re-indexing so we can put it all in a df
    test_info = test_info.reset_index().drop(columns=['index'])
    ytest = ytest.reset_index().drop(columns=['index'])

    accuracy = model.score(Xtest, ytest)
    print(f'The hard predictions were right {100 * accuracy:5.2f}% of the time')

    df_probs = pd.DataFrame()
    df_probs['model soft predictions'] = probs[:, 1]
    df_probs['groundtruth'] = ytest
    df_probs['date'] = test_info['date']
    df_probs['district'] = test_info['district']
    df_probs.to_csv('{}/predictions_and_groundtruth_trainsource{}_.csv'.format(results_dir, source))
