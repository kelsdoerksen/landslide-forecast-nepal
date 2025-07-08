"""
Aggregating RF runs and generating feature importance plots/csvs for Nature Comms
"""

import pandas as pd
import matplotlib.pyplot as plt
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Feature Importance Plotting')
    parser.add_argument('--year', help='Year of analysis')

    return parser.parse_args()

# This is hard-coded list for nature
rf_list_2023 = ["restful-yogurt-1841", "fearless-frog-1840", "glorious-hill-1839", "hardy-mountain-1838",
                "feasible-lion-1837"]

rf_list_2024 = ["hearty-resonance-1836", "effortless-disco-1835", "charmed-energy-1834", "kind-sponge-1833",
                "usual-snowball-1832"]

root_dir = '/Volumes/PRO-G40/landslides/Nepal_Landslides_Forecasting_Project/Monsoon2024_Prep/Results/GPMv07'
save_dir = '/Users/kelseydoerksen/Desktop/Nature_Comms_Analysis_Plotting/RF_FI'


def agg_fi(year):
    if year == 2023:
        rf_list = rf_list_2023
    if year == 2024:
        rf_list = rf_list_2024

    dict_list = []
    for r in rf_list:
        df = pd.read_csv('{}/{}_ForecastModel_UKMO_EnsembleNum0/FI.csv'.format(root_dir, r))
        fi_dict = df.loc[0].to_dict()
        fi_dict.pop('Unnamed: 0')
        dict_list.append(fi_dict)

    keys = fi_dict.keys()
    avg_dict = {k: sum((d[k] for d in dict_list)) / len(dict_list) for k in fi_dict.keys()}
    return avg_dict

def agg_pi(year):
    if year == 2023:
        rf_list = rf_list_2023
    if year == 2024:
        rf_list = rf_list_2024

    dict_list = []
    stderr_list = []
    for r in rf_list:
        df = pd.read_csv('{}/{}_ForecastModel_UKMO_EnsembleNum0/PermI.csv'.format(root_dir, r))
        pi_dict = df.loc[0].to_dict()
        stderr_dict = df.loc[1].to_dict()
        pi_dict.pop('Unnamed: 0')
        stderr_dict.pop('Unnamed: 0')
        dict_list.append(pi_dict)
        stderr_list.append(stderr_dict)

    keys = pi_dict.keys()
    avg_dict = {k: sum((d[k] for d in dict_list)) / len(dict_list) for k in pi_dict.keys()}
    avg_std = {k: sum((d[k] for d in stderr_list)) / len(stderr_list) for k in stderr_dict.keys()}
    return avg_dict, avg_std


if __name__ == '__main__':
    args = get_args()
    year = args.year

    # Run for year
    dict_avg = agg_fi(int(year))
    fi_avg_df = pd.DataFrame(columns=dict_avg.keys())
    fi_avg_df.loc[0] = dict_avg.values()
    fi_avg_df.to_csv('{}/FI_avg_scores_{}.csv'.format(save_dir, year))
    names = dict_avg.keys()
    colors = ['slateblue' if 'UKMO' in feat else 'mediumseagreen' if 'dem' in feat else 'mediumseagreen' if 'slope' in feat else 'mediumseagreen' if 'aspect' in feat else 'mediumseagreen' if 'lc' in feat else 'deepskyblue' for feat in names]

    # Bar Plotting
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.bar(dict_avg.keys(), dict_avg.values(), align='edge', width=0.5, color=colors)
    plt.xticks(rotation=90)
    plt.ylabel('Feature Importance Score')
    plt.ylim(0,0.08)
    #plt.show()
    plt.close()

    # Pie Chart Plotting
    forecast_list = [v for k, v in dict_avg.items() if 'UKMO' in k]
    slope_list = [v for k, v in dict_avg.items() if 'slope' in k]
    aspect_list = [v for k, v in dict_avg.items() if 'aspect' in k]
    dem_list = [v for k, v in dict_avg.items() if 'dem' in k]
    lc_list = [v for k, v in dict_avg.items() if 'lc' in k]

    forecast_val = sum(forecast_list)
    geo_val = sum(slope_list) + sum(aspect_list) + sum(dem_list) + sum(lc_list)
    obs_val = sum(dict_avg.values()) - forecast_val - geo_val

    vals = [forecast_val, obs_val, geo_val]
    labels=['Precipitation Forecast', 'Precipitation Observation', 'Geomorphic']
    colors=['slateblue', 'deepskyblue', 'mediumseagreen']
    fig, ax = plt.subplots()
    ax.pie(vals, labels=labels, colors=colors, autopct='%1.1f%%', textprops={'color':"w"})
    #plt.show()
    plt.close()


    # Permutation Importance Plotting
    # Run for year
    dict_avg, std_avg = agg_pi(int(year))
    pi_avg_df = pd.DataFrame(columns=dict_avg.keys())
    pi_avg_df.loc[0] = dict_avg.values()
    pi_avg_df.to_csv('{}/PI_avg_scores_{}.csv'.format(save_dir, year))
    names = dict_avg.keys()

    df = pd.DataFrame()
    df['feature'] = dict_avg.keys()
    df['PI'] = dict_avg.values()
    df['std_err'] = std_avg.values()
    df = df.sort_values(by='PI', ascending=False)

    # Bar Plotting
    fig, ax = plt.subplots(figsize=(18, 10))
    ax.bar(df['feature'], df['PI'], align='edge', width=0.5, color=colors, yerr=df['std_err'])
    plt.xticks(rotation=90)
    plt.ylabel('Permutation Importance Score')
    #plt.show()
    plt.close()

