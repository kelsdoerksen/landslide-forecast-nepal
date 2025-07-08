"""
Script to read in and aggregate f1 csvs so I don't have to do things by hand
"""

import pandas as pd
import matplotlib.pyplot as plt

# This is hard-coded list for nature
rf_list_2023 = ["laced-yogurt-1740", "worldly-monkey-1739", "easy-planet-1738", "glad-pyramid-1737",
"azure-vortex-1736"]

gb_list_2023 = ["proud-firefly-1726", "lively-sound-1725", "ethereal-terrain-1724", "smooth-dew-1723",
"curious-brook-1722"]

xgb_list_2023 = ["glorious-surf-1735", "breezy-puddle-1734", "amber-frog-1733", "confused-rain-1732",
"amber-capybara-1731"]

rf_list_2024 = ["usual-armadillo-1721", "unique-cloud-1720", "glowing-vortex-1719", "peach-fog-1718",
"earthy-meadow-1717"]

gb_list_2024 = ["major-universe-1716", "hopeful-blaze-1714", "gentle-lake-1713", "ethereal-morning-1712",
"laced-dragon-1711"]

xgb_list_2024 = ["chocolate-frog-1710", "northern-feather-1709", "glowing-resonance-1708", "grateful-lion-1707",
"fine-durian-1706"]

root_dir = '/Volumes/PRO-G40/landslides/Nepal_Landslides_Forecasting_Project/Monsoon2024_Prep/Results/GPMv07'
save_dir = '/Users/kelseydoerksen/Desktop/Nature_Comms_Analysis_Plotting/ML_Timeseries_F1'

def f1_avg(run_list, year):
    """
    F1 timeseries avg over runs
    """
    df_list = []
    for r in run_list:
        df = pd.read_csv('{}/{}_ForecastModel_UKMO_EnsembleNum0/f1_timeseries_{}_thr0.2.csv'.format(root_dir, r, year))
        df_list.append(df)

    df_avg = pd.DataFrame()
    df_avg['doy'] = df_list[0]['doy']
    df_avg['f1'] = (df_list[0]['f1'] + df_list[1]['f1'] + df_list[2]['f1'] + df_list[3]['f1'] + df_list[4][
        'f1']) / 5

    return df_avg

rf_2023 = f1_avg(rf_list_2023, '2023')
rf_2023.to_csv('{}/rf_2023.csv'.format(save_dir))

gb_2023 = f1_avg(gb_list_2023, '2023')
gb_2023.to_csv('{}/gb_2023.csv'.format(save_dir))

xgb_2023 = f1_avg(xgb_list_2023, '2023')
xgb_2023.to_csv('{}/xgb_2023.csv'.format(save_dir))

rf_2024 = f1_avg(rf_list_2024, '2024')
rf_2024.to_csv('{}/rf_2024.csv'.format(save_dir))

gb_2024 = f1_avg(gb_list_2024, '2024')
gb_2024.to_csv('{}/gb_2024.csv'.format(save_dir))

xgb_2024 = f1_avg(xgb_list_2024, '2024')
xgb_2024.to_csv('{}/xgb_2024.csv'.format(save_dir))

# --- Plotting 2023
def plot_2023(rf, gb, xgb):
    """
    Plotting 2023 data
    """
    year = 2023
    sorted_dates = rf['doy'].tolist()
    apr = sorted_dates.index("{}-04-01".format(year))
    may = sorted_dates.index("{}-05-01".format(year))
    june = sorted_dates.index("{}-06-01".format(year))
    july = sorted_dates.index("{}-07-01".format(year))
    aug = sorted_dates.index("{}-08-01".format(year))
    sept = sorted_dates.index("{}-09-01".format(year))
    oct = sorted_dates.index("{}-10-01".format(year))

    fig, ax = plt.subplots(figsize=(7,5))
    plt.plot(rf['doy'], rf['f1'], color='lightblue')
    plt.plot(gb['doy'], gb['f1'], color='gold')
    plt.plot(xgb['doy'], xgb['f1'], color='coral')
    ax.set_xticks([sorted_dates[apr], sorted_dates[may], sorted_dates[june], sorted_dates[july], sorted_dates[aug],
                   sorted_dates[sept], sorted_dates[oct]])
    ax.tick_params(axis='x', labelsize=8)
    plt.xlabel('Date')
    plt.ylim(0, 1)
    plt.ylabel('F1 Score')
    plt.show()


def plot_2024(rf, gb, xgb):
    """
    Plotting 2024
    """
    # --- Plotting 2024
    # Include plotting since there are gaps in data for 2024
    sorted_dates = rf['doy'].tolist()
    year = 2024
    apr = sorted_dates.index("{}-04-01".format(year))
    may = sorted_dates.index("{}-05-01".format(year))
    june = sorted_dates.index("{}-06-01".format(year))
    sept = sorted_dates.index("{}-09-01".format(year))
    oct = sorted_dates.index("{}-10-31".format(year))

    fig, ax = plt.subplots(figsize=(7,5))
    plt.plot(rf.loc[0:64]['doy'], rf.loc[0:64]['f1'], color='lightblue')
    plt.plot(rf.loc[65:]['doy'], rf.loc[65:]['f1'], color='lightblue')

    plt.plot(gb.loc[0:64]['doy'], gb.loc[0:64]['f1'], color='gold')
    plt.plot(gb.loc[65:]['doy'], gb.loc[65:]['f1'], color='gold')

    plt.plot(xgb.loc[0:64]['doy'], xgb.loc[0:64]['f1'], color='coral')
    plt.plot(xgb.loc[65:]['doy'], xgb.loc[65:]['f1'], color='coral')
    ax.set_xticks([sorted_dates[apr], sorted_dates[may], sorted_dates[june],
                   sorted_dates[sept], sorted_dates[oct]])
    ax.tick_params(axis='x', labelsize=8)
    plt.xlabel('Date')
    plt.ylim(0, 1)
    plt.ylabel('F1 Score')
    plt.show()


plot_2023(rf_2023, gb_2023, xgb_2023)
plot_2024(rf_2024, gb_2024, xgb_2024)

