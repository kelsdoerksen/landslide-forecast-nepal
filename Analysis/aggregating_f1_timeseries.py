"""
Script to read in and aggregate f1 csvs so I don't have to do things by hand
"""

import pandas as pd
import matplotlib.pyplot as plt

# This is hard-coded list for nature
# rf_list_2023 = ["laced-yogurt-1740", "worldly-monkey-1739", "easy-planet-1738", "glad-pyramid-1737",
# "azure-vortex-1736"]
# gb_list_2023 = ["proud-firefly-1726", "lively-sound-1725", "ethereal-terrain-1724", "smooth-dew-1723",
# "curious-brook-1722"]
# xgb_list_2023 = ["glorious-surf-1735", "breezy-puddle-1734", "amber-frog-1733", "confused-rain-1732",
# "amber-capybara-1731"]
# rf_list_2024 = ["usual-armadillo-1721", "unique-cloud-1720", "glowing-vortex-1719", "peach-fog-1718",
# "earthy-meadow-1717"]
# gb_list_2024 = ["major-universe-1716", "hopeful-blaze-1714", "gentle-lake-1713", "ethereal-morning-1712",
# "laced-dragon-1711"]
# xgb_list_2024 = ["chocolate-frog-1710", "northern-feather-1709", "glowing-resonance-1708", "grateful-lion-1707",
# "fine-durian-1706"]

# 2025/2026 retrain 5-seed UKMO RF runs (GPMv07 hindcast). Only RF was rerun.
rf_list_2023 = ["s5vmtkwd", "ybrwrcuo", "g1c5gncb", "1pq76brf", "0ywrra8l"]
rf_list_2024 = ["bk4ocih2", "j5o9v9zr", "7frh12gd", "sdwk3jax", "kfl3hvka"]

# F1 timeseries files are written at threshold 0.24 by Analysis/analyze_run.py
THRESHOLD = 0.24

root_dir = PROJECT_DIR + '/Monsoon2024_Prep/Results/GPMv07'
# save_dir = '$LANDSLIDE_PROJECT_DIR/Nature_Comms'
save_dir = PROJECT_DIR + '/Monsoon2024_Prep/Results/GPMv07/aggregate_5seed_retrain'
import os

PROJECT_DIR = os.environ.get('LANDSLIDE_PROJECT_DIR', 'data')  # see README > Data
os.makedirs(save_dir, exist_ok=True)

def f1_avg(run_list, year):
    """
    F1 timeseries avg over runs (uses the 14-day rolling F1 column 'f1').
    """
    df_list = []
    for r in run_list:
        df = pd.read_csv('{}/{}_ForecastModel_UKMO_EnsembleNum0/f1_timeseries_{}_thr{}.csv'.format(
            root_dir, r, year, THRESHOLD))
        df_list.append(df[['doy', 'f1', 'landslide_count']])

    df_avg = pd.DataFrame()
    df_avg['doy'] = df_list[0]['doy']
    df_avg['f1'] = sum(d['f1'].values for d in df_list) / len(df_list)
    df_avg['landslide_count'] = df_list[0]['landslide_count'].values
    return df_avg

rf_2023 = f1_avg(rf_list_2023, '2023')
rf_2023.to_csv('{}/rf_2023.csv'.format(save_dir), index=False)

rf_2024 = f1_avg(rf_list_2024, '2024')
rf_2024.to_csv('{}/rf_2024.csv'.format(save_dir), index=False)

# GB / XGB were not rerun for the 2025/2026 retrain; only RF is aggregated here.
# gb_2023 = f1_avg(gb_list_2023, '2023');  gb_2023.to_csv('{}/gb_2023.csv'.format(save_dir))
# xgb_2023 = f1_avg(xgb_list_2023, '2023'); xgb_2023.to_csv('{}/xgb_2023.csv'.format(save_dir))
# gb_2024 = f1_avg(gb_list_2024, '2024');  gb_2024.to_csv('{}/gb_2024.csv'.format(save_dir))
# xgb_2024 = f1_avg(xgb_list_2024, '2024'); xgb_2024.to_csv('{}/xgb_2024.csv'.format(save_dir))


def plot_rf(rf, year):
    """
    Plot the 5-seed average RF 14-day rolling F1 timeseries with landslide-count bars.
    """
    dates = rf['doy'].tolist()

    def idx(m):
        tag = '{}-{}-01'.format(year, m)
        return dates.index(tag) if tag in dates else None
    ticks = [i for i in (idx('04'), idx('05'), idx('06'), idx('07'), idx('08'), idx('09'), idx('10'))
             if i is not None]

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    ax2.bar(range(len(rf)), rf['landslide_count'], color='gray', alpha=0.3, width=1.0,
            label='Landsliding districts')
    ax1.plot(range(len(rf)), rf['f1'], color='slateblue', lw=2.0, label='RF 14-day rolling F1 (5-seed avg)')
    ax1.set_ylim(0, 1)
    ax1.set_ylabel('F1 Score')
    ax2.set_ylabel('Landsliding district count')
    ax1.set_xticks(ticks)
    ax1.set_xticklabels([dates[i][:7] for i in ticks])
    ax1.set_xlabel('Date')
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)
    lines, labels = ax1.get_legend_handles_labels()
    bars, blabels = ax2.get_legend_handles_labels()
    ax1.legend(lines + bars, labels + blabels, loc='upper left')
    plt.title('{} RF 14-day rolling F1 (5-seed avg, threshold {})'.format(year, THRESHOLD))
    plt.tight_layout()
    plt.savefig('{}/f1_timeseries_{}_5seed.png'.format(save_dir, year), dpi=200)
    plt.close()


plot_rf(rf_2023, 2023)
plot_rf(rf_2024, 2024)
print('Wrote 5-seed F1 timeseries plots to {}'.format(save_dir))

