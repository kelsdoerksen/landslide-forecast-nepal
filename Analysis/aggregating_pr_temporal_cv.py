"""
Script to read in the PR data for the temporal CV runs and then plot for
Nature Comms paper
"""

import pandas as pd
import matplotlib.pyplot as plt

rf_2016 = ["jumping-sound-1778", "clear-totem-1777", "fragrant-brook-1776", "scarlet-pond-1775", "earnest-fire-1774"]
rf_2017 = ["stilted-cosmos-1783", "likely-voice-1782", "fresh-frost-1781", "denim-snowflake-1780", "spring-totem-1779"]
rf_2018 = ["lunar-eon-1788", "worthy-plant-1787", "astral-oath-1786", "comfy-wind-1785", "peach-pine-1784"]
rf_2019 = ["gentle-elevator-1793", "swept-armadillo-1792", "vague-capybara-1791", "glad-sponge-1790", "ethereal-sun-1789"]
rf_2020 = ["peach-aardvark-1798", "hearty-moon-1797", "autumn-leaf-1796", "pleasant-donkey-1795", "vibrant-snowflake-1794"]
rf_2021 = ["soft-gorge-1803", "silvery-flower-1802", "floral-flower-1801", "icy-rain-1800", "fresh-snow-1799"]
rf_2022 = ["fanciful-frog-1808", "light-jazz-1807", "deep-energy-1806", "cosmic-galaxy-1805", "dutiful-music-1804"]
rf_2023 = ["dark-durian-1813", "prime-oath-1812", "devout-surf-1811", "fanciful-salad-1810", "atomic-flower-1809",]
rf_2024 = ["lemon-sunset-1818", "quiet-meadow-1817", "radiant-aardvark-1816", "wild-pine-1815", "wise-surf-1814"]

root_dir = '/Volumes/PRO-G40/landslides/Nepal_Landslides_Forecasting_Project/Monsoon2024_Prep/Results/GPMv07'
save_dir = '/Users/kelseydoerksen/Desktop/Nature_Comms_Analysis_Plotting/ML_TemporalCV_PR'

def pr_avg(run_list, year):
    """
    PR avg over runs
    """
    df_list = []
    for r in run_list:
        df = pd.read_csv('{}/{}_ForecastModel_UKMO_EnsembleNum0/precision_recall.csv'.format(root_dir, r, year))
        df_list.append(df)

    df_avg = pd.DataFrame()
    df_avg['precision'] = (df_list[0]['precision'] + df_list[1]['precision'] + df_list[2]['precision'] +
                           df_list[3]['precision'] + df_list[4]['precision']) / 5
    df_avg['recall'] = (df_list[0]['recall'] + df_list[1]['recall'] + df_list[2]['recall'] + df_list[3]['recall'] +
                        df_list[4]['recall']) / 5

    return df_avg


rf_pr_2016 = pr_avg(rf_2016, '2016')
rf_pr_2016.to_csv('{}/rf_2016.csv'.format(save_dir))

rf_pr_2017 = pr_avg(rf_2017, '2017')
rf_pr_2017.to_csv('{}/rf_2017.csv'.format(save_dir))

rf_pr_2018 = pr_avg(rf_2018, '2018')
rf_pr_2018.to_csv('{}/rf_2018.csv'.format(save_dir))

rf_pr_2019 = pr_avg(rf_2019, '2019')
rf_pr_2019.to_csv('{}/rf_2019.csv'.format(save_dir))

rf_pr_2020 = pr_avg(rf_2020, '2020')
rf_pr_2020.to_csv('{}/rf_2020.csv'.format(save_dir))

rf_pr_2021 = pr_avg(rf_2021, '2021')
rf_pr_2021.to_csv('{}/rf_2021.csv'.format(save_dir))

rf_pr_2022 = pr_avg(rf_2022, '2022')
rf_pr_2022.to_csv('{}/rf_2022.csv'.format(save_dir))

rf_pr_2023 = pr_avg(rf_2023, '2023')
rf_pr_2023.to_csv('{}/rf_2023.csv'.format(save_dir))

rf_pr_2024 = pr_avg(rf_2024, '2024')
rf_pr_2024.to_csv('{}/rf_2024.csv'.format(save_dir))

plt.plot(rf_pr_2016.recall, rf_pr_2016.precision, label='2016')
plt.plot(rf_pr_2017.recall, rf_pr_2017.precision, label='2017')
plt.plot(rf_pr_2018.recall, rf_pr_2018.precision, label='2018')
plt.plot(rf_pr_2019.recall, rf_pr_2019.precision, label='2019')
plt.plot(rf_pr_2020.recall, rf_pr_2020.precision, label='2020')
plt.plot(rf_pr_2021.recall, rf_pr_2021.precision, label='2021')
plt.plot(rf_pr_2022.recall, rf_pr_2022.precision, label='2022')
plt.plot(rf_pr_2023.recall, rf_pr_2023.precision, label='2023')
plt.plot(rf_pr_2024.recall, rf_pr_2024.precision, label='2024')
plt.ylim(0, 1)
plt.xlim(0, 1)
plt.ylabel('Precision')
plt.xlabel('Recall')
plt.legend()
plt.tight_layout()
plt.show()
