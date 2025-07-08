"""
Aggregating RF runs and generating feature importance plots/csvs for Nature Comms
"""

import pandas as pd
import matplotlib.pyplot as plt
from setuptools.command.easy_install import is_python_script

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

# Run for 2023
dict_avg_2023 = agg_fi(2024)
names = dict_avg_2023.keys()
colors = ['slateblue' if 'UKMO' in feat else 'mediumseagreen' if 'dem' in feat else 'mediumseagreen' if 'slope' in feat else 'mediumseagreen' if 'aspect' in feat else 'mediumseagreen' if 'lc' in feat else 'deepskyblue' for feat in names]

# Bar Plotting
fig, ax = plt.subplots(figsize=(18, 10))
ax.bar(dict_avg_2023.keys(), dict_avg_2023.values(), align='edge', width=0.5, color=colors)
plt.xticks(rotation=90)
plt.ylabel('Feature Importance Score')
plt.ylim(0,0.08)
plt.show()

# Pie Chart Plotting



