"""
Plotting PR curves from Wandb outputs
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/Volumes/PRO-G40 1/landslides/Nepal_Landslides_Forecasting_Project/Nepal_September_Workshop/ICLR_2025/'
                 'rf_varying_forecast.csv')

forecast_unique = df['name'].unique()

for f in forecast_unique:
    if 'KMA' in f:
        color = 'blue'
        linestyle = '-.'
    if 'UKMO' in f:
        color = 'green'
        linestyle = '--'
    if 'NCEP' in f:
        color = 'red'
        linestyle = '-'
    subset_df = df[df['name'] == f]
    precision = subset_df['precision']
    recall = subset_df['recall']
    plt.plot(recall, precision, linestyle=linestyle, color=color, label=f)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title('Precision-Recall Curve for Varying Precipitation Forecast Model using Random Forest')

plt.axhline(y=0.225, color='black', linestyle='dotted', label='Baseline')
plt.legend(fontsize="x-small")
plt.show()