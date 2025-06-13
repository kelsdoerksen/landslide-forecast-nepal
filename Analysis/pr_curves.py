"""
Plotting PR curves from Wandb outputs
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/kelseydoerksen/Desktop/Nature_Comms_Analysis_Plotting/Varying_Forecast/VaryingForecast_GB/GB_Forecast_Varying_All.csv')

# rf filepath from iclr: /Users/kelseydoerksen/Desktop/Nature_Comms_Analysis_Plotting/Varying_Forecast/VaryingForecast_RFrf_varying_forecast_iclr.csv
# gb filepath: /Users/kelseydoerksen/Desktop/Nature_Comms_Analysis_Plotting/Varying_Forecast/VaryingForecast_GB/GB_Forecast_Varying_All.csv'

forecast_unique = df['Name'].unique()

for f in forecast_unique:
    if 'KMA' in f:
        color = 'blue'
        linestyle = '-.'
    if 'UKMO' in f:
        color = 'green'
        linestyle = '--'
    if 'NCEP' in f:
        color = 'orange'
        linestyle = '-'
    subset_df = df[df['Name'] == f]
    precision = subset_df['Precision']
    recall = subset_df['Recall']
    plt.plot(recall, precision, linestyle=linestyle, color=color, label=f)
    plt.xlabel("Recall")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.ylabel("Precision")
    plt.title('Precision-Recall Curve for Varying Precipitation Forecast Model using Gradient Boosting')

#plt.axhline(y=0.225, color='black', linestyle='dotted', label='Baseline')
plt.legend(fontsize="x-small", loc="upper right", mode = "expand", ncol = 6)
plt.show()