#%%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
# Load the data
data = pd.read_csv('Netherlands_day-ahead-2015-2023.csv')
data['Datetime (Local)'] = pd.to_datetime(data['Datetime (Local)'])

# Ensure the 'Datetime (Local)' is set as the DataFrame index
data.set_index('Datetime (Local)', inplace=True)

# Filter for the last year
end_date = data.index.max()
start_date = end_date - pd.DateOffset(years=1)
last_year_data = data[(data.index >= start_date) & (data.index <= end_date)]
Y = np.array(last_year_data['Price (EUR/MWhe)'])

#%%
stp = 24

dic = {}
for i in range(34,365):
    
    model = SARIMAX(Y, order=(1, 1, 1), seasonal_order=(1, 1, 1, 24)) # Assuming daily seasonality if hourly data
    model_fit = model.fit()
    next_prediction = model_fit.forecast(steps=stp)
    dic[i] = next_prediction
    # predictions = np.append(predictions, next_prediction)
    Y = np.append(Y, next_prediction)
    Y = Y[stp:]
    print(i,end=', ')

