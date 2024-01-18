Name :- Hari singh r
batch id :- DSWDMCOD 25082022 B

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing # 
# from datetime import datetime

cocacola = pd.read_excel("D:/assignments of data science/22 forecasting 3/CocaCola_Sales_Rawdata.xlsx")

cocacola.Sales.plot() # time series plot 

# Centering moving average for the time series
cocacola.Sales.plot(label = "org")
for i in range(2, 9, 2):
    cocacola["Sales"].rolling(i).mean().plot(label = str(i))
plt.legend(loc = 3)
    
# Time series decomposition plot 
decompose_ts_add = seasonal_decompose(cocacola.Sales, model = "additive", period = 4)
decompose_ts_add.plot()
decompose_ts_mul = seasonal_decompose(cocacola.Sales, model = "multiplicative", period = 4)
decompose_ts_mul.plot()

# ACF plot on Original data sets 
import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(cocacola.Sales, lags = 4)
# tsa_plots.plot_pacf(cocacola.Sales, lags=4)

# splitting the data into Train and Test data
# Recent 4 time period values are Test data

Train = cocacola.head(38)
Test = cocacola.tail(4)

# to change the index value in pandas data frame 
# Test.set_index(np.arange(1,4),inplace=True)

# Creating a function to calculate the MAPE value for test data 
def MAPE(pred,org):
    temp = np.abs((pred-org)/org)*100
    return np.mean(temp)


# Simple Exponential Method
ses_model = SimpleExpSmoothing(Train["Sales"]).fit()
pred_ses = ses_model.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_ses, Test.Sales) 

# Holt method 
hw_model = Holt(Train["Sales"]).fit()
pred_hw = hw_model.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_hw, Test.Sales) 

# Holts winter exponential smoothing with additive seasonality and additive trend
hwe_model_add_add = ExponentialSmoothing(Train["Sales"], seasonal = "add", trend = "add", seasonal_periods = 4).fit()
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_hwe_add_add, Test.Sales) 

# Holts winter exponential smoothing with multiplicative seasonality and additive trend
hwe_model_mul_add = ExponentialSmoothing(Train["Sales"], seasonal = "mul", trend = "add", seasonal_periods = 4).fit()
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0], end = Test.index[-1])
MAPE(pred_hwe_mul_add, Test.Sales) 


# Final Model on 100% Data
hwe_model_add_add = ExponentialSmoothing(cocacola["Sales"], seasonal = "add", trend = "add", seasonal_periods = 4).fit()

# Load the new data which includes the entry for future 4 values
new_data = pd.read_excel("E:/DATA SCIENCE ASSIGNMENT/Class And Assignment Dataset/Asss/Forecasting-Time Series/new_cocacola_data.xlsx")

newdata_pred = hwe_model_add_add.predict(start = new_data.index[0], end = new_data.index[-1])
newdata_pred
