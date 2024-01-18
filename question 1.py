Name :- Hari singh r
batch id :- DSWDMCOD 25082022 B

import numpy as np
import pandas as pd
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 10,6

dataset = pd.read_excel("D:/assignments of data science/22 forecasting 3/Airlines Data.xlsx")
dataset.columns
#Parse strings to datetime type
dataset['Month'] = pd.to_datetime(dataset['Month'],infer_datetime_format=True)
indexedDataset = dataset.set_index(['Month'])

from datetime import datetime
indexedDataset.head(5)

#plot graph
plt.xlabel("Date")
plt.ylabel("Number of air passengers")
plt.plot(indexedDataset)

# 2 methods used To check the stationarity of the data Rolling Statistics and Dickey-Fuller
#Determining Rolling Statistics
rolmean = indexedDataset.rolling(window=12).mean()

rolstd = indexedDataset.rolling(window=12).std()
print(rolmean,rolstd)

#plot Rolling Statistics:
orig = plt.plot(indexedDataset,color = 'blue',label = 'Original')
mean = plt.plot(rolmean,color = 'red',label = 'Rolling Mean')   
std = plt.plot(rolstd,color = 'black',label = 'Rolling Std')   
plt.legend(loc= 'best')
plt.title('Rolling Mean & Standard Deviation')

#Perform Dickey-Fuller test:
from statsmodels.tsa.stattools import adfuller    

print("Result of Dickey-Fuller test:")
dftest = adfuller(indexedDataset['Passengers'], autolag='AIC')

dfoutput = pd.Series(dftest[0:4], index=["Test Statitics","p-value","#Lags Used","Number of Observations Used"])
for key,value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print(dfoutput)    

#Estimated trend
indexedDataset_logscale = np.log(indexedDataset)
plt.plot(indexedDataset_logscale)

movingAverage = indexedDataset_logscale.rolling(window = 12).mean()
movingSTD= indexedDataset_logscale.rolling(window = 12).std()
plt.plot(indexedDataset_logscale)
plt.plot(movingAverage,color = "red")

datasetLogScaleMinusMovingAverage = indexedDataset_logscale - movingAverage
datasetLogScaleMinusMovingAverage.head(12)

#Remove Nan values
datasetLogScaleMinusMovingAverage.dropna(inplace = True)
datasetLogScaleMinusMovingAverage.head(10)

from statsmodels.tsa.stattools import adfuller   
def test_stationarity(timeseries):
    
    #determining rolling statistics
    movingAverage = timeseries.rolling(window = 12).mean()
    movingSTD= timeseries.rolling(window = 12).std()
    
    #plot rolling statistics:
    orig = plt.plot(timeseries,color = 'blue',label = 'Original')
    mean = plt.plot(movingAverage,color = 'red',label = 'Rolling Mean')   
    std = plt.plot(movingSTD,color = 'black',label = 'Rolling Std')   
    plt.legend(loc = 'best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
        
    #Perform Dickey-Fuller test:
    print("Result of Dickey-Fuller test:")
    dftest = adfuller(timeseries['Passengers'], autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=["Test Statitics","p-value","#Lags Used","Number of Observations Used"])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)    

test_stationarity(datasetLogScaleMinusMovingAverage)   

exponentialDecayWeightedAverage = indexedDataset_logscale.ewm(halflife=12, min_periods = 0,adjust = True).mean() 
plt.plot(indexedDataset_logscale)
plt.plot(exponentialDecayWeightedAverage, color ="red")

datasetLogScaleMinusexponentialDecayWeightedAverage = indexedDataset_logscale - exponentialDecayWeightedAverage 
test_stationarity(datasetLogScaleMinusexponentialDecayWeightedAverage)

datasetLogDiffShifting = indexedDataset_logscale - indexedDataset_logscale.shift() 
plt.plot(datasetLogDiffShifting)

datasetLogDiffShifting.dropna(inplace = True)
test_stationarity(datasetLogDiffShifting)

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition = seasonal_decompose(indexedDataset_logscale)

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(indexedDataset_logscale, label = "Original")
plt.legend(loc="best")
plt.subplot(412)
plt.plot(trend,label = "Trend")
plt.legend(loc="best")
plt.subplot(413)
plt.plot(seasonal,label="Seasonality")
plt.legend(loc="best")
plt.subplot(414)
plt.plot(residual,label="Residuals")
plt.legend(loc="best")
plt.tight_layout()

#decomposedLogData = residual
#decomposedLogData.dropna(inplace=True)
#test_stationarity(decomposedLogData)

#ACFand PACF plots: In order to find q & p value
from statsmodels.tsa.stattools import acf,pacf

lag_acf = acf(datasetLogDiffShifting, nlags = 20)
lag_pacf = pacf(datasetLogDiffShifting, nlags = 20, method = 'ols')

#Plot ACF:
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color='gray')
plt.title("Autocorrelation Function")

#plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color='gray')
plt.axhline(y=-1.96/np.sqrt(len(datasetLogDiffShifting)),linestyle='--',color='gray')
plt.title("Partial Autocorrelation Function")
plt.tight_layout()

from statsmodels.tsa.arima_model import ARIMA

#AR MODEL
model = ARIMA(indexedDataset_logscale, order=(2,1,2))
results_AR = model.fit(disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_AR.fittedvalues, color = "red")
plt.title("RSS: %.4f"% sum(results_AR.fittedvalues-datasetLogDiffShifting['Passengers'])**2)
print("Plotting AR model")

#MA MODEL
model = ARIMA(indexedDataset_logscale, order=(2,1,0))
results_MA = model.fit(disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_MA.fittedvalues, color = "red")
plt.title("RSS: %.4f"% sum(results_MA.fittedvalues-datasetLogDiffShifting['Passengers'])**2)
print("Plotting AR model")

#Combining AR and MA model
#MA MODEL
model = ARIMA(indexedDataset_logscale, order=(2,1,2))
results_ARIMA = model.fit(disp=-1)
plt.plot(datasetLogDiffShifting)
plt.plot(results_ARIMA.fittedvalues, color = "red")
plt.title("RSS: %.4f"% sum(results_ARIMA.fittedvalues-datasetLogDiffShifting['Passengers'])**2)
print("Plotting AR model")

prediction_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy = True)
print(prediction_ARIMA_diff.head())

#convert to cumulative sum
prediction_ARIMA_diff_cumsum = prediction_ARIMA_diff.cumsum()
print(prediction_ARIMA_diff_cumsum.head())

prediction_ARIMA_log = pd.Series(indexedDataset_logscale["Passengers"].iloc[0], index=indexedDataset_logscale.index)
prediction_ARIMA_log = prediction_ARIMA_log.add(prediction_ARIMA_diff_cumsum,fill_value=0)
prediction_ARIMA_log.head()

prediction_ARIMA = np.exp(prediction_ARIMA_log)
plt.plot(indexedDataset)
plt.plot(prediction_ARIMA)

indexedDataset_logscale

results_ARIMA.plot_predict(1,156)
results_ARIMA.forecast(steps=60)