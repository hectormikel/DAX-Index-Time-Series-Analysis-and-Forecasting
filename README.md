# DAX Index Time Series Analysis and Forecasting

This repository contains a comprehensive analysis and forecasting of the DAX Index using time series models. The analysis includes data preprocessing, stationarity tests, ARIMA modeling, and volatility forecasting using ARCH and GARCH models.

## Table of Contents

- [Introduction](#introduction)
- [Data Preprocessing](#data-preprocessing)
- [Stationarity Analysis](#stationarity-analysis)
- [ARIMA Modeling](#arima-modeling)
- [Volatility Modeling](#volatility-modeling)
- [Forecasting](#forecasting)
- [Evaluation](#evaluation)
- [Conclusion](#conclusion)

## Introduction

The DAX Index is a stock market index representing 40 of the largest and most liquid companies on the German stock market. This project aims to analyze the historical prices of the DAX Index, check for stationarity, fit appropriate time series models, and forecast future prices and volatility.

## Data Preprocessing

The dataset contains historical DAX index prices. The following steps were taken to preprocess the data:

1. Load the data from a CSV file.
2. Select relevant columns and rename them.
3. Convert the date column to datetime format and set it as the index.
4. Sort the data by date.
5. Handle missing values by forward-filling.
6. Remove commas and convert the price column to numeric.

```python
import pandas as pd
import numpy as np

df = pd.read_csv('DAX Historical Results Price Data.csv')
df = df[['Date', 'Price']].rename(columns={'Price': 'close', 'Date': 'date'})
df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
df.set_index('date', inplace=True)
df = df.sort_index(ascending=True)
df['close'] = df['close'].str.replace(',', '').astype(float)
df = df.fillna(method='ffill')
```

## Stationarity Analysis

The Augmented Dickey-Fuller (ADF) test was used to check for stationarity in the time series. The results indicated that the time series is non-stationary, requiring differencing to make it stationary.

```python
from statsmodels.tsa.stattools import adfuller

result = adfuller(df['close'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))
```

## ARIMA Modeling

The ARIMA model was used to forecast future prices. The best model was selected through stepwise fitting, and the residuals were analyzed to ensure no significant autocorrelation.

```python
from pmdarima import auto_arima
import matplotlib.pyplot as plt

stepwise_fit = auto_arima(df['close'], trace=True, suppress_warnings=True)
stepwise_fit.summary()

predictions = stepwise_fit.predict_in_sample(start=1, end=len(df)-1)
plt.figure(figsize=(12, 8))
plt.plot(df.index, df['close'], label='Actual')
plt.plot(df.index[1:], predictions, label='Fitted', color='red', linestyle='-.')
plt.legend()
plt.title('DAX: Actual vs Fitted')
plt.show()
```

## Volatility Modeling

ARCH and GARCH models were applied to model and forecast volatility. The GARCH(1,1) model provided a better fit compared to the ARCH model.

```python
from arch import arch_model

arch_model_instance = arch_model(df['close'], vol='ARCH', p=1)
arch_fit = arch_model_instance.fit(disp='off')
print(arch_fit.summary())

garch_model = arch_model(df['close'], vol='Garch', p=1, q=1)
garch_fit = garch_model.fit(disp='off')
print(garch_fit.summary())
```

## Forecasting

The ARIMA model was used to predict future DAX index prices for 30 business days, and the GARCH model was used to forecast future volatility for the same period. The combined forecasts were visualized to show the expected price range with volatility as a confidence band.

```python
steps = 30
future_predictions = stepwise_fit.predict(n_periods=steps)
last_date = df.index[-1]
future_dates = pd.date_range(start=last_date, periods=steps+1, freq='B')[1:]
future_df = pd.DataFrame(future_predictions, index=future_dates, columns=['Predicted_Close'])

garch_forecast = garch_fit.forecast(horizon=steps)
forecasted_volatility = np.sqrt(garch_forecast.variance.values[-1, :])

combined_forecast = pd.DataFrame({
    'Predicted_Close': future_df['Predicted_Close'],
    'Forecasted_Volatility': forecasted_volatility
}, index=future_dates)

plt.figure(figsize=(12, 8))
plt.plot(combined_forecast.index, combined_forecast['Predicted_Close'], label='Predicted Close', color='blue')
plt.fill_between(combined_forecast.index,
                 combined_forecast['Predicted_Close'] - combined_forecast['Forecasted_Volatility'],
                 combined_forecast['Predicted_Close'] + combined_forecast['Forecasted_Volatility'],
                 color='gray', alpha=0.3, label='Forecasted Volatility')
plt.title('DAX: Combined Forecast (Mean and Variance)')
plt.legend()
plt.show()
```


## Evaluation

The forecasted values were evaluated using various metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), Mean Absolute Percentage Error (MAPE), and R-squared (R²).

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

evaluation_df = pd.read_excel('Difference calculations.xlsx')
evaluation_df = evaluation_df.set_index('Trading Date')
evaluation_df = evaluation_df[['Predicted Close [a]', 'Actual Close [b]']]

predicted = evaluation_df['Predicted Close [a]']
actual = evaluation_df['Actual Close [b]']

mae = mean_absolute_error(actual, predicted)
mse = mean_squared_error(actual, predicted)
rmse = mse ** 0.5
mape = (abs(actual - predicted) / actual).mean() * 100
r2 = r2_score(actual, predicted)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"R-squared (R²): {r2}")
```


## Conclusion

The analysis demonstrates that ARIMA and GARCH models are effective tools for forecasting prices and volatility in financial time series. The combined approach provides a comprehensive view of expected prices and associated risks. Further improvements could involve incorporating external factors or exploring advanced models.



## Key Findings

1. The DAX index prices exhibit non-stationarity and require differencing for modeling.

2. ARIMA(1,1,0) is effective for forecasting prices, but the residuals suggest room for improvement.

3. Volatility clustering is evident, and GARCH(1,1) effectively captures this behavior.

4. The combined forecast provides a comprehensive view of expected prices and associated risks.

