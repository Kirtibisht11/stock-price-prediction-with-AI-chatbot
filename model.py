import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pmdarima import auto_arima

# Function to fetch stock data
def fetch_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        return data['Close']
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

# Function to create ARIMA model
def arima_model(data):
    try:
        model = auto_arima(data, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)
        return model
    except Exception as e:
        print(f"Error creating ARIMA model: {e}")
        return None

# Main function
if __name__ == "__main__":
    ticker = input("Enter stock ticker: ")
    start_date = input("Enter start date (YYYY-MM-DD): ")
    end_date = input("Enter end date (YYYY-MM-DD): ")
    prediction_days = int(input("Enter number of prediction days: "))

    data = fetch_data(ticker, start_date, end_date)
    if data is None:
        exit()

    arima_model_fit = arima_model(data)
    if arima_model_fit is None:
        exit()

    arima_forecast = arima_model_fit.predict(n_periods=prediction_days)

    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=prediction_days, freq='B')

    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data, label='Historical Prices', color='blue')
    plt.plot(future_dates, arima_forecast, label='ARIMA Forecast', color='orange')
    plt.title(f'{ticker} Stock Price Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()


