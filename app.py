from flask import Flask, request, render_template
from model import fetch_data, arima_model
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('combine.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['POST'])
def predict():
    ticker = request.form['ticker']
    start_date = request.form['start_date']
    end_date = request.form['end_date']
    prediction_days = int(request.form['prediction_days'])

    # Retrieve prediction days
    data = fetch_data(ticker, start_date, end_date)

    # ARIMA Prediction
    arima_model_fit = arima_model(data)
    arima_forecast = arima_model_fit.predict(n_periods=prediction_days)

   # Prepare results for rendering
    results = {
        'ticker': ticker,
        'arima_forecast': arima_forecast.tolist(),
        'dates': pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=prediction_days, freq='B').tolist()
    }

    # Use zip to combine dates and forecast
    results['forecast'] = list(zip(results['dates'], results['arima_forecast']))

    # Generate graph
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, data, label='Historical Prices')
    plt.plot(results['dates'], results['arima_forecast'], label='ARIMA Forecast')
    plt.title(f'{ticker} Stock Price Forecast')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)

    # Convert graph to base64 encoded string
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()

    # Pass graph to HTML template
    return render_template('combine.html', results=results, graph_url=graph_url)




if __name__ == '__main__':
    app.run(debug=True)

