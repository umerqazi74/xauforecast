from flask import Flask, jsonify
import yfinance as yf
import pandas as pd

app = Flask(__name__)

# Function to fetch daily XAU/USD data
def fetch_xau_data_daily():
    xau_data = yf.download("GC=F", interval="1d", start="2004-01-01")
    xau_data = xau_data.rename(columns={'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'})
    xau_data = xau_data.asfreq('D', method='ffill')
    xau_data['var_Prev_Close'] = xau_data['Close'].shift(1) - xau_data['Close'].shift(2)
    xau_data.dropna(inplace=True)
    return xau_data

@app.route('/fetch-xau-data', methods=['GET'])
def fetch_xau_data():
    df = fetch_xau_data_daily()
    
    # Reset index to avoid any issues with indices in JSON serialization
    df.reset_index(inplace=True)
    
    # Convert all column names and index names to strings
    df.columns = df.columns.map(str)
    
    # Convert any datetime index to string format
    if isinstance(df.index, pd.DatetimeIndex):
        df.index = df.index.astype(str)
    if 'Date' in df.columns:
        df['Date'] = df['Date'].astype(str)
    
    # Ensure all data is converted to Python native types, e.g., int, float, str
    data = df.to_dict(orient='records')
    
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
