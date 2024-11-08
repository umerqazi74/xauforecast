from flask import Flask, jsonify
import yfinance as yf
import pandas as pd

app = Flask(__name__)

# Function to fetch daily XAU/USD data
def fetch_xau_data_daily():
    xau_data = yf.download("GC=F", interval="1d", start="2004-01-01")
    
    # Flatten the multi-level columns by joining tuple elements
    xau_data.columns = ['_'.join(filter(None, col)).strip() for col in xau_data.columns]
    
    xau_data = xau_data.asfreq('D', method='ffill')
    xau_data['var_Prev_Close'] = xau_data['Close_GC=F'].shift(1) - xau_data['Close_GC=F'].shift(2)
    xau_data.dropna(inplace=True)
    return xau_data

@app.route('/fetch-xau-data', methods=['GET'])
def fetch_xau_data():
    df = fetch_xau_data_daily()
    
    # Reset index to avoid any issues with indices in JSON serialization
    df.reset_index(inplace=True)
    
    # Convert any datetime index to string format
    if 'Date' in df.columns:
        df['Date'] = df['Date'].astype(str)
    
    # Convert the DataFrame to a dictionary format that is JSON serializable
    data = df.to_dict(orient='records')
    
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
