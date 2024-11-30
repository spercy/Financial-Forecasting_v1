import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Fetch stock data
def fetch_stock_data(stock_symbol, period="1y", interval="1d"):
    data = yf.download(stock_symbol, period=period, interval=interval)
    data['Spread'] = data['High'] - data['Low']  # Calculate spread
    data['Volatility'] = (data['High'] - data['Low']) / data['Open']  # Approximate volatility
    data['Date'] = data.index
    return data

# Add features (example: lagged features for machine learning)
def add_features(data):
    data['Prev Close'] = data['Close'].shift(1)
    data['Prev Volume'] = data['Volume'].shift(1)
    data['Short Interest'] = np.random.uniform(0.1, 0.5, len(data))  # Simulated data for demo purposes
    data.dropna(inplace=True)
    return data

# Prepare data for machine learning
def prepare_data(data):
    features = ['Prev Close', 'Volume', 'Short Interest', 'Volatility', 'Spread']
    X = data[features]
    y = data['Close']  # Predicting next day's close price
    return train_test_split(X, y, test_size=0.2, random_state=42)

# Train and evaluate model
def train_model(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    print(f"Mean Squared Error: {mse:.2f}")
    return model, predictions

# Plot predictions vs actuals
def plot_results(y_test, predictions):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.values, label="Actual Prices", color="blue")
    plt.plot(predictions, label="Predicted Prices", color="orange")
    plt.title("Actual vs Predicted Prices")
    plt.xlabel("Test Data Points")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.show()

# Main Execution
if __name__ == "__main__":
    stock_symbol = "AAPL"
    stock_data = fetch_stock_data(stock_symbol, period="1y")
    stock_data = add_features(stock_data)
    
    print(f"Data after adding features: {stock_data.head()}")
    
    X_train, X_test, y_train, y_test = prepare_data(stock_data)
    
    print(f"Training data points: {len(X_train)}, Testing data points: {len(X_test)}")
    
    model, predictions = train_model(X_train, y_train, X_test, y_test)
    
    plot_results(y_test, predictions)
