from dotenv import load_dotenv
import pandas as pd
from icici_breeze import ICICIBreezeAPI
import os

# Load environment variables from .env file
load_dotenv()

breeze = ICICIBreezeAPI()

# Add your ICICI Breeze credentials
user_id = os.getenv("ICICI_USER_ID")
password = os.getenv("ICICI_PASSWORD")

if breeze.login(user_id, password):
    print("Login successful!")
    
    # Get quote for HDFC Bank
    quote = breeze.get_stock_quote("NSE", "HDFCBANK")
    if quote:
        print(f"Current price of HDFC Bank: {quote.get('lastTradedPrice', 'N/A')}")
    
    # Get historical data for HDFC Bank
    hist_data = breeze.get_historical_data("NSE", "HDFCBANK", "2025-01-01", "2025-04-18", "1d")
    if isinstance(hist_data, pd.DataFrame):
        print(f"Retrieved {len(hist_data)} historical data points")
        
        # Basic analysis
        print(f"Highest price: {hist_data['high'].max()}")
        print(f"Lowest price: {hist_data['low'].min()}")
        print(f"Average closing price: {hist_data['close'].mean():.2f}")
else:
    print("Login failed. Check your credentials.")