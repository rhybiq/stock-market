import os
import requests
import json
import pandas as pd
from dotenv import load_dotenv



class ICICIBreezeAPI:
    def __init__(self):
        self.base_url = "https://api.icicidirect.com/apibroker"
        self.api_key = os.getenv("ICICI_API_KEY")
        self.api_secret = os.getenv("ICICI_API_SECRET")
        self.session_token = None
        
    def login(self, user_id, password):
        """
        Login to ICICI Breeze API and get session token
        """
        url = f"{self.base_url}/auth/login"
        
        headers = {
            "Content-Type": "application/json",
            "X-API-KEY": self.api_key
        }
        
        payload = {
            "userId": user_id,
            "password": password
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            self.session_token = data.get("sessionToken")
            return True
        except requests.exceptions.RequestException as e:
            print(f"Login failed: {e}")
            return False
    
    def get_stock_quote(self, exchange, stock_code):
        """
        Get real-time stock quotes
        
        Parameters:
        -----------
        exchange : str
            Exchange code (NSE, BSE)
        stock_code : str
            Stock symbol/code
        """
        if not self.session_token:
            print("Please login first")
            return None
            
        url = f"{self.base_url}/quotes/v1/quote"
        
        headers = {
            "Content-Type": "application/json",
            "X-API-KEY": self.api_key,
            "X-SESSION-TOKEN": self.session_token
        }
        
        params = {
            "exchange": exchange,
            "stock_code": stock_code
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Failed to get stock quote: {e}")
            return None
    
    def get_historical_data(self, exchange, stock_code, from_date, to_date, interval="1d"):
        """
        Get historical price data
        
        Parameters:
        -----------
        exchange : str
            Exchange code (NSE, BSE)
        stock_code : str
            Stock symbol/code
        from_date : str
            Start date in format YYYY-MM-DD
        to_date : str
            End date in format YYYY-MM-DD
        interval : str
            Time interval (1d, 1h, 15m, etc.)
        """
        if not self.session_token:
            print("Please login first")
            return None
            
        url = f"{self.base_url}/historical/v1/historical"
        
        headers = {
            "Content-Type": "application/json",
            "X-API-KEY": self.api_key,
            "X-SESSION-TOKEN": self.session_token
        }
        
        params = {
            "exchange": exchange,
            "stock_code": stock_code,
            "from_date": from_date,
            "to_date": to_date,
            "interval": interval
        }
        
        try:
            response = requests.get(url, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Convert to pandas DataFrame for easier analysis
            if data:
                df = pd.DataFrame(data)
                return df
            return None
        except requests.exceptions.RequestException as e:
            print(f"Failed to get historical data: {e}")
            return None

# Example usage
