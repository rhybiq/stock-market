import os
import requests
import json
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timezone
import hashlib  

load_dotenv()

class ICICIBreezeAPI:
    def __init__(self):
        self.base_url = "https://api.icicidirect.com/breezeapi/api/v1/"
        self.api_key = os.getenv("ICICI_API_KEY")
        self.appkey = os.getenv("ICICI_APP_KEY")
        self.api_secret = os.getenv("ICICI_API_SECRET")
        self.session_token = 51232362
        
    def login(self):
        """
        Login to ICICI Breeze API and get session token
        """
        url = f"{self.base_url}customerdetails"
        
        
        customerDetail_payload = json.dumps({
                    "SessionToken": self.session_token,
                    "AppKey": self.api_key
                    })

        customerDetail_headers = {
            'Content-Type': 'application/json',
        }

        try:
            customerDetail_response = requests.request("GET", url, headers=customerDetail_headers, data=customerDetail_payload)
            data = json.loads(customerDetail_response.text)
            self.session_token = data["Success"]["session_token"]
            return True
        except requests.exceptions.RequestException as e:
            print(f"Login failed: {e}")
            return False
    
     
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
        time_stamp = datetime.now(timezone.utc).isoformat()[:19] + '.000Z'
        self.login()
        if not self.session_token:
            print("Please login first")
            return None
            
        url = f"{self.base_url}historicalcharts"
        
             
        
        payload = json.dumps({
            "interval": interval,
            "from_date": "2025-04-12T09:20:00.000Z",
            "to_date": "2025-04-14T09:20:00.000Z",
            "stock_code": stock_code,
            "exchange_code": exchange,
            "product_type": "Cash"
        }, separators=(',', ':'))
        
        checksum = hashlib.sha256((time_stamp+payload+self.api_secret).encode("utf-8")).hexdigest()

        headers = {
           'Content-Type': 'application/json',
            'X-Checksum': 'token '+ checksum,
            'X-Timestamp': time_stamp,
            'X-AppKey': self.api_key,
            'X-SessionToken': self.session_token
        }
        try:
            response = requests.request("GET", url, headers=headers, data=payload)
            response.raise_for_status()
            data = response.json()
            print(data)
            if data.get("Status") == "200":
            # Convert to pandas DataFrame for easier analysis
                if data:
                    df = pd.DataFrame(data)
                    return df
            return None
        except requests.exceptions.RequestException as e:
            print(f"Failed to get historical data: {e}")
            return None
        
        
    def get_data(self, exchange, stock_code, from_date, to_date, interval="1d"):
        """
        Get historical data and return as pandas DataFrame
        """
        df = self.get_historical_data(exchange, stock_code, from_date, to_date, interval)
        if df is not None:
            return df
        else:
            print("No data found")
            return None

# Example usage
