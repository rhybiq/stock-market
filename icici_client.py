from breeze_connect import BreezeConnect
import urllib
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("ICICI_API_KEY")

# Initialize SDK
breeze = BreezeConnect(api_key=api_key)

# Obtain your session key from https://api.icicidirect.com/apiuser/login?api_key=YOUR_API_KEY
# Incase your api-key has special characters(like +,=,!) then encode the api key before using in the url as shown below.

print("https://api.icicidirect.com/apiuser/login?api_key="+urllib.parse.quote_plus(api_key))


# Generate Session
breeze.generate_session(api_secret="217341f5hx)e167T+9C99*7ZE753w424",
                        session_token="51232352")

# Connect to websocket(it will connect to tick-by-tick data server)


test = breeze.get_historical_data(interval="1minute",
                  from_date= "2025-02-03T09:20:00.000Z",
                  to_date= "2025-02-03T09:22:00.000Z",
                  stock_code="RELIND",
                  exchange_code="NSE",
                  product_type="cash")
print(test)
