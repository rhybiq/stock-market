from dotenv import load_dotenv
import pandas as pd
from icici_breeze import ICICIBreezeAPI
import os

# Load environment variables from .env file
load_dotenv()

breeze = ICICIBreezeAPI()

# Add your ICICI Breeze credentials
df = breez.get_data("NSE", "RELIANCE", "2023-01-01", "2023-12-31", "1d")

