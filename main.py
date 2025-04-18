from dotenv import load_dotenv
import pandas as pd
#from icici_breeze import ICICIBreezeAPI
#import os
import matplotlib
import matplotlib.pyplot as plt
import ta

# Load environment variables from .env file
load_dotenv()

def convert_amount(val):
    if isinstance(val, str):
        if val.endswith('M'):
            return float(val[:-1]) * 1_000_000
        elif val.endswith('K'):
            return float(val[:-1]) * 1_000
        elif val.endswith('B'):
            return float(val[:-1]) * 1_000_000_000
    try:
        return float(val)
    except:
        return None

#breeze = ICICIBreezeAPI()

# Add your ICICI Breeze credentials
#df = breeze.get_data("NSE", "SBIN", "2023-01-01", "2023-12-31", "day")
df = pd.read_csv("C:/Users/balaj/Documents/GitHub/stock-market/data.csv")
print(df.head())

#sorting by date
df['PriceDate']= pd.to_datetime(df['Date'], format='%m/%d/%Y')
df = df.sort_values(by='PriceDate')

#fixing volume
df['Volume'] = df['Vol.'].apply(convert_amount)


df['Rsi'] = ta.momentum.rsi(df['Price'],9,True)

df['Oversold'] = (df['Rsi'] > 70 ).astype(int)
df['Undersold'] = (df['Rsi'] < 30 ).astype(int)

df = df.drop('Rsi', axis=1)

df['Upper Band']= ta.volatility.BollingerBands(df['Price'],fillna=True).bollinger_hband_indicator() 
df['Lower Band']= ta.volatility.BollingerBands(df['Price'],fillna=True).bollinger_lband_indicator() 
df['MACD Signal']=ta.trend.macd_signal(df['Price'],fillna=True)

df = df.set_index('PriceDate')

plt.plot(df.index, df['Price'])
plt.scatter(x, y, kwargs)

