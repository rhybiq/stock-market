import pandas as pd
import ta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('technical_indicators')

def calculate_indicators(df):
    if df is None or df.empty:
        logger.error("Cannot calculate indicators on empty DataFrame")
        return df
    
    # Make a copy of the dataframe to avoid modifying the original
    df = df.copy()
    
    # Handle different date column names
    date_column = None
    for col in ['PriceDate', 'Date', 'date', 'datetime', 'Timestamp', 'Time']:
        if col in df.columns:
            date_column = col
            # If it's not named 'PriceDate', create a 'PriceDate' column
            if col != 'PriceDate':
                df['PriceDate'] = df[col]
            break
    
    # Ensure the Price column exists or map from available column
    price_column = None
    for col in ['Price', 'Close', 'close', 'Closing Price', 'ClosingPrice', 'LTP']:
        if col in df.columns:
            price_column = col
            # If it's not named 'Price', create a 'Price' column
            if col != 'Price':
                df['Price'] = df[col]
            break
    
    if price_column is None:
        logger.error("No price column found in DataFrame")
        return df
        
    try:
        # Calculate RSI and identify conditions
        df = identify_rsi_conditions(df)
        
        # Calculate Bollinger Bands indicators
        df = identify_bollinger_indicators(df)
        # Calculate MACD
        df = identify_macd_indicators(df)
        
        logger.info("Technical indicators calculated successfully")
    except Exception as e:
        logger.error(f"Error calculating indicators: {e}")
    
    return df


def identify_stochastic_reversals(df):
    if df is None or df.empty:
        logger.error("Cannot identify reversals on empty DataFrame")
        return None
    
    # Make a copy of the dataframe
    df = df.copy()
    
    # Check if stochastic oscillator indicators are available
    df['Stoch_K'] = ta.momentum.stoch(df['high'], df['low'], df['Price'], window=14, smooth_window=3)
    df['Stoch_D'] = ta.momentum.stoch_signal(df['high'], df['low'], df['Price'], window=14, smooth_window=3)
        
    
    try:
        # Identify overbought conditions (Stochastic values above 80)
        df['Stoch_Overbought'] = ((df['Stoch_K'] > 80) & (df['Stoch_D'] > 80)).astype(int)
        
        # Identify oversold conditions (Stochastic values below 20)
        df['Stoch_Oversold'] = ((df['Stoch_K'] < 20) & (df['Stoch_D'] < 20)).astype(int)
        
        # Bullish reversal signal: K line crosses above D line while in oversold territory
        df['Stoch_Bullish_Cross'] = ((df['Stoch_K'] > df['Stoch_D']) & 
                                    (df['Stoch_K'].shift(1) <= df['Stoch_D'].shift(1)) & 
                                    (df['Stoch_Oversold'] == 1)).astype(int)
        
        # Bearish reversal signal: K line crosses below D line while in overbought territory
        df['Stoch_Bearish_Cross'] = ((df['Stoch_K'] < df['Stoch_D']) & 
                                    (df['Stoch_K'].shift(1) >= df['Stoch_D'].shift(1)) & 
                                    (df['Stoch_Overbought'] == 1)).astype(int)
        
        # Stronger reversal signal: Divergence detection
        # Price making higher highs while stochastic makes lower highs (bearish)
        # This is a simplified divergence detection - more complex implementation would require peak detection
        if 'Price' in df.columns:
            df['Price_Diff'] = df['Price'].diff(5)  # 5-period price difference
            df['Stoch_K_Diff'] = df['Stoch_K'].diff(5)  # 5-period stochastic difference
            
            # Bearish divergence: Price up, Stochastic down while overbought
            df['Stoch_Bearish_Divergence'] = ((df['Price_Diff'] > 0) & 
                                             (df['Stoch_K_Diff'] < 0) & 
                                             (df['Stoch_Overbought'] == 1)).astype(int)
            
            # Bullish divergence: Price down, Stochastic up while oversold
            df['Stoch_Bullish_Divergence'] = ((df['Price_Diff'] < 0) & 
                                             (df['Stoch_K_Diff'] > 0) & 
                                             (df['Stoch_Oversold'] == 1)).astype(int)
        
        # General reversal potential score (0-100)
        # Higher values indicate stronger reversal signals
        df['Stoch_Reversal_Score'] = 0
        
        # Add points for each reversal condition
        df.loc[df['Stoch_Bullish_Cross'] == 1, 'Stoch_Reversal_Score'] += 30
        df.loc[df['Stoch_Bearish_Cross'] == 1, 'Stoch_Reversal_Score'] += 30
        
        if 'Stoch_Bullish_Divergence' in df.columns:
            df.loc[df['Stoch_Bullish_Divergence'] == 1, 'Stoch_Reversal_Score'] += 40
            df.loc[df['Stoch_Bearish_Divergence'] == 1, 'Stoch_Reversal_Score'] += 40
        
        # Add points when stochastic reaches extreme values
        df.loc[df['Stoch_K'] < 5, 'Stoch_Reversal_Score'] += 15  # Extremely oversold
        df.loc[df['Stoch_K'] > 95, 'Stoch_Reversal_Score'] += 15  # Extremely overbought
        
        # Calculate reversal direction (1=bullish reversal, -1=bearish reversal, 0=no clear signal)
        df['Stoch_Reversal_Direction'] = 0
        df.loc[df['Stoch_Bullish_Cross'] == 1, 'Stoch_Reversal_Direction'] = 1
        df.loc[df['Stoch_Bearish_Cross'] == 1, 'Stoch_Reversal_Direction'] = -1
        
        if 'Stoch_Bullish_Divergence' in df.columns:
            df.loc[df['Stoch_Bullish_Divergence'] == 1, 'Stoch_Reversal_Direction'] = 1
            df.loc[df['Stoch_Bearish_Divergence'] == 1, 'Stoch_Reversal_Direction'] = -1
        
        # Create a new column stoch_Score which is the product of Stoch_Reversal_Score and Stoch_Reversal_Direction divided by 100
        stoch_score = (df['Stoch_Reversal_Score'] * df['Stoch_Reversal_Direction']) / 100
        
        logger.info("Stochastic reversal signals calculated successfully")

        # Make sure to include Stoch_K and Stoch_D in the list of columns to drop
        df['stoch_Score'] = stoch_score
        columns_to_drop = ['Stoch_Overbought', 'Stoch_Oversold', 'Stoch_Bullish_Cross',
                        'Stoch_Bearish_Cross', 'Price_Diff', 'Stoch_K_Diff', 'Stoch_K', 'Stoch_D']
        
        if 'Stoch_Bullish_Divergence' in df.columns:
            columns_to_drop.extend(['Stoch_Bullish_Divergence', 'Stoch_Bearish_Divergence'])
            
        columns_to_drop.extend(['Stoch_Reversal_Score', 'Stoch_Reversal_Direction'])
        
        # Only drop columns that exist in the dataframe
        columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        if columns_to_drop:
            df.drop(columns=columns_to_drop, inplace=True)
        
        # Return the DataFrame with stoch_Score
        return df
        
    except Exception as e:
        logger.error(f"Error calculating stochastic reversals: {e}")
        return None

# Update the calculate_advanced_indicators function to handle the new return type
def calculate_advanced_indicators(df):
    if df is None or df.empty:
        return df
    
    # Make a copy of the dataframe
    df = df.copy()
    
    # Ensure required columns exist
    if 'Price' not in df.columns:
        # Try to find a price column
        for col in ['Close', 'close', 'Closing Price', 'ClosingPrice', 'LTP']:
            if col in df.columns:
                df['Price'] = df[col]
                break
    
    # Check for high/low prices needed for some indicators
    high_col = 'high'
    low_col = 'low'
    
    
    # If we don't have high/low, we can't calculate some indicators
        
    try:
        # Add Stochastic Oscillator
        df['Stoch_K'] = ta.momentum.stoch(df[high_col], df[low_col], df['Price'], window=14, smooth_window=3)
        df['Stoch_D'] = ta.momentum.stoch_signal(df[high_col], df[low_col], df['Price'], window=14, smooth_window=3)
        
        # Add Average Directional Index (ADX)
        df['ADX'] = ta.trend.adx(df[high_col], df[low_col], df['Price'], window=14)
        
        # Add Ichimoku Cloud indicators
        ichimoku = ta.trend.IchimokuIndicator(df[high_col], df[low_col])
        df['Ichimoku_A'] = ichimoku.ichimoku_a()
        df['Ichimoku_B'] = ichimoku.ichimoku_b()
        
        # Add On-Balance Volume (OBV) if Volume is available
        if 'Volume' in df.columns or 'volume' in df.columns:
            vol_col = 'Volume' if 'Volume' in df.columns else 'volume'
            df['OBV'] = ta.volume.on_balance_volume(df['Price'], df[vol_col])
        
        # Add stochastic reversal signals (now returns just the stoch_Score)
        stoch_score = identify_stochastic_reversals(df)
        if stoch_score is not None:
            df['stoch_Score'] = stoch_score
            
        logger.info("Advanced indicators calculated successfully")
    except Exception as e:
        logger.error(f"Error calculating advanced indicators: {e}")
        
    return df

def identify_rsi_conditions(df, window=9, overbought_threshold=70, oversold_threshold=30):
    if df is None or df.empty:
        logger.error("Cannot identify RSI conditions on empty DataFrame")
        return df
    
    # Make a copy of the dataframe
    df = df.copy()
    
    # Ensure the Price column exists or map from available column
    price_column = None
    for col in ['Price', 'Close', 'close', 'Closing Price', 'ClosingPrice', 'LTP']:
        if col in df.columns:
            price_column = col
            # If it's not named 'Price', create a 'Price' column
            if col != 'Price':
                df['Price'] = df[col]
            break
    
    if price_column is None:
        logger.error("No price column found in DataFrame for RSI calculation")
        return df
    
    try:
        # Calculate RSI
        df['RSI'] = ta.momentum.rsi(df['Price'], window=window, fillna=True)
        
        # Identify oversold/overbought conditions
        df['Oversold'] = (df['RSI'] > overbought_threshold).astype(int)
        df['Undersold'] = (df['RSI'] < oversold_threshold).astype(int)
        df['Rsi_Indicator'] = df['Oversold'] - df['Undersold']
        # Remove the RSI column if it exists



        df.drop(columns=['Oversold', 'Undersold','RSI'], inplace=True)
        logger.info("RSI calculated and conditions identified successfully")
    except Exception as e:
        logger.error(f"Error identifying RSI conditions: {e}")
    
    return df

def identify_bollinger_indicators(df):
    if df is None or df.empty:
        logger.error("Cannot identify Bollinger indicators on empty DataFrame")
        return df
    
    # Make a copy of the dataframe
    df = df.copy()
    
    # Ensure the Price column exists or map from available column
    price_column = None
    for col in ['Price', 'Close', 'close', 'Closing Price', 'ClosingPrice', 'LTP']:
        if col in df.columns:
            price_column = col
            # If it's not named 'Price', create a 'Price' column
            if col != 'Price':
                df['Price'] = df[col]
            break
    
    if price_column is None:
        logger.error("No price column found in DataFrame for Bollinger Bands calculation")
        return df
    
    try:
        # Calculate Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['Price'], window=20, window_dev=2)
        df['Upper Band'] = bollinger.bollinger_hband_indicator()
        df['Lower Band'] = bollinger.bollinger_lband_indicator()
        df['BB_Ind'] = df['Upper Band'] - df['Lower Band']

        df.drop(columns=['Upper Band','Lower Band'], inplace=True)
        logger.info("Bollinger Bands calculated successfully")
    except Exception as e:
        logger.error(f"Error identifying Bollinger indicators: {e}")
    
    return df

def identify_macd_indicators(df, window_slow=26, window_fast=12, window_sign=9):
    if df is None or df.empty:
        logger.error("Cannot identify MACD indicators on empty DataFrame")
        return df
    
    # Make a copy of the dataframe
    df = df.copy()
    
    # Ensure the Price column exists or map from available column
    price_column = None
    for col in ['Price', 'Close', 'close', 'Closing Price', 'ClosingPrice', 'LTP']:
        if col in df.columns:
            price_column = col
            # If it's not named 'Price', create a 'Price' column
            if col != 'Price':
                df['Price'] = df[col]
            break
    
    if price_column is None:
        logger.error("No price column found in DataFrame for MACD calculation")
        return df
    
    try:
        # Calculate MACD
        df['MACD'] = ta.trend.macd(df['Price'], window_slow=window_slow, window_fast=window_fast, fillna=True)
        df['MACD_Signal'] = ta.trend.macd_signal(df['Price'], window_slow=window_slow, window_fast=window_fast, window_sign=window_sign, fillna=True)
        df['MACD_Hist'] = ta.trend.macd_diff(df['Price'], window_slow=window_slow, window_fast=window_fast, window_sign=window_sign, fillna=True)
        
        # Add MACD crossover signals
        df['MACD_Bullish_Cross'] = ((df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))).astype(int)
        df['MACD_Bearish_Cross'] = ((df['MACD'] < df['MACD_Signal']) & (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1))).astype(int)
        
        df['MACD_Trend'] = df['MACD_Bullish_Cross']-   df['MACD_Bearish_Cross']
        
        df.drop(columns=['MACD_Bullish_Cross', 'MACD_Bearish_Cross','MACD','MACD_Signal','MACD_Hist'], inplace=True)

        logger.info("MACD indicators calculated successfully")
    except Exception as e:
        logger.error(f"Error identifying MACD indicators: {e}")
    
    return df