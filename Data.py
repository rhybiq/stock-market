import pandas as pd
import logging
import Technical_Indicators as ti
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def prepare_data(df):
    if df is None or df.empty:
        logging.error("Cannot prepare data on empty DataFrame")
        return df
    
    df = df.copy()
    
    grouped_data = []
    unique_stocks = df['stock_code'].unique()
    
    logging.info(f"Processing {len(unique_stocks)} unique stocks")
    
    for stock in unique_stocks:
        stock_data = df[df['stock_code'] == stock].copy()
        
        # Sort data by datetime
        stock_data = stock_data.sort_values('datetime')
        
        # Apply technical indicators
        stock_data = ti.identify_rsi_conditions(stock_data)
        stock_data = ti.identify_bollinger_indicators(stock_data)
        stock_data = ti.identify_macd_indicators(stock_data)
        stock_data = ti.identify_stochastic_reversals(stock_data)

        # Rationalize data for machine learning
        stock_data = rationalize_for_ml(stock_data)

        grouped_data.append(stock_data)
    
    processed_df = pd.concat(grouped_data, ignore_index=True)
    
    logging.info(f"Processed {len(processed_df)} rows with indicators for machine learning")
    
    return processed_df

def rationalize_for_ml(df):
    """
    Rationalize data for machine learning algorithms by:
    1. Handling missing values
    2. Normalizing/scaling features
    3. Creating derived features
    4. Handling categorical variables
    5. Feature selection
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with stock data and calculated indicators
        
    Returns:
    --------
    df : pandas.DataFrame
        DataFrame prepared for machine learning
    """
    if df is None or df.empty:
        logging.error("Cannot rationalize empty DataFrame")
        return df
    
    # Make a copy
    df = df.copy()
    
    # 1. Handle missing values
    # For numeric columns, fill NaN with median of that column
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # 2. Create lagged features for time series modeling
    # These help the model learn from past data
    indicator_cols = ['Rsi_Indicator', 'BB_Ind', 'MACD_Trend', 'stoch_Score']
    price_cols = [col for col in df.columns if col in ['Price', 'Close', 'open', 'high', 'low']]
    
    for col in indicator_cols + price_cols:
        if col in df.columns:
            # Create lag features (t-1, t-2, t-3)
            for lag in [1, 2, 3, 5]:
                df[f'{col}_lag_{lag}'] = df[col].shift(lag)
            
            # Create rolling mean features
            for window in [3, 5, 10]:
                df[f'{col}_rolling_{window}'] = df[col].rolling(window=window).mean()
    
    # 3. Feature engineering: create additional useful features
    
    # Volatility (standard deviation of price over last n periods)
    if 'Price' in df.columns:
        df['volatility_5d'] = df['Price'].rolling(window=5).std()
        df['volatility_10d'] = df['Price'].rolling(window=10).std()
        
        # Percentage price change
        df['price_pct_change_1d'] = df['Price'].pct_change(1)
        df['price_pct_change_5d'] = df['Price'].pct_change(5)
    
    # 4. Scale/normalize features for better model performance
    # Use StandardScaler for normal distribution, MinMaxScaler for bounded algorithms
    
    # Create a list of columns to scale - typically the numeric features
    cols_to_scale = [col for col in df.columns if df[col].dtype in ['float64', 'int64'] 
                    and col not in ['datetime', 'stock_code', 'Symbol']]
    
    # Handle any remaining missing values before scaling
    for col in cols_to_scale:
        df[col] = df[col].fillna(0)
    
    # Apply StandardScaler to relevant columns
    if cols_to_scale:
        scaler = StandardScaler()
        df_scaled = pd.DataFrame(
            scaler.fit_transform(df[cols_to_scale]),
            columns=cols_to_scale,
            index=df.index
        )
        
        # Replace original columns with scaled versions
        for col in cols_to_scale:
            df[f'{col}_scaled'] = df_scaled[col]
    
    # 5. Drop rows with NaN values that couldn't be filled
    # This is typically rows at the beginning where lagged features created NaNs
    df = df.dropna()
    
    # 6. Create a target variable for supervised learning
    # Example: future price movement direction (1 for up, 0 for down)
    if 'Price' in df.columns:
        # Forward-looking price change (next day's return)
        df['target_next_day_return'] = df['Price'].pct_change(1).shift(-1)
        df['target_next_day_direction'] = (df['target_next_day_return'] > 0).astype(int)
        
        # 5-day future return
        df['target_5d_return'] = df['Price'].pct_change(5).shift(-5)
        df['target_5d_direction'] = (df['target_5d_return'] > 0).astype(int)
    
    logging.info(f"Rationalized DataFrame for ML with {len(df.columns)} features")
    
    return df