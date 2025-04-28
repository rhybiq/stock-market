import pandas as pd
import logging
import os
from plot_utils import create_interactive_plot, print_indicator_stats
import sys
import icici_client as icici
import time
from tqdm import tqdm
import Data as data
from predictor import StockPredictor  # Import the StockPredictor class
import matplotlib.pyplot as plt
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('stock_market_app')

def getData():
    scrip_master_file = "resources/Data/NSEScripMaster.txt"
    stock_list = icici.load_stock_symbols(scrip_master_file)

    # Filter to only Nifty 100 stocks
    stock_list = icici.client.filter_nifty_100_stocks(stock_list)
    output_dir = "stock_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Download data for each stock
    successful_downloads = 0
    failed_downloads = 0
    
    # Initialize empty DataFrame to store all stock data
    df = pd.DataFrame()
    
    # Use progress bar for better visualization
    for stock_info in tqdm(stock_list, desc="Downloading stock data"):
        exchange, stock_code = stock_info
        
        # Get historical data (now handles file existence check and saving internally)
        df_stock = icici.get_historical_data(stock_info)
        # Use concat instead of append (append is deprecated)
        if not df_stock.empty:
            # Add stock_code as a column to identify the stock
            if 'Symbol' not in df_stock.columns:
                df_stock['Symbol'] = stock_code
            df = pd.concat([df, df_stock], ignore_index=True)
        time.sleep(1)
    return df

def prepare_all_stocks_data():
    """
    Prepare a comprehensive dataset from all available stock files
    
    Returns:
    --------
    all_stocks_df : pandas.DataFrame
        DataFrame containing data from all stocks with technical indicators
    """
    # Check if processed file already exists
    if os.path.exists('all_stocks_with_indicators.csv'):
        logger.info("Loading existing processed data file")
        return pd.read_csv('all_stocks_with_indicators.csv')
    
    # Get stock data directory
    stock_dir = 'stock_data'
    
    # Get list of all stock files
    stock_files = [f for f in os.listdir(stock_dir) if f.endswith('.csv')]
    logger.info(f"Found {len(stock_files)} stock files")
    
    # Initialize empty DataFrame to store all stock data
    all_stocks_df = pd.DataFrame()
    
    # Process each stock file
    for stock_file in tqdm(stock_files, desc="Processing stock files"):
        # Extract symbol from filename
        symbol = stock_file.split('.')[0]
        
        # Read stock data
        file_path = os.path.join(stock_dir, stock_file)
        stock_df = pd.read_csv(file_path)
        
        # Add symbol column if not present
        if 'Symbol' not in stock_df.columns:
            stock_df['Symbol'] = symbol
            
        # Append to all stocks DataFrame
        all_stocks_df = pd.concat([all_stocks_df, stock_df], ignore_index=True)
    
    # Process data with technical indicators
    if not all_stocks_df.empty:
        all_stocks_df = data.prepare_data(all_stocks_df)
        
        # Save to CSV
        all_stocks_df.to_csv('all_stocks_with_indicators.csv', index=False)
        logger.info(f"Saved processed data for {len(stock_files)} stocks to all_stocks_with_indicators.csv")
    
    return all_stocks_df

def create_target_variables(df):
    """
    Create target variables for prediction:
    - Next day return
    - Price direction (up/down)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with stock data
        
    Returns:
    --------
    df : pandas.DataFrame
        DataFrame with added target variables
    """
    # Group by Symbol to handle each stock separately
    symbols = df['Symbol'].unique()
    result_df = pd.DataFrame()
    
    for symbol in symbols:
        stock_df = df[df['Symbol'] == symbol].copy().sort_values('datetime')
        
        # Calculate next day's closing price
        stock_df['next_close'] = stock_df['Price'].shift(-1)
        
        # Calculate return (percentage change)
        stock_df['target_next_day_return'] = (stock_df['next_close'] - stock_df['Price']) / stock_df['Price'] * 100
        
        # Calculate price direction (1 for up, 0 for down or same)
        stock_df['target_direction'] = (stock_df['next_close'] > stock_df['Price']).astype(int)
        
        # Add to result DataFrame
        result_df = pd.concat([result_df, stock_df], ignore_index=True)
    
    # Drop rows with NaN in target variables
    result_df = result_df.dropna(subset=['target_next_day_return', 'target_direction'])
    
    logger.info(f"Created target variables. DataFrame shape: {result_df.shape}")
    return result_df

def train_comprehensive_model(df, target_column='target_next_day_return'):
    """
    Train a comprehensive model using data from all stocks
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing processed stock data with indicators
    target_column : str
        Target column to predict
        
    Returns:
    --------
    predictor : StockPredictor
        Trained stock predictor model
    """
    # Import tensorflow and our preprocessing function
    import tensorflow as tf
    from predictor import data_preprocess_for_model
    
    # Check for required columns
    required_cols = ['Symbol', 'datetime', 'Price', target_column]
    for col in required_cols:
        if col not in df.columns:
            logger.error(f"Required column '{col}' not found in DataFrame")
            if col == target_column:
                logger.info("Creating target variables...")
                df = create_target_variables(df)
            else:
                return None
    
    # Check target variable characteristics
    if target_column in df.columns:
        target_min = df[target_column].min()
        target_max = df[target_column].max()
        target_mean = df[target_column].mean()
        target_std = df[target_column].std()
        logger.info(f"Target '{target_column}' statistics: min={target_min:.4f}, max={target_max:.4f}, mean={target_mean:.4f}, std={target_std:.4f}")
        
        # Check for extreme values that could cause instability
        if abs(target_min) > 50 or abs(target_max) > 50:
            logger.warning(f"Target variable has extreme values. Clipping to [-10, 10] range")
            df[target_column] = df[target_column].clip(-10, 10)
            
        # If target is percentage return, ensure it has reasonable distribution
        if 'return' in target_column.lower():
            # Most stock returns are within ±5% daily
            extreme_returns = df[(df[target_column] < -5) | (df[target_column] > 5)]
            if len(extreme_returns) > 0:
                logger.warning(f"Found {len(extreme_returns)} extreme return values (>±5%). Example rows:")
                logger.warning(extreme_returns[['Symbol', 'datetime', 'Price', target_column]].head())
    
    # Apply comprehensive preprocessing
    df = data_preprocess_for_model(df, target_column)
    
    # Split into train and test sets
    # Use 80% of the data for training, but maintain chronological order
    train_size = int(len(df) * 0.8)
    train_data = df.iloc[:train_size]
    test_data = df.iloc[train_size:]
    
    logger.info(f"Train data shape: {train_data.shape}, Test data shape: {test_data.shape}")
    
    # Create model directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Initialize the predictor
    predictor = StockPredictor(model_dir='models')
    
    # Define feature columns - exclude non-numeric and target-related columns
    exclude_cols = ['datetime', 'Symbol', 'Date', 'next_close', 
                    'target_next_day_return', 'target_direction']
    feature_cols = [col for col in df.columns 
                   if col not in exclude_cols 
                   and df[col].dtype in ['float64', 'int64']]
    
    logger.info(f"Using {len(feature_cols)} features for model training")
    
    # Set gradient clipping for stability
    tf.keras.backend.clear_session()
    
    # Train with optimized parameters
    try:
        # Set the target column on the predictor for later use in predictions
        predictor.target_column = target_column
        
        # Create early stopping callback - removed from direct passing
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        # Train the model - removed callbacks parameter
        history = predictor.train(
            data=train_data,
            target_column=target_column,
            feature_columns=feature_cols,
            sequence_length=15,  # Longer sequence for better patterns
            epochs=50,
            batch_size=32,
            model_type='lstm',
            validation_split=0.2
        )
        
        # Create training history plot
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Model Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('model_training_history.png')
        plt.close()
        logger.info("Saved model training history plot")
        
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        return None
    
    # Evaluate the model
    eval_results = predictor.evaluate(test_data, target_column)
    
    # Save the model
    model_path = predictor.save_model()
    logger.info(f"Comprehensive model saved to {model_path}")
    
    return predictor

def predict_tcs_stock(predictor, all_stocks_df, days_to_predict=30):
    """
    Use the comprehensive model to predict TCS stock prices
    
    Parameters:
    -----------
    predictor : StockPredictor
        Trained stock predictor model
    all_stocks_df : pandas.DataFrame
        DataFrame with all stocks data
    days_to_predict : int
        Number of days ahead to predict
        
    Returns:
    --------
    predictions : pandas.DataFrame
        DataFrame with predicted values for TCS
    """
    # Filter for TCS data
    tcs_data = all_stocks_df[all_stocks_df['Symbol'] == 'TCS'].copy()
    
    if tcs_data.empty:
        logger.error("No TCS data found in the dataset")
        return None
    
    # Sort by datetime
    tcs_data = tcs_data.sort_values('datetime').reset_index(drop=True)
    
    # Get the most recent data for prediction
    recent_data = tcs_data.tail(100)  # Use last 100 days of data
    
    logger.info(f"Using {len(recent_data)} recent data points for TCS prediction")
    
    # Make initial prediction using available data
    raw_predictions = predictor.predict(recent_data)
    
    # Get last known prices
    last_prices = recent_data['Price'].values[-len(raw_predictions):]
    
    # Only use the prediction data points that have corresponding actual values
    prediction_length = min(len(raw_predictions), len(last_prices))
    raw_predictions = raw_predictions[:prediction_length]
    last_prices = last_prices[:prediction_length]
    
    # If we're predicting percentage returns, convert to actual prices
    if predictor.target_column == 'target_next_day_return':
        logger.info(f"Converting percentage returns to price predictions. Raw range: {np.min(raw_predictions):.4f} to {np.max(raw_predictions):.4f}")
        
        # Check if predictions are severely off and need adjustment
        if np.min(raw_predictions) < -50 or np.max(raw_predictions) > 50:
            logger.warning("Raw predictions appear to be outside reasonable return ranges, applying scaling")
            # Scale predictions to a more reasonable range (-5% to +5%)
            raw_predictions = 5.0 * (raw_predictions - np.min(raw_predictions)) / (np.max(raw_predictions) - np.min(raw_predictions) + 1e-8) - 2.5
        
        # Convert percentage change to actual price
        # Formula: new_price = old_price * (1 + percentage/100)
        scaled_predictions = np.array([
            price * (1 + pred/100) 
            for price, pred in zip(last_prices, raw_predictions)
        ])
    else:
        # If not percentage returns, assume direct price prediction
        scaled_predictions = raw_predictions
        
        # Apply minimal scaling if predictions are very far from actual values
        pred_mean, pred_std = np.mean(scaled_predictions), np.std(scaled_predictions)
        actual_mean, actual_std = np.mean(last_prices), np.std(last_prices)
        
        # If prediction mean is more than 50% different from actual mean, rescale
        if abs(pred_mean - actual_mean) / actual_mean > 0.5:
            logger.warning(f"Predictions far from actual values. Applying rescaling: {pred_mean:.2f} → {actual_mean:.2f}")
            scaled_predictions = actual_mean + (scaled_predictions - pred_mean) * (actual_std / max(0.001, pred_std))
    
    # Ensure we have proper datetime values for the predictions
    dates = recent_data['datetime'].values[-len(scaled_predictions):]
    
    # Create a DataFrame to store the actual and predicted values
    prediction_df = pd.DataFrame({
        'Date': dates,
        'Actual_Close': last_prices,
        'Predicted_Close': scaled_predictions,
        'Raw_Prediction': raw_predictions  # Keep the raw predictions for reference
    })
    
    # Calculate error metrics
    mape = np.mean(np.abs((prediction_df['Actual_Close'] - prediction_df['Predicted_Close']) / 
                        prediction_df['Actual_Close'])) * 100
    rmse = np.sqrt(np.mean((prediction_df['Actual_Close'] - prediction_df['Predicted_Close'])**2))
    
    logger.info(f"TCS prediction MAPE: {mape:.2f}%, RMSE: {rmse:.2f}")
    
    # Save prediction results
    prediction_df.to_csv('TCS_predictions.csv', index=False)
    logger.info(f"TCS predictions saved to TCS_predictions.csv")
    
    # Create and save a visualization
    plt.figure(figsize=(12, 6))
    plt.plot(prediction_df['Date'], prediction_df['Actual_Close'], label='Actual Close')
    plt.plot(prediction_df['Date'], prediction_df['Predicted_Close'], label='Predicted Close')
    plt.title('TCS Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('TCS_prediction_plot.png')
    plt.close()
    
    # Also plot the raw predictions to understand model behavior
    plt.figure(figsize=(12, 6))
    plt.plot(prediction_df['Date'], prediction_df['Raw_Prediction'], label='Raw Predicted Returns (%)')
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    plt.title('TCS Raw Return Predictions')
    plt.xlabel('Date')
    plt.ylabel('Predicted Return (%)')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('TCS_raw_prediction_plot.png')
    
    logger.info(f"TCS prediction plots saved")
    
    return prediction_df

if __name__ == "__main__":
    logger.info("Starting comprehensive stock market prediction")
    
    # Step 1: Get and prepare all stocks data
    logger.info("Preparing comprehensive dataset from all stocks")
    all_stocks_df = prepare_all_stocks_data()
    
    if all_stocks_df.empty:
        logger.error("Failed to prepare stock data")
        sys.exit(1)
    
    # Step 2: Ensure target variables are created
    logger.info("Preparing target variables")
    all_stocks_df = create_target_variables(all_stocks_df)
    
    # Step 3: Train a comprehensive model on all stocks
    logger.info("Training comprehensive model on all stocks data")
    predictor = train_comprehensive_model(all_stocks_df, target_column='target_next_day_return')
    
    if predictor is None:
        logger.error("Failed to train comprehensive model")
        sys.exit(1)
    
    # Step 4: Use the model to predict TCS stock specifically
    logger.info("Predicting TCS stock with the comprehensive model")
    tcs_predictions = predict_tcs_stock(predictor, all_stocks_df)
    
    if tcs_predictions is not None:
        logger.info("Successfully completed TCS stock prediction")
        # Display the first few predictions
        print("\nTCS Stock Predictions (first 5 days):")
        print(tcs_predictions.head())
    else:
        logger.error("Failed to predict TCS stock")


