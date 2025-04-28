import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import glob

# Import custom modules
from predictor import StockPredictor
from plot_utils import create_interactive_plot
from Technical_Indicators import calculate_indicators  # Fixed import

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('tcs_prediction')

def get_latest_model():
    """
    Get the path to the latest trained model in the models directory
    
    Returns:
    --------
    model_path : str
        Path to the latest model file
    metadata_path : str
        Path to the corresponding metadata file
    """
    # Get all model files
    model_files = glob.glob('models/stock_model_*.h5')
    
    if not model_files:
        logger.error("No model files found in the models directory")
        return None, None
    
    # Find the latest model by creation time
    latest_model = max(model_files, key=os.path.getctime)
    logger.info(f"Latest model found: {latest_model}")
    
    # Check if metadata file exists
    metadata_path = latest_model.replace('.h5', '_metadata.joblib')
    if not os.path.exists(metadata_path):
        logger.warning(f"Metadata file not found for model: {latest_model}")
        metadata_path = None
    
    return latest_model, metadata_path

def load_tcs_data():
    """
    Load and prepare TCS stock data
    
    Returns:
    --------
    data : pandas.DataFrame
        Processed TCS stock data
    """
    # Try to load from processed file first
    if os.path.exists('processed_TCS.csv'):
        logger.info("Loading processed TCS data")
        return pd.read_csv('processed_TCS.csv')
    
    # Otherwise load from raw file and process
    try:
        tcs_data_path = 'stock_data/TCS.csv'
        data = pd.read_csv(tcs_data_path)
        logger.info(f"Loaded TCS data from {tcs_data_path} with shape {data.shape}")
        
        # Check if data needs to be processed
        if 'RSI' not in data.columns:
            logger.info("Adding technical indicators to TCS data")
            data = calculate_indicators(data)  # Fixed function call
            
            # Save processed data for future use
            data.to_csv('processed_TCS.csv', index=False)
            logger.info("Saved processed TCS data to processed_TCS.csv")
        
        return data
    except Exception as e:
        logger.error(f"Error loading TCS data: {e}")
        return None

def predict_tcs_stock():
    """
    Main function to predict TCS stock using the latest trained model
    """
    # Get latest model
    model_path, _ = get_latest_model()
    if not model_path:
        logger.error("Could not find a trained model. Please train a model first.")
        return
    
    # Load TCS data
    tcs_data = load_tcs_data()
    if tcs_data is None or tcs_data.empty:
        logger.error("Failed to load TCS data")
        return
    
    # Initialize the predictor and load the model
    predictor = StockPredictor(model_dir='models')
    if not predictor.load_model(model_path):
        logger.error("Failed to load model")
        return
    
    # Make sure we have the right column names
    # First, detect the price column
    price_column = None
    for col in ['Price', 'Close', 'close', 'Closing Price', 'ClosingPrice', 'LTP']:
        if col in tcs_data.columns:
            price_column = col
            break
    
    if price_column is None:
        logger.error("No price column found in TCS data")
        return
    
    # Make predictions
    logger.info("Making predictions for TCS stock")
    # We use a moving window approach - each prediction is for the next day after the window
    predictions = predictor.predict(tcs_data)
    
    if predictions is None or len(predictions) == 0:
        logger.error("Failed to generate predictions")
        return
    
    # Create a dataframe with actual and predicted values
    # Predictions will be offset since each prediction is for a point after the sequence
    pred_df = pd.DataFrame()
    
    # Add date/time and actual prices
    if 'datetime' in tcs_data.columns:
        pred_df['Date'] = tcs_data['datetime'].iloc[predictor.sequence_length-1:].reset_index(drop=True)
    elif 'Date' in tcs_data.columns:
        pred_df['Date'] = tcs_data['Date'].iloc[predictor.sequence_length-1:].reset_index(drop=True)
    else:
        # Create a date range if no date column exists
        pred_df['Date'] = pd.date_range(start='today', periods=len(predictions), freq='D')
    
    # Add actual prices
    pred_df['Actual'] = tcs_data[price_column].iloc[predictor.sequence_length-1:].reset_index(drop=True)
    
    # Add predictions
    pred_df['Predicted'] = predictions
    
    # Save predictions to CSV
    pred_df.to_csv('TCS_predictions.csv', index=False)
    logger.info("Saved predictions to TCS_predictions.csv")
    
    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(pred_df['Date'], pred_df['Actual'], label='Actual', color='blue')
    plt.plot(pred_df['Date'], pred_df['Predicted'], label='Predicted', color='red')
    plt.title('TCS Stock Price Prediction')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('TCS_prediction_plot.png')
    plt.close()
    logger.info("Saved prediction plot to TCS_prediction_plot.png")
    
    # Create interactive plot using the plot_utils module
    # First, prepare the dataframe in the expected format
    interactive_df = tcs_data.copy()
    # Add predictions to the interactive dataframe
    interactive_df.loc[predictor.sequence_length-1:, 'Predicted'] = predictions
    
    # Set the correct date and price column names expected by the plotting function
    if 'datetime' in interactive_df.columns:
        interactive_df['PriceDate'] = interactive_df['datetime']
    elif 'Date' in interactive_df.columns:
        interactive_df['PriceDate'] = interactive_df['Date']
        
    # Ensure 'Price' column exists
    if price_column != 'Price':
        interactive_df['Price'] = interactive_df[price_column]
    
    # Create and save interactive plot
    create_interactive_plot(
        interactive_df, 
        save_path='.', 
        auto_open=False,
        show_rsi=True,
        show_macd=True
    )
    logger.info("Created interactive chart: interactive_chart.html")
    
    # Print summary statistics
    print("\nTCS Prediction Summary:")
    print(f"Number of days predicted: {len(predictions)}")
    print(f"Average actual price: ${pred_df['Actual'].mean():.2f}")
    print(f"Average predicted price: ${pred_df['Predicted'].mean():.2f}")
    
    # Calculate metrics
    mse = np.mean((pred_df['Actual'] - pred_df['Predicted'])**2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((pred_df['Actual'] - pred_df['Predicted']) / pred_df['Actual'])) * 100
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Absolute Percentage Error: {mape:.2f}%")
    
    # Prediction for next trading day
    latest_window = tcs_data.iloc[-predictor.sequence_length:].reset_index(drop=True)
    next_day_prediction = predictor.predict(latest_window)
    
    if next_day_prediction is not None and len(next_day_prediction) > 0:
        last_price = tcs_data[price_column].iloc[-1]
        next_price = next_day_prediction[0]
        change_pct = ((next_price - last_price) / last_price) * 100
        
        print(f"\nPrediction for next trading day:")
        print(f"Last known price: ${last_price:.2f}")
        print(f"Predicted price: ${next_price:.2f}")
        print(f"Predicted change: {change_pct:.2f}%")
        
        if change_pct > 0:
            print("Recommendation: BUY ⬆️")
        else:
            print("Recommendation: SELL ⬇️")
    
    logger.info("TCS prediction completed successfully")
    
    return pred_df

if __name__ == "__main__":
    predict_tcs_stock()