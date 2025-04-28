from flask import Flask, render_template, request, jsonify, Response, send_file
import os
import pandas as pd
import io
from main import process
from plot_utils import create_interactive_plot, print_indicator_stats
import plotly
import json
import plotly.io as pio
from dotenv import load_dotenv, set_key
import logging
from Technical_Indicators import calculate_indicators, calculate_advanced_indicators
import time
import pandas as pd
import logging
import os
import time
from tqdm import tqdm
import icici_client
from Data import get_indicators

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('stock_market_app')

def process():
    """
    Process stock market data - connects to ICICI API, fetches historical data,
    and saves it to the stock_data directory.
    
    Returns:
        tuple: (successful_downloads, failed_downloads, output_dir) with counts and path
    """
    # Verify API connection first
    logger.info("Verifying ICICI API connection...")
    if not icici_client.client.connect():
        logger.error("Failed to connect to ICICI API. Please check your API credentials and session token.")
        print("ERROR: Failed to connect to ICICI API. Please check your API credentials and session token.")
        return 0, 0, None
    else:
        logger.info("Successfully connected to ICICI API")
        print("Successfully connected to ICICI API")
    
    # Load stock symbols from NSE Scrip Master file
    scrip_master_file = "resources/Data/NSEScripMaster.txt"
    stock_list = icici_client.load_stock_symbols(scrip_master_file)

    # Filter to only Nifty 100 stocks
    stock_list = icici_client.client.filter_nifty_100_stocks(stock_list)
    
    # # Limit to first 10 symbols for testing
    # test_limit = 10
    # if len(stock_list) > test_limit:
    #     logger.info(f"Limiting to first {test_limit} symbols for testing purposes")
    #     stock_list = stock_list[:test_limit]
    
    # Create output directory
    output_dir = "stock_data"
    os.makedirs(output_dir, exist_ok=True)
    
    # Download data for each stock
    successful_downloads = 0
    failed_downloads = 0
    
    # Use progress bar for better visualization
    for stock_info in tqdm(stock_list, desc="Downloading stock data"):
        exchange, stock_code = stock_info
        
        # Get historical data (now handles file existence check and saving internally)
        df_stock = icici_client.get_historical_data(stock_info)
        
        if df_stock is not None and not df_stock.empty:
            successful_downloads += 1
            
            # Add technical indicators to the stock data
            try:
                # Apply technical indicators
                df_stock_with_indicators = get_indicators(df_stock)
                
                # Save the data with indicators
                file_path = os.path.join(output_dir, f"{stock_code}_with_indicators.csv")
                df_stock_with_indicators.to_csv(file_path, index=False)
                logger.info(f"Saved data with indicators for {stock_code}")
            except Exception as e:
                logger.error(f"Failed to add indicators for {stock_code}: {e}")
        else:
            logger.warning(f"Failed to get data for {stock_code}")
            failed_downloads += 1
        
        if df_stock is not None:
            logger.info(f"DataFrame columns: {df_stock.columns.tolist()}")
            
        # Sleep to avoid rate limiting
        time.sleep(1)
    
    logger.info(f"Downloaded data for {successful_downloads} stocks, failed for {failed_downloads} stocks")
    print(f"Downloaded data for {successful_downloads} stocks, failed for {failed_downloads} stocks")
    print(f"Data saved to {os.path.abspath(output_dir)}")
    
    return successful_downloads, failed_downloads, output_dir

app = Flask(__name__, 
            template_folder='resources/templates',  # Updated template directory path
            static_folder='resources/static')

# Load environment variables
load_dotenv()

# Global dictionary to store preloaded stock data
preloaded_stocks = {}

def init():
    """
    Initialize the application by processing stock market data and storing files.
    This function calls the process() function from main.py to download and process data,
    then preloads all stock files with calculated indicators.
    """
    global preloaded_stocks
    
    logger.info("Initializing application - processing stock market data...")
    successful_downloads, failed_downloads, output_dir = process()
    
    if successful_downloads > 0:
        logger.info(f"Initialization complete. Downloaded {successful_downloads} stocks, failed {failed_downloads}.")
        logger.info(f"Data stored in {os.path.abspath(output_dir)}")
        
        # Preload all stock files
        logger.info("Preloading stock files and calculating indicators...")
        stock_names = get_stock_files()
        total_files = len(stock_names)
        logger.info(f"Found {total_files} stock files to load")
        
        for i, stock_name in enumerate(stock_names):
            try:
                # Load the stock data (add .csv extension for file path)
                stock_path = os.path.join("stock_data", stock_name + ".csv")
                stock_data = pd.read_csv(stock_path)
                
                # Calculate technical indicators
                stock_data = calculate_all_indicators(stock_data)
                
                # Store in global dictionary
                preloaded_stocks[stock_name] = {
                    'data': stock_data,
                    'file': stock_name + ".csv",
                    'last_updated': time.time()
                }
                
                if (i + 1) % 50 == 0 or (i + 1) == total_files:
                    logger.info(f"Preloaded {i + 1}/{total_files} stocks")
                
            except Exception as e:
                logger.error(f"Error preloading stock {stock_name}: {str(e)}")
        
        logger.info(f"Finished preloading {len(preloaded_stocks)} stock files with indicators")
        return True
    else:
        logger.error("Initialization failed. No data was downloaded.")
        return False

def prepare_data(file_path):
    """
    Load and prepare data from a CSV file
    
    Parameters:
    -----------
    file_path : str
        Path to the CSV file
    
    Returns:
    --------
    pandas.DataFrame
        Prepared DataFrame with datetime index
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # If datetime column exists, convert it to datetime type
        date_col = None
        for col in ['datetime', 'date', 'Date', 'PriceDate', 'Time', 'Timestamp']:
            if col in df.columns:
                date_col = col
                df[col] = pd.to_datetime(df[col])
                break
        
        if date_col is not None:
            # Sort data by date
            df = df.sort_values(by=date_col)
            
            # Create PriceDate column if it doesn't exist
            if 'PriceDate' not in df.columns:
                df['PriceDate'] = df[date_col]
        
        logger.info(f"Loaded and prepared data from {file_path}")
        return df
    
    except Exception as e:
        logger.error(f"Error preparing data from {file_path}: {str(e)}")
        raise

def calculate_all_indicators(df):
    """
    Calculate all technical indicators for a given DataFrame
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with price data
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame with calculated indicators
    """
    try:
        # Calculate basic indicators
        df = calculate_indicators(df)
        
        # Try to calculate advanced indicators if high/low data is available
        if all(col in df.columns for col in ['high', 'low']) or all(col in df.columns for col in ['High', 'Low']):
            df = calculate_advanced_indicators(df)
        
        logger.info("All technical indicators calculated successfully")
        return df
    
    except Exception as e:
        logger.error(f"Error calculating all indicators: {str(e)}")
        return df

def get_stock_files():
    """Get the list of available stock CSV files in the stock_data directory but return names without .csv extension."""
    stock_dir = "stock_data"
    if not os.path.exists(stock_dir):
        return []
    
    # Get all CSV files but remove the .csv extension for display
    csv_files = [f for f in os.listdir(stock_dir) if f.endswith('.csv')]
    # Return stock names without the .csv extension
    stock_names = [f.rsplit('.', 1)[0] for f in csv_files]
    return sorted(stock_names)

@app.route('/')
def index():
    # Get list of available stock files
    stock_files = get_stock_files()
    return render_template('index.html', stock_files=stock_files)

@app.route('/generate_chart', methods=['GET', 'POST'])
def generate_chart():
    global preloaded_stocks
    
    # Default to showing an error if no stock is selected
    if request.method != 'POST' or 'stock_file' not in request.form:
        return render_template('error.html', error="Please select a stock from the list")
    
    # Get the selected stock
    selected_stock = request.form['stock_file']
    
    # Check if the selected stock is preloaded
    if selected_stock not in preloaded_stocks:
        # If not in preloaded stocks, try to load it
        stock_path = os.path.join("stock_data", selected_stock + ".csv")
        
        if not os.path.exists(stock_path):
            return render_template('error.html', error=f"Stock data for '{selected_stock}' not found")
        
        try:
            # Load and prepare the stock data
            stock_data = prepare_data(stock_path)
            
            # Calculate technical indicators
            stock_data = calculate_all_indicators(stock_data)
            
            # Store in preloaded_stocks for future use
            preloaded_stocks[selected_stock] = {
                'data': stock_data,
                'file': selected_stock + ".csv",
                'last_updated': time.time()
            }
            
            logger.info(f"Loaded and added {selected_stock} to preloaded stocks")
        except Exception as e:
            logger.error(f"Error loading stock {selected_stock}: {str(e)}")
            return render_template('error.html', error=f"Error processing stock data: {str(e)}")
    
    # Get the stock data from preloaded_stocks
    stock_data = preloaded_stocks[selected_stock]['data']
    
    # Ensure PriceDate is in proper datetime format for Plotly
    if 'PriceDate' in stock_data.columns:
        if not pd.api.types.is_datetime64_any_dtype(stock_data['PriceDate']):
            stock_data['PriceDate'] = pd.to_datetime(stock_data['PriceDate'])
    
    # Display indicator stats in console for debugging
    print_indicator_stats(stock_data)
    
    # Create interactive plot but don't auto open - disable RSI and MACD panels
    fig, _ = create_interactive_plot(stock_data, auto_open=False, show_rsi=False, show_macd=False)
    
    # Update layout for better responsiveness
    fig.update_layout(
        autosize=True,
        title=f"Stock Analysis: {selected_stock}",
        width=None,
        height=None
    )
    
    # Convert plotly figure to JSON for direct embedding
    plot_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Get indicator statistics
    stats = {
        "oversold": int(stock_data['Oversold'].sum() if 'Oversold' in stock_data.columns else 0),
        "undersold": int(stock_data['Undersold'].sum() if 'Undersold' in stock_data.columns else 0),
        "upper_band": int(stock_data['Upper Band'].sum() if 'Upper Band' in stock_data.columns else 0),
        "lower_band": int(stock_data['Lower Band'].sum() if 'Lower Band' in stock_data.columns else 0),
        "stock_name": selected_stock,
        "data_source": "Preloaded",  # Always show as preloaded
        "available_indicators": [ind for ind in ['RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'Upper Band', 'Lower Band'] 
                              if ind in stock_data.columns]
    }
    
    # Safely add date information
    if 'PriceDate' in stock_data.columns and not stock_data.empty:
        try:
            # Check if PriceDate is already a datetime
            if pd.api.types.is_datetime64_any_dtype(stock_data['PriceDate']):
                stats["start_date"] = stock_data['PriceDate'].min().strftime('%Y-%m-%d')
                stats["end_date"] = stock_data['PriceDate'].max().strftime('%Y-%m-%d')
            else:
                # Try to convert to datetime first
                start_date = pd.to_datetime(stock_data['PriceDate'].min())
                end_date = pd.to_datetime(stock_data['PriceDate'].max())
                stats["start_date"] = start_date.strftime('%Y-%m-%d')
                stats["end_date"] = end_date.strftime('%Y-%m-%d')
        except Exception as e:
            logger.error(f"Error formatting dates: {str(e)}")
            # Fallback to string representation
            stats["start_date"] = str(stock_data['PriceDate'].min())
            stats["end_date"] = str(stock_data['PriceDate'].max())
    else:
        stats["start_date"] = "N/A"
        stats["end_date"] = "N/A"
    
    return render_template('chart.html', plot_json=plot_json, stats=stats)

if __name__ == "__main__":
    # Initialize the application to process stock data before running the server
    init()
    # Start the Flask server
    app.run(debug=False)

