from breeze_connect import BreezeConnect
import urllib
from dotenv import load_dotenv
import os
import pandas as pd
import logging
from datetime import datetime, timedelta
import time
import Technical_Indicators as ti

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('icici_client')

class ICICIClient:
    def __init__(self):
        """
        Initialize the ICICI client with API credentials
        """
        # Load environment variables from .env file
        load_dotenv()
        
        self.api_key = os.getenv("ICICI_API_KEY")
        self.api_secret = os.getenv("ICICI_API_SECRET")
        self.session_token = os.getenv("ICICI_SESSION_TOKEN")
        if not self.session_token:
            logger.error("Session token is missing. Please set ICICI_SESSION_TOKEN in the environment variables.")
        
        # Initialize the Breeze SDK
        self.breeze = BreezeConnect(api_key=self.api_key)
        self.is_connected = False
        
    def connect(self):
        """
        Connect to the ICICI API and generate a session
        """
        if not self.api_key:
            logger.error("API key is missing. Please set ICICI_API_KEY in the environment variables.")
            return False
            
        if not self.api_secret:
            logger.error("API secret is missing. Please set ICICI_API_SECRET in the environment variables.")
            return False
            
        if not self.session_token:
            logger.error("Session token is missing. Please set ICICI_SESSION_TOKEN in the environment variables.")
            return False
            
        try:
            # Generate Session using the proper method as per docs
            self.breeze.generate_session(
                api_secret=self.api_secret,
                session_token=self.session_token
            )
            logger.info(f"Session generated with token: {self.session_token}")
            
            # Validate the connection by fetching customer details
            try:
                customer_details = self.breeze.get_customer_details(self.session_token)
                if customer_details and 'Success' in customer_details:
                    client_name = customer_details.get('Success', {}).get('ClientName', 'Unknown')
                    logger.info(f"Connected to account: {client_name}")
                    print(f"Connected to account: {client_name}")
                    self.is_connected = True
                    return True
                else:
                    error_msg = customer_details.get('Error', {}).get('message', 'Unknown error')
                    logger.error(f"Failed to verify connection: {error_msg}")
                    print(f"Failed to verify connection: {error_msg}")
                    self.is_connected = False
                    return False
            except Exception as e:
                logger.error(f"Error verifying connection: {e}")
                print(f"Error verifying connection: {e}")
                self.is_connected = False
                return False
                
        except Exception as e:
            logger.error(f"Error generating session: {e}")
            print(f"Error generating session: {e}")
            self.is_connected = False
            return False
            
    def get_historical_data(self, stock_info, from_date=None, to_date=None, output_dir="stock_data"):
        """
        Get historical data for a stock with multiple fallback approaches
        
        Parameters:
        -----------        
        stock_info : tuple
            Tuple containing (exchange, stock_code)
        from_date : str
            Start date in format YYYY-MM-DD
        to_date : str
            End date in format YYYY-MM-DD
        output_dir : str
            Directory where stock data is saved
            
        Returns:
        --------
        df : pandas.DataFrame
            DataFrame with price data and calculated indicators
        """
        exchange, stock_code = stock_info
        
        # Check if the data already exists in a CSV file
        csv_file_path = os.path.join(output_dir, f"{stock_code}.csv")
        if os.path.exists(csv_file_path):
            logger.info(f"Data for {stock_code} already exists, loading from file")
            try:
                # Load data from CSV
                df = pd.read_csv(csv_file_path)
                if not df.empty:
                    # Ensure date column is in datetime format
                    if 'PriceDate' in df.columns:
                        df['PriceDate'] = pd.to_datetime(df['PriceDate'])
                    return df
            except Exception as e:
                logger.warning(f"Failed to read existing data for {stock_code}: {e}")
                # If file exists but can't be read, continue with API fetch
        
        # Set default dates if not provided
        if from_date is None:
            # Use 10 years of data by default
            from_date = (datetime.now() - timedelta(days=3650)).strftime('%Y-%m-%d')
        if to_date is None:
            to_date = datetime.now().strftime('%Y-%m-%d')
            
        logger.info(f"Fetching data for {exchange}:{stock_code} from {from_date} to {to_date}")
        
        # Check if connected
        if not self.is_connected:
            if not self.connect():
                logger.error("Not connected to ICICI API. Cannot fetch data.")
                return None
        
        all_data = []
        
        try:
            # Convert end date to datetime object for pagination calculations
            end_date_dt = datetime.strptime(to_date, "%Y-%m-%d")
            # Start with 1 year chunks to avoid hitting the 1000 record limit
            chunk_size_days = 365
            current_start_date = datetime.strptime(from_date, "%Y-%m-%d")
            
            # Try different intervals to get data
            intervals = ["1day"]
            # Try with both NSE and BSE exchanges
            exchanges_to_try = [exchange]
            if exchange == "NSE":
                exchanges_to_try.append("BSE")
            elif exchange == "BSE":
                exchanges_to_try.append("NSE")
            
            # Flag to indicate if we've successfully retrieved data for at least one chunk
            data_retrieved = False
            
            # Loop through date chunks
            while current_start_date < end_date_dt:
                # Calculate end date for current chunk
                current_end_date = min(current_start_date + timedelta(days=chunk_size_days), end_date_dt)
                
                # Format dates for API call
                from_date_iso = current_start_date.strftime("%Y-%m-%dT09:15:00.000Z")
                to_date_iso = current_end_date.strftime("%Y-%m-%dT15:30:00.000Z")
                
                logger.info(f"Fetching chunk: {current_start_date.strftime('%Y-%m-%d')} to {current_end_date.strftime('%Y-%m-%d')}")
                
                chunk_data = None
                
                # Try each exchange and interval
                exch="NSE"
                    
                for interval in intervals:
                    try:
                        logger.info(f"Trying with exchange: {exch}, interval: {interval}")
                        historical_data = self.breeze.get_historical_data_v2(
                            interval=interval,
                            from_date=from_date_iso,
                            to_date=to_date_iso,
                            stock_code=stock_code,
                            exchange_code=exch,
                            product_type="cash"
                        )
                        
                        if historical_data.get('Status') == 200 and historical_data.get('Success') and len(historical_data.get('Success')) > 0:
                            data_list = historical_data.get('Success')
                            chunk_data = pd.DataFrame(data_list)
                            logger.info(f"Successfully got {len(chunk_data)} records for chunk")
                            data_retrieved = True
                            break
                        else:
                            logger.info(f"No data for chunk with exchange: {exch}, interval: {interval}: {historical_data}")
                    except Exception as e:
                        logger.warning(f"Failed to get data for chunk with exchange: {exch}, interval: {interval}: {e}")
                        continue
            
                # Add chunk data to the combined dataset
                if chunk_data is not None and not chunk_data.empty:
                    all_data.append(chunk_data)
                
                # Move to next chunk
                current_start_date = current_end_date + timedelta(days=1)
            
            # Combine all chunks into a single DataFrame
            if all_data:
                df = pd.concat(all_data, ignore_index=True)
                # Drop duplicates if any
                if 'datetime' in df.columns:
                    df = df.drop_duplicates(subset=['datetime'], keep='first')
                    # Sort by date
                    df['datetime'] = pd.to_datetime(df['datetime'])
                    df = df.sort_values('datetime')
                
                logger.info(f"Total records after combining chunks: {len(df)}")
            else:
                if data_retrieved:
                    # If we got data but it's empty after processing
                    logger.warning(f"No valid data found for {stock_code} after all chunks")
                    return None
                else:
                    # If we didn't get any data at all
                    logger.warning(f"No data found for {stock_code} after all attempts")
                    return None
                
            # Calculate technical indicators
    
            
            # Save the newly fetched data to CSV
            if df is not None and not df.empty:
                self.save_stock_data(df, stock_code, output_dir)
            
            return df
                
        except Exception as e:
            logger.error(f"Error getting data for {stock_code}: {e}")
            return None
            
    def load_stock_symbols(self, file_path):
        """
        Load stock symbols from NSE Scrip Master file using pandas DataFrame
        
        Parameters:
        -----------        
        file_path : str
            Path to the NSE Scrip Master file
            
        Returns:
        --------
        list : List of tuples (exchange, stock_code)
        """
        stock_symbols = []
        
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                logger.error(f"File does not exist: {file_path}")
                return []
                
            logger.info(f"File exists: {file_path}, size: {os.path.getsize(file_path)} bytes")
            
            # Try to read file as CSV using pandas
            try:
                # First, attempt with comma delimiter (standard CSV)
                df = pd.read_csv(file_path, escapechar='\\')
                logger.info(f"Successfully read file as CSV with comma delimiter")
            except Exception as e:
                logger.warning(f"Failed to read with comma delimiter: {e}")
                # Try with tab delimiter
                try:
                    df = pd.read_csv(file_path, sep='\t', quotechar='"', escapechar='\\')
                    logger.info(f"Successfully read file as CSV with tab delimiter")
                except Exception as e:
                    logger.error(f"Failed to read with tab delimiter: {e}")
                    return []
            
            # Rename columns to remove leading/trailing spaces
            df.columns = df.columns.str.strip().str.replace('"', '', regex=False)
            
            # Log the DataFrame information
            logger.info(f"DataFrame shape: {df.shape}")
            logger.info(f"DataFrame columns: {df.columns.tolist()}")
            
            # Focus on the 'Symbol' column which contains stock symbols
            if 'Symbol' in df.columns:
                symbol_col = 'Symbol'
                logger.info(f"Using Symbol column for symbols")
            else:
                logger.error("Symbol column not found in DataFrame")
                exit(1)
            
            # Process each row to extract symbols
            valid_symbols = 0
            for _, row in df.iterrows():
                symbol = str(row[symbol_col]).strip()
                
                # Skip empty symbols or invalid ones
                if symbol and symbol not in ['0', '00', '000011', 'xyz', 'nan', 'None']:
                    stock_symbols.append(("NSE", symbol))
                    valid_symbols += 1
            
            logger.info(f"Found {valid_symbols} valid stock symbols")
            
            return stock_symbols
        
        except Exception as e:
            logger.error(f"Error loading stock symbols: {e}")
            return []
            
    def filter_nifty_250_stocks(self, stock_symbols, nifty_list_path="resources/Data/nifty250.csv"):
        """
        Filter stock symbols to only include those in the Nifty 250 index
        
        Parameters:
        -----------        
        stock_symbols : list
            List of tuples (exchange, stock_code)
        nifty_list_path : str
            Path to the CSV file containing Nifty 250 stocks
            
        Returns:
        --------
        list : Filtered list of tuples (exchange, stock_code)
        """
        try:
            # Check if the Nifty 250 list file exists
            if not os.path.exists(nifty_list_path):
                logger.error(f"Nifty 250 list file not found: {nifty_list_path}")
                return stock_symbols
                
            # Read the Nifty 250 list file
            nifty_df = pd.read_csv(nifty_list_path)
            
            # Check if the symbol column exists
            symbol_col = None
            for col in nifty_df.columns:
                if any(keyword in col.lower() for keyword in ['symbol', 'ticker', 'stock', 'shortname']):
                    symbol_col = col
                    break
                    
            if not symbol_col:
                logger.error(f"Could not find symbol column in Nifty 250 list file")
                return stock_symbols
                
            # Extract the list of Nifty 250 symbols
            nifty_symbols = set(nifty_df[symbol_col].str.strip().tolist())
            logger.info(f"Loaded {len(nifty_symbols)} symbols from Nifty 250 list")
            
            # Filter the stock symbols
            filtered_symbols = [(exchange, code) for exchange, code in stock_symbols 
                               if code in nifty_symbols]
            
            logger.info(f"Filtered {len(stock_symbols)} symbols down to {len(filtered_symbols)} Nifty 250 stocks")
            return filtered_symbols
            
        except Exception as e:
            logger.error(f"Error filtering Nifty 250 stocks: {e}")
            return stock_symbols
        
    def filter_nifty_100_stocks(self, stock_symbols, nifty_list_path="resources/Data/nifty100.csv"):
        """
        Filter stock symbols to only include those in the Nifty 100 index
        
        Parameters:
        -----------        
        stock_symbols : list
            List of tuples (exchange, stock_code)
        nifty_list_path : str
            Path to the CSV file containing Nifty 100 stocks
            
        Returns:
        --------
        list : Filtered list of tuples (exchange, stock_code)
        """
        try:
            # Check if the Nifty 100 list file exists
            if not os.path.exists(nifty_list_path):
                logger.error(f"Nifty 100 list file not found: {nifty_list_path}")
                return stock_symbols
                
            # Read the Nifty 100 list file
            try:
                # First check if file can be read with default parameters
                nifty_df = pd.read_csv(nifty_list_path)
                logger.info(f"Successfully read nifty100 file with default parameters")
            except Exception as e:
                logger.warning(f"Failed to read nifty100 file with default parameters: {e}")
                try:
                    # Try with tab delimiter
                    nifty_df = pd.read_csv(nifty_list_path, sep='\t')
                    logger.info(f"Successfully read nifty100 file with tab delimiter")
                except Exception as e:
                    logger.error(f"Failed to read nifty100 file: {e}")
                    return stock_symbols
            
            logger.info(f"Nifty100 DataFrame columns: {nifty_df.columns.tolist()}")
            
            # Check if the Symbol column exists directly
            if 'Symbol' in nifty_df.columns:
                symbol_col = 'Symbol'
                logger.info(f"Using 'Symbol' column directly")
            else:
                # Try to find a column that might contain symbol information
                symbol_col = None
                for col in nifty_df.columns:
                    if any(keyword in col.lower() for keyword in ['symbol', 'ticker', 'stock', 'shortname']):
                        symbol_col = col
                        break
                
                if not symbol_col:
                    # If still no symbol column found, use the first column as a fallback
                    if not nifty_df.empty and len(nifty_df.columns) > 0:
                        symbol_col = nifty_df.columns[0]
                        logger.warning(f"No explicit symbol column found, using first column: {symbol_col}")
                    else:
                        logger.error(f"Could not find any suitable column in Nifty 100 list file")
                        return stock_symbols
            
            # Extract the list of Nifty 100 symbols
            nifty_symbols = set(nifty_df[symbol_col].astype(str).str.strip().tolist())
            logger.info(f"Loaded {len(nifty_symbols)} symbols from Nifty 100 list")
            
            # Filter the stock symbols
            filtered_symbols = [(exchange, code) for exchange, code in stock_symbols 
                              if code in nifty_symbols]
            
            logger.info(f"Filtered {len(stock_symbols)} symbols down to {len(filtered_symbols)} Nifty 100 stocks")
            return filtered_symbols
            
        except Exception as e:
            logger.error(f"Error filtering Nifty 100 stocks: {e}")
            return stock_symbols

    def save_stock_data(self, df, stock_code, output_dir="stock_data"):
        """
        Save stock data to CSV file
        
        Parameters:
        -----------        
        df : pandas.DataFrame
            DataFrame with stock data
        stock_code : str
            Stock code/symbol
        output_dir : str
            Directory to save the CSV file
            
        Returns:
        --------
        bool : True if data was saved successfully, False otherwise
        """
        if df is None or df.empty:
            return False
            
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save data to CSV
        output_file = os.path.join(output_dir, f"{stock_code}.csv")
        df.to_csv(output_file, index=False)
        logger.info(f"Saved data for {stock_code} to {output_file}")
        return True

# Create a global instance of the client for backward compatibility
client = ICICIClient()

# For backward compatibility
def get_historical_data(stock_info, from_date=None, to_date=None):
    """Wrapper function for backward compatibility"""
    return client.get_historical_data(stock_info, from_date, to_date)

def load_stock_symbols(file_path):
    """Wrapper function for backward compatibility"""
    return client.load_stock_symbols(file_path)

def save_stock_data(df, stock_code, output_dir="stock_data"):
    """Wrapper function for backward compatibility"""
    return client.save_stock_data(df, stock_code, output_dir)
