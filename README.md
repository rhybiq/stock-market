# ICICI Breeze Stock Market Data Fetcher

This project allows you to fetch and visualize stock price data from the ICICI Breeze API.

## Setup

1. Clone this repository to your local machine.
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Edit the `.env` file to add your ICICI Breeze API credentials:
   - ICICI_API_KEY - Your API key from ICICI Breeze
   - ICICI_API_SECRET - Your API secret
   - ICICI_USER_ID - Your ICICI Direct user ID
   - ICICI_PASSWORD - Your ICICI Direct password

## Usage

### Fetching Stock Data

Run the main script to fetch stock prices:

```
python icici_breeze.py
```

This script demonstrates:
- Logging into the ICICI Breeze API
- Fetching real-time quotes for a stock
- Getting historical price data
- Basic price analysis

### Visualizing Stock Data

Run the visualization script:

```
python plot_stock_data.py
```

This script:
- Fetches historical data for a specified stock
- Creates a visualization with closing prices, price range, and moving averages
- Saves the chart as a PNG image
- Displays basic statistics for the stock

## Customization

You can modify the scripts to:
- Change the stock symbols
- Adjust the date ranges
- Modify the chart appearance
- Add more technical indicators

## Important Notes

- ICICI Breeze API has rate limits. Avoid making too many requests in a short period.
- Keep your API credentials secure and never commit the `.env` file to public repositories.
- The API endpoints may change. Refer to the official ICICI Breeze API documentation for updates.