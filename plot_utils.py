import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import plotly.io as pio
import logging

# Configure logging
logger = logging.getLogger('plot_utils')

def create_interactive_plot(df, save_path=None, auto_open=True, show_rsi=False, show_macd=False):
    """
    Create an interactive plot with Plotly showing price data, with optional technical indicators.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing price data and calculated indicators
    save_path : str, optional
        Path where to save the HTML file. If None, saves to current directory
    auto_open : bool, optional
        Whether to automatically open the plot in a browser
    show_rsi : bool, optional
        Whether to show the RSI indicator panel
    show_macd : bool, optional
        Whether to show the MACD indicator panel
        
    Returns:
    --------
    fig : plotly.graph_objects.Figure
        The created figure object
    html_file : str
        Path to the saved HTML file
    """
    # Make a copy of the dataframe to avoid modifying the original
    df = df.copy()
    
    # Check for required columns and handle different column names
    # First, ensure we have a price column
    price_column = None
    for col in ['Price', 'Close', 'close', 'Closing Price', 'ClosingPrice', 'LTP']:
        if col in df.columns:
            price_column = col
            # If it's not named 'Price', create a 'Price' column
            if col != 'Price':
                df['Price'] = df[col]
            break
    
    if price_column is None:
        logger.error("No price column found in the data")
        raise ValueError("The stock data doesn't contain a recognizable price column. Expected 'Price', 'Close', or similar.")
    
    # Check for date column
    date_column = None
    for col in ['PriceDate', 'Date', 'date', 'Timestamp', 'Time']:
        if col in df.columns:
            date_column = col
            # If it's not named 'PriceDate', create a 'PriceDate' column
            if col != 'PriceDate':
                df['PriceDate'] = df[col]
            break
    
    if date_column is None:
        logger.error("No date column found in the data")
        raise ValueError("The stock data doesn't contain a recognizable date column. Expected 'PriceDate', 'Date', or similar.")
    
    # Ensure required indicator columns exist
    required_cols = ['RSI', 'MACD', 'MACD_Signal', 'MACD_Hist', 'Upper Band', 'Lower Band', 'Oversold', 'Undersold']
    for col in required_cols:
        if col not in df.columns:
            logger.warning(f"Missing column '{col}', adding with default values")
            df[col] = 0  # Default value
    
    # Create the subplots based on which indicators to show
    if show_rsi and show_macd:
        # All indicators
        fig = make_subplots(rows=3, cols=1, 
                          shared_xaxes=True,
                          row_heights=[0.6, 0.2, 0.2],
                          vertical_spacing=0.03,
                          subplot_titles=('Price Chart with Indicators', 'RSI', 'MACD'))
    elif show_rsi:
        # Only show RSI
        fig = make_subplots(rows=2, cols=1, 
                          shared_xaxes=True,
                          row_heights=[0.7, 0.3],
                          vertical_spacing=0.03,
                          subplot_titles=('Price Chart with Indicators', 'RSI'))
    elif show_macd:
        # Only show MACD
        fig = make_subplots(rows=2, cols=1, 
                          shared_xaxes=True,
                          row_heights=[0.7, 0.3],
                          vertical_spacing=0.03,
                          subplot_titles=('Price Chart with Indicators', 'MACD'))
    else:
        # Just show price chart
        fig = make_subplots(rows=1, cols=1,
                          subplot_titles=('Price Chart with Indicators',))

    # Add price to main chart
    fig.add_trace(go.Scatter(
        x=df['PriceDate'], 
        y=df['Price'], 
        name='Price', 
        line=dict(color='blue', width=1.5)
    ), row=1, col=1)

    # Add Upper and Lower Band indicator points
    upper_band_points = df[df['Upper Band'] == 1]
    lower_band_points = df[df['Lower Band'] == 1]

    fig.add_trace(go.Scatter(
        x=upper_band_points['PriceDate'], 
        y=upper_band_points['Price'], 
        mode='markers',
        name='Upper Band Signal',
        marker=dict(color='purple', size=8, symbol='triangle-down')
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=lower_band_points['PriceDate'], 
        y=lower_band_points['Price'], 
        mode='markers',
        name='Lower Band Signal',
        marker=dict(color='orange', size=8, symbol='triangle-up')
    ), row=1, col=1)

    # Add oversold/undersold markers
    oversold_df = df[df['Oversold'] == 1]
    undersold_df = df[df['Undersold'] == 1]

    fig.add_trace(go.Scatter(
        x=oversold_df['PriceDate'], 
        y=oversold_df['Price'], 
        mode='markers',
        name='Oversold (Sell Signal)',
        marker=dict(color='red', size=8, symbol='triangle-down')
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=undersold_df['PriceDate'], 
        y=undersold_df['Price'], 
        mode='markers',
        name='Undersold (Buy Signal)',
        marker=dict(color='green', size=8, symbol='triangle-up')
    ), row=1, col=1)

    # Add stoch_score signals when above 0.80 or below -0.80
    if 'stoch_Score' in df.columns:
        strong_bullish_signals = df[df['stoch_Score'] > 0.50]
        strong_bearish_signals = df[df['stoch_Score'] < -0.50]
        
        fig.add_trace(go.Scatter(
            x=strong_bullish_signals['PriceDate'], 
            y=strong_bullish_signals['Price'], 
            mode='markers',
            name='Strong Bullish Signal (stoch_score > 0.50)',
            marker=dict(color='lime', size=10, symbol='star', line=dict(width=1, color='darkgreen'))
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=strong_bearish_signals['PriceDate'], 
            y=strong_bearish_signals['Price'], 
            mode='markers',
            name='Strong Bearish Signal (stoch_score < -0.50)',
            marker=dict(color='magenta', size=10, symbol='star', line=dict(width=1, color='darkred'))
        ), row=1, col=1)

    # Only add RSI if requested
    if show_rsi:
        rsi_row = 2
        
        # Add RSI indicator
        fig.add_trace(go.Scatter(
            x=df['PriceDate'], 
            y=df['RSI'], 
            name='RSI', 
            line=dict(color='purple', width=1)
        ), row=rsi_row, col=1)
        
        fig.add_trace(go.Scatter(
            x=df['PriceDate'], 
            y=[70] * len(df), 
            name='Overbought Line', 
            line=dict(color='red', width=1, dash='dash')
        ), row=rsi_row, col=1)
        
        fig.add_trace(go.Scatter(
            x=df['PriceDate'], 
            y=[30] * len(df), 
            name='Oversold Line', 
            line=dict(color='green', width=1, dash='dash')
        ), row=rsi_row, col=1)
        
        # Add annotations for RSI thresholds with better positioning
        fig.add_annotation(
            x=df['PriceDate'].iloc[0], 
            y=70, 
            text="Overbought (70)", 
            showarrow=False, 
            xshift=50, 
            xanchor="left",
            align="left", 
            font=dict(size=9, color="red"),
            row=rsi_row, 
            col=1
        )
        
        fig.add_annotation(
            x=df['PriceDate'].iloc[0], 
            y=30, 
            text="Oversold (30)", 
            showarrow=False, 
            xshift=50, 
            xanchor="left",
            align="left",
            font=dict(size=9, color="green"), 
            row=rsi_row, 
            col=1
        )
        
        # Set RSI y-axis
        fig.update_yaxes(title_text="RSI", row=rsi_row, col=1, gridcolor='rgba(200, 200, 220, 0.3)', range=[0, 100], title_font=dict(size=11))

    # Only add MACD if requested
    if show_macd:
        macd_row = 2 if not show_rsi else 3
        
        # Add MACD indicators
        fig.add_trace(go.Scatter(
            x=df['PriceDate'], 
            y=df['MACD'], 
            name='MACD', 
            line=dict(color='blue', width=1)
        ), row=macd_row, col=1)
        
        fig.add_trace(go.Scatter(
            x=df['PriceDate'], 
            y=df['MACD_Signal'], 
            name='MACD Signal', 
            line=dict(color='red', width=1)
        ), row=macd_row, col=1)

        # Add MACD histogram
        colors = ['green' if val >= 0 else 'red' for val in df['MACD_Hist']]
        fig.add_trace(go.Bar(
            x=df['PriceDate'], 
            y=df['MACD_Hist'], 
            name='MACD Histogram', 
            marker_color=colors,
            opacity=0.5
        ), row=macd_row, col=1)
        
        # Set MACD y-axis
        fig.update_yaxes(title_text="MACD", row=macd_row, col=1, gridcolor='rgba(200, 200, 220, 0.3)', title_font=dict(size=11))

    # Update layout with range sliders and buttons - removing fixed width/height
    fig.update_layout(
        title={
            'text': 'Interactive Stock Chart with Technical Indicators',
            'y':0.98,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(size=16)
        },
        autosize=True,  # Enable autosize for better responsiveness
        legend={
            'orientation': 'h',  # Horizontal orientation
            'yanchor': 'top',    # Anchor to top of the legend box
            'y': -0.15,          # Position below the chart
            'xanchor': 'center', 
            'x': 0.5,            # Center horizontally
            'itemsizing': 'constant',
            'bgcolor': 'rgba(255, 255, 255, 0.7)',
            'bordercolor': 'rgba(0, 0, 0, 0.2)',
            'borderwidth': 1
        },
        margin=dict(l=40, r=40, t=120, b=80),  # Increased top margin to prevent overlap
        xaxis=dict(
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all", label="All")
                ]),
                bgcolor='rgba(150, 200, 250, 0.4)',
                x=1,
                xanchor='right',
                y=1.15,  # Move buttons higher up
                font=dict(size=9)
            ),
            rangeslider=dict(visible=True, thickness=0.03),
            type="date"
        ),
        plot_bgcolor='rgba(240, 244, 250, 0.8)'
    )

    # Update y-axes labels and formats for price chart
    fig.update_yaxes(title_text="Price", row=1, col=1, gridcolor='rgba(200, 200, 220, 0.3)', title_font=dict(size=11))
    
    # Update x-axes formats
    num_rows = 1 + int(show_rsi) + int(show_macd)
    for i in range(1, num_rows + 1):
        fig.update_xaxes(gridcolor='rgba(200, 200, 220, 0.3)', row=i, col=1)
    
    # Only show x-axis title on the bottom subplot
    bottom_row = 1
    if show_rsi:
        bottom_row = 2
    if show_macd:
        bottom_row = 3 if show_rsi else 2
    
    fig.update_xaxes(title_text="Date", row=bottom_row, col=1, title_font=dict(size=11))

    # Enable all interactions with improved hover positioning
    fig.update_layout(
        dragmode='zoom',
        hovermode='x unified',
        hoverdistance=100,
        spikedistance=1000,
        hoverlabel=dict(
            bgcolor="white",
            font_size=12,
            font_family="Segoe UI"
        ),
    )
    
    # Add spikes for better hover experience on all subplots
    for i in range(1, num_rows + 1):
        fig.update_xaxes(
            showspikes=True,
            spikesnap='cursor',
            spikemode='across',
            spikethickness=1,
            row=i, col=1
        )
        
    # Move hover tooltips below cursor instead of centered on cursor
    fig.update_layout(
        hoverlabel_align='left',
        hovermode='closest'
    )

    # For web embedding, adjust the figure config
    config = {
        'scrollZoom': True,
        'displayModeBar': True,
        'modeBarButtonsToAdd': ['drawline', 'drawopenpath', 'eraseshape'],
        'displaylogo': False,
        'responsive': True
    }
    
    # Save the figure as HTML if requested
    html_file = None
    if save_path is not None:
        if not os.path.exists(save_path):
            save_path = os.getcwd()
        
        html_file = os.path.join(save_path, "interactive_chart.html")
        pio.write_html(fig, file=html_file, auto_open=auto_open, config=config)
    
    return fig, html_file


def print_indicator_stats(df):
    """
    Print statistics about the technical indicators
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing calculated indicators
    """
    print("\nIndicator Statistics:")
    print(f"Number of Oversold signals: {df['Oversold'].sum() if 'Oversold' in df.columns else 0}")
    print(f"Number of Undersold signals: {df['Undersold'].sum() if 'Undersold' in df.columns else 0}")
    print(f"Number of Upper Band signals: {df['Upper Band'].sum() if 'Upper Band' in df.columns else 0}")
    print(f"Number of Lower Band signals: {df['Lower Band'].sum() if 'Lower Band' in df.columns else 0}")
    
    # Handle date objects safely - check if PriceDate exists and convert safely
    if 'PriceDate' in df.columns and not df.empty:
        try:
            # First ensure PriceDate is a datetime object
            if pd.api.types.is_datetime64_any_dtype(df['PriceDate']):
                start_date = df['PriceDate'].min().strftime('%Y-%m-%d')
                end_date = df['PriceDate'].max().strftime('%Y-%m-%d')
            else:
                # Try to convert strings to datetime
                start_date = pd.to_datetime(df['PriceDate'].min()).strftime('%Y-%m-%d')
                end_date = pd.to_datetime(df['PriceDate'].max()).strftime('%Y-%m-%d')
            
            print(f"Analysis period: {start_date} to {end_date}")
        except Exception as e:
            print(f"Could not format date range: {str(e)}")
            print(f"Analysis period: {df['PriceDate'].min()} to {df['PriceDate'].max()}")
    else:
        print("Analysis period: Not available (PriceDate column missing)")