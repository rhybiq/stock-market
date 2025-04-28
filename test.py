import pandas as pd
import Technical_Indicators as ti
from Data import prepare_data

# Load your stock data
df = pd.read_csv('stock_data/TCS.csv')

df = prepare_data(df)
# Calculate basic and advanced indicators
print(df.head())  # Print just the first few rows

# Save the DataFrame to a CSV file
output_file = 'processed_TCS.csv'
df.to_csv(output_file, index=False)
print(f"Data saved to {output_file}")
