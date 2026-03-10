# %%
# Cell 1: Convert Daily NASA Weather into Seasonal Agricultural Blocks
import pandas as pd
import os

# 1. Load the raw daily weather data you just downloaded
file_path = 'data/raw/weather_2019_2026.csv'
print(f"Loading daily weather data from {file_path}...")
df_daily = pd.read_csv(file_path)

# Ensure Date is a datetime object
df_daily['Date'] = pd.to_datetime(df_daily['Date'])

# Extract Year and Month
df_daily['Year'] = df_daily['Date'].dt.year
df_daily['Month'] = df_daily['Date'].dt.month

# 2. Define the Agricultural Seasons based on Indian standards
def get_season(month):
    if month in [7, 8, 9, 10]:
        return 'Kharif'       # Monsoon season
    elif month in [11, 12, 1, 2]:
        return 'Rabi'         # Winter season
    elif month in [3, 4, 5, 6]:
        return 'Zaid'         # Summer season
    else:
        return 'Unknown'

# Apply the function to create a new 'Season' column
df_daily['Season'] = df_daily['Month'].apply(get_season)

# Note: Since Rabi spans across years (Nov-Feb), January and February 
# technically belong to the previous year's agricultural cycle. 
# For strict matching with standard datasets, we adjust the year for Jan/Feb.
df_daily.loc[df_daily['Month'].isin([1, 2]), 'Year'] -= 1

# 3. Aggregate the data (The Spatiotemporal Math)
print("Aggregating daily data into seasonal blocks...")

# We want the MEAN temperature and humidity, but the TOTAL (sum) rainfall for the season
seasonal_weather = df_daily.groupby(['Location', 'Year', 'Season']).agg({
    'Temperature': 'mean',
    'Humidity': 'mean',
    'Rainfall': 'sum'  # Total cumulative rainfall is what matters for crops
}).reset_index()

# Round the numbers to make them clean
seasonal_weather['Temperature'] = seasonal_weather['Temperature'].round(2)
seasonal_weather['Humidity'] = seasonal_weather['Humidity'].round(2)
seasonal_weather['Rainfall'] = seasonal_weather['Rainfall'].round(2)

# 4. Save the processed seasonal data
# Ensure the directory exists
os.makedirs('data/processed', exist_ok=True)

output_path = 'data/processed/seasonal_weather_2019_2026.csv'
seasonal_weather.to_csv(output_path, index=False)

print(f"\nSuccess! Daily data successfully compressed into seasons.")
print(f"Data saved to {output_path}")
print("\nPreview of the mathematically aligned weather data:")
print(seasonal_weather.head(10))
# %%
