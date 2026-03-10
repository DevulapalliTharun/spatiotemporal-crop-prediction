# %%
# Cell 1: Automate Satellite Weather Data Ingestion (NASA POWER API)
import pandas as pd
import requests
import time
import os

# 1. GPS Coordinates for your 11 locations
# Replace/edit any of these if your dataset uses different district names!
locations = {
    'Mangalore': {'lat': 12.9141, 'lon': 74.8560},
    'Kodagu': {'lat': 12.4244, 'lon': 75.7382},
    'Kasaragodu': {'lat': 12.4968, 'lon': 74.9869},
    'Raichur': {'lat': 16.2076, 'lon': 77.3463},
    'Hassan': {'lat': 13.0072, 'lon': 76.1016},
    'Udupi': {'lat': 13.3409, 'lon': 74.7421},
    'Chikmagalur': {'lat': 13.3161, 'lon': 75.7720},
    'Shimoga': {'lat': 13.9299, 'lon': 75.5681},
    'Uttara Kannada': {'lat': 14.7310, 'lon': 74.6509},
    'Davanagere': {'lat': 14.4644, 'lon': 75.9218},
    'Mysore': {'lat': 12.2958, 'lon': 76.6394}
}

# 2. Timeframe: The gap from your dataset's end to the present
# NASA POWER takes dates in YYYYMMDD format
start_date = "20190101"
end_date = "20260228" # Up to recent days in 2026

# 3. Initialize an empty list to store all the dataframes
all_weather_data = []

print("Initializing NASA POWER API Connection...")
print(f"Fetching daily weather from {start_date} to {end_date}\n")

# 4. Loop through each location and pull the data
for loc_name, coords in locations.items():
    print(f"Fetching satellite data for {loc_name} (Lat: {coords['lat']}, Lon: {coords['lon']})...")
    
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    
    # Parameters: T2M (Temp), PRECTOTCORR (Rainfall), RH2M (Relative Humidity)
    params = {
        "parameters": "T2M,PRECTOTCORR,RH2M",
        "community": "AG", # Agricultural community
        "longitude": coords['lon'],
        "latitude": coords['lat'],
        "start": start_date,
        "end": end_date,
        "format": "JSON"
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status() # Check for HTTP errors
        
        # Parse JSON response
        data = response.json()['properties']['parameter']
        
        # Convert to Pandas DataFrame
        df_loc = pd.DataFrame(data)
        
        # Clean up the index (which contains the dates)
        df_loc.index = pd.to_datetime(df_loc.index, format='%Y%m%d')
        df_loc.index.name = 'Date'
        
        # Rename columns to match your base dataset style
        df_loc.rename(columns={
            'T2M': 'Temperature', 
            'PRECTOTCORR': 'Rainfall', 
            'RH2M': 'Humidity'
        }, inplace=True)
        
        # Add a column for the location so we can merge it later
        df_loc['Location'] = loc_name
        
        # Reset index to make Date a standard column
        df_loc.reset_index(inplace=True)
        
        # Append to our master list
        all_weather_data.append(df_loc)
        
        print(f"   -> Success! Fetched {len(df_loc)} days of data.")
        
    except Exception as e:
        print(f"   -> Error fetching data for {loc_name}: {e}")
    
    # Pause for 1 second between requests to avoid overloading the NASA API
    time.sleep(1)

# 5. Combine all locations into one massive DataFrame
print("\nCombining all location data...")
final_weather_df = pd.concat(all_weather_data, ignore_index=True)

# 6. Save the raw downloaded data
# Ensure the directory exists
os.makedirs('data/raw', exist_ok=True)

output_path = 'data/raw/weather_2019_2026.csv'
final_weather_df.to_csv(output_path, index=False)

print(f"\nPipeline Complete! 🛰️")
print(f"Total rows fetched: {len(final_weather_df)}")
print(f"Data saved to {output_path}")
print(final_weather_df.head())
# %%
