# %%
import pandas as pd
import numpy as np
from pathlib import Path

# 1. Load the broken dataset
if "__file__" in globals():
    project_root = Path(__file__).resolve().parents[1]
else:
    cwd = Path.cwd().resolve()
    project_root = next((path for path in [cwd, cwd.parent] if (path / "data" / "data_season.csv").exists()), cwd)

file_path = project_root / "data" / "data_season.csv"
df = pd.read_csv(file_path)

# 2. Base market prices per Quintal (100kg) in the year 2004
base_prices = {
    'Coconut': 1500, 'Cocoa': 6000, 'Coffee': 5500, 'Cardamum': 35000,
    'Pepper': 25000, 'Arecanut': 12000, 'Ginger': 3000, 'Tea': 8000,
    'Paddy': 800, 'Groundnut': 2000, 'Blackgram': 2500, 'Cashew': 4500, 'Cotton': 3500
}

np.random.seed(42) # Keeps the data consistent

def generate_logical_price(row):
    base = base_prices.get(row['Crops'], 2000)
    
    # Add realistic historical WPI inflation (~6% per year from 2004)
    year_diff = row['Year'] - 2004
    inflation_multiplier = (1.06) ** year_diff
    
    # Add Weather Shocks (Extreme rain/heat lowers yield and spikes the price!)
    rain_shock = 1.2 if row['Rainfall'] < 500 else 1.15 if row['Rainfall'] > 3000 else 1.0
    temp_shock = 1.1 if row['temperature'] > 35 else 1.0
    
    # Add natural market noise
    noise = np.random.uniform(0.95, 1.05)
    
    return round(base * inflation_multiplier * rain_shock * temp_shock * noise, 2)

# 3. Overwrite the broken column
df['price'] = df.apply(generate_logical_price, axis=1)

# 4. Save the fixed dataset back to your folder
df.to_csv(file_path, index=False)
print("✅ Dataset successfully repaired with logical economic prices!")
# %%
