# %%
# Cell 1: Generate Yearly Inflation Multipliers (2019 to 2026)
import pandas as pd
import os

# The exact Financial Year WPI data you provided (Base 2011-12=100)
# We map them to the calendar years for your dataset's timeline
wpi_yearly = {
    2019: 121.8, # Base year (End of your dataset)
    2020: 123.4,
    2021: 139.4,
    2022: 152.5,
    2023: 151.4,
    2024: 154.9,
    2025: 154.9, # Using the latest available index for the current timeline
    2026: 154.9  # Baseline projection for early 2026
}

base_wpi = wpi_yearly[2019]

processed_data = []

print("Calculating year-by-year mathematical scaling factors...")

for year, wpi in wpi_yearly.items():
    multiplier = wpi / base_wpi
    processed_data.append({
        'Year': year,
        'Official_WPI': wpi,
        'Inflation_Multiplier': round(multiplier, 4)
    })

# Convert to DataFrame
df_yearly_inflation = pd.DataFrame(processed_data)

# Ensure the directory exists
os.makedirs('data/macro', exist_ok=True)

# Save the yearly multipliers
output_path = 'data/macro/yearly_inflation.csv'
df_yearly_inflation.to_csv(output_path, index=False)

print(f"\nSuccess! Yearly inflation data saved to {output_path}")
print("\nHere is your Year-by-Year trajectory:")
print(df_yearly_inflation)
# %%
