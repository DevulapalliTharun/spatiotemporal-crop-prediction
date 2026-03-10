# Spatiotemporal Crop Price Forecasting

Streamlit application and end-to-end data pipeline for forecasting agricultural crop prices using historical market data, seasonal weather behavior, inflation-aware logic, and an XGBoost regression model.

## Quick Start

Run these commands first if you just want the app without reading the full documentation.

```bash
git clone https://github.com/DevulapalliTharun/spatiotemporal-crop-prediction.git
cd spatiotemporal-crop-prediction
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL shown by Streamlit, usually:

```text
http://localhost:8501
```

## Full Pipeline Commands

Use this sequence if you want to regenerate the entire project from scratch before launching the app.

```bash
git clone https://github.com/DevulapalliTharun/spatiotemporal-crop-prediction.git
cd spatiotemporal-crop-prediction
python -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
python scripts/00_fix_dataset.py
cd scripts
python 01_fetch_weather.py
python 02_process_inflation.py
python 03_process_weather.py
cd ..
python scripts/04_train_model.py
python scripts/05_evaluate_model.py
streamlit run app.py
```

Important path note:
The current `app.py` reads processed future weather from `scripts/data/processed/seasonal_weather_2019_2026.csv`. That is why `01_fetch_weather.py`, `02_process_inflation.py`, and `03_process_weather.py` are shown above from inside the `scripts/` directory.

## What This Project Does

This project forecasts crop prices by combining:

- historical crop market records
- district-level weather variables
- spatiotemporal seasonal structure
- inflation-aware price reconstruction
- machine learning using XGBoost
- an interactive Streamlit dashboard for inference and visualization

The application allows a user to:

- choose a crop
- choose a district
- see the historical price trajectory
- see predicted future prices for upcoming seasons
- inspect the latest predicted seasonal table

## Project Objectives

The main purpose of the project is to build a mathematically defensible price forecasting workflow rather than a simple static dashboard. The design integrates time, geography, and climate signals into one pipeline:

- time is modeled using `Year` and season blocks
- geography is modeled using district names
- climate is modeled using temperature, humidity, and rainfall
- price scale is stabilized using a log-transformed target

## Repository Structure

```text
spatiotemporal-crop-prediction/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── data_season.csv
├── models/
│   ├── xgboost_base.pkl
│   ├── random_forest_base.pkl
│   └── label_encoders.pkl
├── scripts/
│   ├── 00_fix_dataset.py
│   ├── 01_fetch_weather.py
│   ├── 02_process_inflation.py
│   ├── 03_process_weather.py
│   ├── 04_train_model.py
│   ├── 05_evaluate_model.py
│   └── data/
│       ├── raw/weather_2019_2026.csv
│       ├── processed/seasonal_weather_2019_2026.csv
│       └── macro/yearly_inflation.csv
├── visualizations/
│   ├── actual_vs_predicted.png
│   ├── evaluation_metrics.txt
│   └── feature_importance.png
└── extra/
    └── inflation.png
```

## Technology Stack

- Python
- Streamlit
- Pandas
- NumPy
- Plotly
- Scikit-learn
- XGBoost
- Joblib
- NASA POWER API

## How To Use The App

After running:

```bash
streamlit run app.py
```

use the sidebar in this order:

1. Select a crop.
2. Select a district where that crop exists in the historical dataset.
3. View the combined chart of historical prices and future predicted prices.
4. Inspect the projected forecast table for the most recent seasons.

The chart shows:

- blue line for historical prices
- red dashed line for predicted future prices
- a vertical boundary at `2019 Zaid`, marking the start of NASA-weather-driven future inference

## Pipeline Overview

The project is organized into six major phases.

1. Dataset repair and logical price generation
2. Daily satellite weather collection
3. Inflation multiplier preparation
4. Seasonal weather aggregation
5. Machine learning training and evaluation
6. Streamlit inference and visualization

## Phase 0: Dataset Repair and Price Reconstruction

Implemented in [scripts/00_fix_dataset.py](/home/devulapalli-tharun/Documents/Mini Project/scripts/00_fix_dataset.py).

The historical dataset originally required correction in the `price` field. This script reconstructs economically plausible crop prices using a deterministic mathematical recipe plus bounded noise.

### Base Price Logic

Each crop is assigned a base price per quintal in the year 2004:

- Coconut: 1500
- Cocoa: 6000
- Coffee: 5500
- Cardamum: 35000
- Pepper: 25000
- Arecanut: 12000
- Ginger: 3000
- Tea: 8000
- Paddy: 800
- Groundnut: 2000
- Blackgram: 2500
- Cashew: 4500
- Cotton: 3500

### Mathematical Price Formula

For each row, the repaired price is generated as:

```text
Price = BaseCropPrice
        × InflationMultiplier
        × RainShock
        × TempShock
        × Noise
```

where:

```text
InflationMultiplier = (1.06) ^ (Year - 2004)
```

This assumes an approximate 6% yearly compounding inflation trend from 2004 onward.

### Weather Shock Logic

The script encodes simple supply-side economic shocks:

- if `Rainfall < 500`, then `RainShock = 1.20`
- if `Rainfall > 3000`, then `RainShock = 1.15`
- otherwise, `RainShock = 1.00`
- if `temperature > 35`, then `TempShock = 1.10`
- otherwise, `TempShock = 1.00`

Interpretation:

- extremely low rainfall can reduce output and increase prices
- extremely high rainfall can also damage output and increase prices
- high temperature can further stress crops and push prices upward

### Controlled Noise

To avoid a perfectly deterministic synthetic price column, the script multiplies by:

```text
Noise ~ Uniform(0.95, 1.05)
```

with `np.random.seed(42)` for reproducibility.

### Output

The repaired `price` column is written back into:

```text
data/data_season.csv
```

## Phase 1: NASA POWER Daily Weather Ingestion

Implemented in [scripts/01_fetch_weather.py](/home/devulapalli-tharun/Documents/Mini Project/scripts/01_fetch_weather.py).

This stage fetches daily satellite-derived weather data from the NASA POWER API for 11 districts.

### Locations Covered

- Mangalore
- Kodagu
- Kasaragodu
- Raichur
- Hassan
- Udupi
- Chikmagalur
- Shimoga
- Uttara Kannada
- Davanagere
- Mysore

### Variables Pulled

The script requests the following daily variables:

- `T2M` -> Temperature
- `PRECTOTCORR` -> Rainfall
- `RH2M` -> Humidity

### Temporal Coverage

The current script requests data from:

- start date: `2019-01-01`
- end date: `2026-02-28`

This range bridges the historical dataset into future seasonal inference.

### Data Engineering Details

For each location:

- latitude and longitude are supplied to NASA POWER
- JSON response is converted to a DataFrame
- date index is parsed from `YYYYMMDD`
- columns are renamed into domain-friendly names
- location name is attached to each row

### Output

The combined raw weather dataset is stored at:

```text
scripts/data/raw/weather_2019_2026.csv
```

## Phase 2: Inflation Multiplier Construction

Implemented in [scripts/02_process_inflation.py](/home/devulapalli-tharun/Documents/Mini Project/scripts/02_process_inflation.py).

This script converts yearly WPI values into normalized inflation multipliers.

### WPI Values Used

- 2019: 121.8
- 2020: 123.4
- 2021: 139.4
- 2022: 152.5
- 2023: 151.4
- 2024: 154.9
- 2025: 154.9
- 2026: 154.9

### Mathematical Logic

The 2019 WPI is used as the base:

```text
Inflation_Multiplier(year) = WPI(year) / WPI(2019)
```

This produces a dimensionless scaling factor that can be used to compare nominal price levels against the 2019 baseline.

### Output

The generated macroeconomic table is saved at:

```text
scripts/data/macro/yearly_inflation.csv
```

## Phase 3: Daily-To-Seasonal Spatiotemporal Aggregation

Implemented in [scripts/03_process_weather.py](/home/devulapalli-tharun/Documents/Mini Project/scripts/03_process_weather.py).

This stage compresses daily weather into agricultural season blocks, which is a key mathematical step in the pipeline.

### Season Definition

Months are mapped into Indian agricultural seasons:

- Kharif: July, August, September, October
- Rabi: November, December, January, February
- Zaid: March, April, May, June

### Agricultural Year Correction

Rabi crosses the calendar year boundary. To preserve agricultural cycle consistency:

- January and February are reassigned to the previous `Year`

This means `Jan-Feb 2020` becomes part of `Rabi 2019`, which is mathematically more correct for seasonal crop modeling.

### Aggregation Rules

The script groups by:

- `Location`
- `Year`
- `Season`

and aggregates as:

- `Temperature` -> mean
- `Humidity` -> mean
- `Rainfall` -> sum

This aggregation is important:

- mean temperature represents typical thermal exposure during the season
- mean humidity captures seasonal atmospheric moisture conditions
- total rainfall represents cumulative water availability during the season

### Output

The processed seasonal weather table is written to:

```text
scripts/data/processed/seasonal_weather_2019_2026.csv
```

## Phase 4: Model Training

Implemented in [scripts/04_train_model.py](/home/devulapalli-tharun/Documents/Mini Project/scripts/04_train_model.py).

The model training script builds an XGBoost regressor to learn crop price behavior from spatiotemporal and climate features.

### Input Data

The training script reads:

- historical dataset from `data/data_season.csv`

It renames:

- `temperature` -> `Temperature`
- `price` -> `Price`

### Feature Set

Categorical features:

- `Location`
- `Season`
- `Crops`

Numerical features:

- `Year`
- `Temperature`
- `Humidity`
- `Rainfall`

Target:

- `Price`

### Encoding Logic

Categorical features are converted using `LabelEncoder`.

This produces integer representations for:

- districts
- seasons
- crop names

The fitted encoders are stored so the Streamlit app can transform future inference inputs with the exact same category mapping.

### Target Transformation

Instead of training directly on raw prices, the model uses:

```text
y_log = log(1 + Price)
```

implemented with:

```python
np.log1p(Price)
```

This has important mathematical benefits:

- compresses large price ranges
- reduces the impact of extreme values
- stabilizes variance
- often improves regression performance for skewed price distributions

### Train-Test Split

The dataset is split as:

- 80% training
- 20% testing
- `random_state = 42`

### XGBoost Configuration

The current model is trained with:

```text
n_estimators = 300
learning_rate = 0.05
max_depth = 6
random_state = 42
n_jobs = -1
```

Interpretation:

- more trees improve expressive power
- smaller learning rate improves stability
- depth 6 allows nonlinear interactions
- parallel jobs speed up training

### Outputs

The training stage saves:

- `models/xgboost_base.pkl`
- `models/label_encoders.pkl`

## Phase 5: Evaluation and Diagnostics

Implemented in [scripts/05_evaluate_model.py](/home/devulapalli-tharun/Documents/Mini Project/scripts/05_evaluate_model.py).

This stage loads the trained model and measures predictive quality on the held-out test set.

### Reverse Transformation

Because the model predicts `log(1 + Price)`, the script converts predictions back to rupee scale using:

```text
PredictedPrice = exp(y_pred_log) - 1
```

implemented with:

```python
np.expm1(y_pred_log)
```

This inverse transform is mathematically paired with `log1p`.

### Metrics Used

- R-squared
- MAPE
- Mean Absolute Error
- Root Mean Squared Error

### Current Saved Metrics

From `visualizations/evaluation_metrics.txt`:

- R-Squared: `0.9974`
- MAPE: `3.05%`
- Mean Absolute Error: `₹ 330.15`
- RMSE: `₹ 760.96`

### Diagnostic Artifacts

The evaluation script generates:

- `visualizations/evaluation_metrics.txt`
- `visualizations/feature_importance.png`
- `visualizations/actual_vs_predicted.png`

These help interpret:

- which features drive model predictions
- how close predictions are to actual market prices

## Phase 6: Streamlit Inference Application

Implemented in [app.py](/home/devulapalli-tharun/Documents/Mini Project/app.py).

The Streamlit app is the final user-facing layer of the project.

### App Logic

The application:

1. loads the trained XGBoost model
2. loads the saved label encoders
3. loads historical market data
4. loads processed seasonal future weather
5. allows the user to select crop and district
6. filters historical observations for that pair
7. filters future weather for the same district
8. injects the chosen crop into future records
9. encodes categorical features using saved encoders
10. predicts log prices for future seasons
11. converts predicted log prices back into rupee values
12. plots historical and future trajectories
13. displays the latest forecast table

### Location Name Harmonization

To align naming differences between datasets, the app applies this mapping:

- `Chikmagalur` -> `Chikmangaluru`
- `Mysore` -> `Mysuru`
- `Davanagere` -> `Davangere`

This is necessary because inconsistent district labels would otherwise break joins and inference filtering.

### Inference Math

Future prediction uses the exact trained feature schema:

```text
[Location, Season, Crops, Year, Temperature, Humidity, Rainfall]
```

The model output is:

```text
log(1 + PredictedPrice)
```

and the app converts it using:

```text
PredictedPrice = exp(ModelOutput) - 1
```

### Unseen Label Fallback

Before encoding, the app checks whether each categorical future value exists in the corresponding `LabelEncoder` class set.

If an unseen label appears, it falls back to the first known class. This avoids runtime inference crashes, although the ideal long-term fix is to ensure the training and inference label spaces remain fully aligned.

## Why This Is A Spatiotemporal Model

This project is not only a standard regression model. It is spatiotemporal because it jointly models:

- space: different districts
- time: year and agricultural season
- environment: weather variables changing across both place and time

The model learns interactions such as:

- the same crop behaving differently in different districts
- seasonal rainfall affecting prices differently year to year
- temporal progression affecting baseline market values

## Mathematical Summary

The full modeling logic can be summarized as:

```text
Historical Price Repair:
P_repaired = P_base(crop) × (1.06)^(year-2004) × rain_shock × temp_shock × noise

Inflation Scaling:
InflationMultiplier(year) = WPI(year) / WPI(2019)

Seasonal Aggregation:
Temp_season = mean(daily temperature)
Humidity_season = mean(daily humidity)
Rainfall_season = sum(daily rainfall)

Training Target:
y = log(1 + Price)

Inference Output:
PredictedPrice = exp(model_output) - 1
```

This combination of domain logic plus machine learning is the core idea of the project.

## Files Generated During The Pipeline

Running the full pipeline creates or refreshes:

- `data/data_season.csv`
- `scripts/data/raw/weather_2019_2026.csv`
- `scripts/data/macro/yearly_inflation.csv`
- `scripts/data/processed/seasonal_weather_2019_2026.csv`
- `models/xgboost_base.pkl`
- `models/label_encoders.pkl`
- `visualizations/evaluation_metrics.txt`
- `visualizations/feature_importance.png`
- `visualizations/actual_vs_predicted.png`

## Reproducing Results

Recommended order:

1. Repair the dataset.
2. Fetch NASA weather data.
3. Build yearly inflation multipliers.
4. Aggregate weather into seasons.
5. Train the XGBoost model.
6. Evaluate the trained model.
7. Launch the Streamlit app.

## Deployment

### Deploy On Streamlit Community Cloud

1. Push this repository to GitHub.
2. Open Streamlit Community Cloud.
3. Connect your GitHub account.
4. Create a new app from this repository.
5. Set the main file path to `app.py`.
6. Deploy the app.

### Local Deployment

For local use:

```bash
streamlit run app.py
```

## Basic GitHub Usage

Clone:

```bash
git clone https://github.com/DevulapalliTharun/spatiotemporal-crop-prediction.git
cd spatiotemporal-crop-prediction
```

Update later:

```bash
git pull
```

## Troubleshooting

### Streamlit says missing `ScriptRunContext`

This happens if you run the app with plain Python or from a notebook kernel. Use:

```bash
streamlit run app.py
```

### App cannot find files

Make sure these files exist:

- `models/xgboost_base.pkl`
- `models/label_encoders.pkl`
- `data/data_season.csv`
- `scripts/data/processed/seasonal_weather_2019_2026.csv`

### Dependency installation fails

Upgrade pip first:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Future Improvements

- add explicit inflation multiplier integration into the final training features
- improve unseen-category handling instead of fallback-to-first-class behavior
- replace label encoding with safer categorical encodings where appropriate
- add automated tests for the pipeline
- standardize all path handling so every script runs identically from any working directory
- add model versioning and experiment tracking

## Author

Devulapalli Tharun

## License

Add a license file if you plan to distribute or open-source the project publicly.
