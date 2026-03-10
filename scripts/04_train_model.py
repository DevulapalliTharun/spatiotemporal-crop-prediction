# %%
import pandas as pd
import numpy as np
import os
import joblib 
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

# Resolve paths from the project root so the script works from both Jupyter and CLI.
if "__file__" in globals():
    project_root = Path(__file__).resolve().parents[1]
else:
    cwd = Path.cwd().resolve()
    project_root = next(
        (path for path in [cwd, cwd.parent] if (path / "data" / "data_season.csv").exists()),
        cwd,
    )

df = pd.read_csv(project_root / "data" / "data_season.csv").dropna()
df = df.rename(columns={'temperature': 'Temperature', 'price': 'Price'})

# Notice 'Year' is back in! The model can finally learn the timeline.
categorical_cols = ['Location', 'Season', 'Crops']
numerical_cols = ['Year', 'Temperature', 'Humidity', 'Rainfall']
target_col = 'Price'

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

X = df[categorical_cols + numerical_cols]
y_log = np.log1p(df[target_col]) # Log transform for price gaps

X_train, X_test, y_train, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)

print("\nTraining Final XGBoost Engine...")
model = XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=6, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

models_dir = project_root / "models"
os.makedirs(models_dir, exist_ok=True)
joblib.dump(model, models_dir / "xgboost_base.pkl")
joblib.dump(encoders, models_dir / "label_encoders.pkl")
print("\n✅ Final Model saved!")
# %%
