# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error

# Resolve paths from the project root so the script works from both Jupyter and CLI.
if "__file__" in globals():
    project_root = Path(__file__).resolve().parents[1]
else:
    cwd = Path.cwd().resolve()
    project_root = next(
        (path for path in [cwd, cwd.parent] if (path / "data" / "data_season.csv").exists()),
        cwd,
    )

viz_dir = project_root / "visualizations"
os.makedirs(viz_dir, exist_ok=True)

df = pd.read_csv(project_root / "data" / "data_season.csv").dropna()
df = df.rename(columns={'temperature': 'Temperature', 'price': 'Price'})

model = joblib.load(project_root / "models" / "xgboost_base.pkl")
encoders = joblib.load(project_root / "models" / "label_encoders.pkl")

categorical_cols = ['Location', 'Season', 'Crops']
numerical_cols = ['Year', 'Temperature', 'Humidity', 'Rainfall']
target_col = 'Price'

for col in categorical_cols:
    df[col] = encoders[col].transform(df[col])

X = df[categorical_cols + numerical_cols]
y_log = np.log1p(df[target_col])

X_train, X_test, y_train, y_test_log = train_test_split(X, y_log, test_size=0.2, random_state=42)

y_pred_log = model.predict(X_test)
y_test_real = np.expm1(y_test_log)
y_pred_real = np.expm1(y_pred_log)

r2 = r2_score(y_test_real, y_pred_real)
mape = mean_absolute_percentage_error(y_test_real, y_pred_real) * 100
mae = mean_absolute_error(y_test_real, y_pred_real)
rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))

metrics_report = f"""===================================
🏆 FINAL MODEL METRICS 🏆
===================================
1. R-Squared (R²):     {r2:.4f}  
2. MAPE:               {mape:.2f}%   
3. Mean Abs Error:     ₹ {mae:,.2f}
4. Root Mean Sq Error: ₹ {rmse:,.2f}
==================================="""
print("\n" + metrics_report + "\n")

with open(viz_dir / "evaluation_metrics.txt", "w", encoding="utf-8") as f:
    f.write(metrics_report)

plt.figure(figsize=(10, 6))
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
features = X.columns
sns.barplot(x=importances[indices], y=[features[i] for i in indices], hue=[features[i] for i in indices], palette="viridis", legend=False)
plt.title("Feature Importance - Spatiotemporal Price Drivers", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(viz_dir / "feature_importance.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 6))
plt.scatter(y_test_real, y_pred_real, alpha=0.6, color='dodgerblue', edgecolor='k', s=50)
plt.plot([y_test_real.min(), y_test_real.max()], [y_test_real.min(), y_test_real.max()], 'r--', lw=3, label="Perfect Accuracy")
plt.title("Model Accuracy: Actual vs Predicted Crop Prices", fontsize=14, fontweight='bold')
plt.xlabel("Actual Prices (₹)", fontsize=12)
plt.ylabel("Predicted Prices (₹)", fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig(viz_dir / "actual_vs_predicted.png", dpi=300)
plt.close()

print(f"✅ Success! Graphs and metrics saved.")

# %%
