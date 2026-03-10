import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx

PROJECT_ROOT = Path(__file__).resolve().parent


def load_assets():
    import joblib

    model = joblib.load(PROJECT_ROOT / "models" / "xgboost_base.pkl")
    encoders = joblib.load(PROJECT_ROOT / "models" / "label_encoders.pkl")
    historical_data = pd.read_csv(PROJECT_ROOT / "data" / "data_season.csv").dropna()
    future_weather = pd.read_csv(PROJECT_ROOT / "scripts" / "data" / "processed" / "seasonal_weather_2019_2026.csv")

    historical_data = historical_data.rename(columns={"price": "Price", "temperature": "Temperature"})
    return model, encoders, historical_data, future_weather


def ensure_streamlit_context():
    if get_script_run_ctx(suppress_warning=True) is None:
        raise SystemExit(
            "This app must be started with Streamlit.\n"
            "Run: streamlit run app.py"
        )


def main():
    ensure_streamlit_context()

    st.set_page_config(page_title="Spatiotemporal Crop Price Forecaster", layout="wide")
    st.title("🌾 Spatiotemporal Crop Price Forecasting (2004 - 2026)")
    st.markdown("Mathematical modeling using XGBoost and NASA Agroclimatology data.")

    cached_load_assets = st.cache_resource(load_assets)
    model, encoders, historical_data, future_weather = cached_load_assets()

    location_mapping = {"Chikmagalur": "Chikmangaluru", "Mysore": "Mysuru", "Davanagere": "Davangere"}
    future_weather["Location"] = future_weather["Location"].replace(location_mapping)

    st.sidebar.header("Decision Support System")
    selected_crop = st.sidebar.selectbox("Select Crop", sorted(historical_data["Crops"].unique()))

    valid_locations = historical_data[historical_data["Crops"] == selected_crop]["Location"].unique()
    selected_location = st.sidebar.selectbox("Select District", sorted(valid_locations))

    st.subheader(f"Price Trajectory for {selected_crop} in {selected_location}")

    history_subset = historical_data[
        (historical_data["Crops"] == selected_crop) & (historical_data["Location"] == selected_location)
    ].copy()
    history_subset = history_subset.sort_values(by=["Year", "Season"])

    future_subset = future_weather[future_weather["Location"] == selected_location].copy()
    future_subset["Crops"] = selected_crop

    if future_subset.empty:
        st.warning("We don't have enough spatial weather data to predict this combination yet.")
        return

    x_future = future_subset.copy()
    for col in ["Location", "Season", "Crops"]:
        if col in encoders:
            classes = encoders[col].classes_
            x_future[col] = x_future[col].apply(lambda value: value if value in classes else classes[0])
            x_future[col] = encoders[col].transform(x_future[col])

    features = ["Location", "Season", "Crops", "Year", "Temperature", "Humidity", "Rainfall"]
    log_predictions = model.predict(x_future[features])
    future_subset["Predicted_Price"] = np.expm1(log_predictions)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history_subset["Year"].astype(str) + " " + history_subset["Season"],
            y=history_subset["Price"],
            mode="lines+markers",
            name="Historical Price (₹)",
            line=dict(color="royalblue", width=2),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=future_subset["Year"].astype(str) + " " + future_subset["Season"],
            y=future_subset["Predicted_Price"],
            mode="lines+markers",
            name="NASA Weather Predicted Price (₹)",
            line=dict(color="firebrick", width=3, dash="dash"),
        )
    )
    fig.add_vline(x="2019 Zaid", line_width=2, line_dash="dash", line_color="green")
    fig.add_annotation(
        x="2019 Zaid",
        y=history_subset["Price"].max(),
        text="Prediction Boundary (Start of NASA Data)",
        showarrow=False,
        xshift=130,
    )
    fig.update_layout(
        xaxis_title="Timeline (Year & Season)",
        yaxis_title="Market Price per 100kg (₹)",
        hovermode="x unified",
        template="plotly_white",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Projected Forecast (Upcoming Seasons)")
    display_cols = ["Year", "Season", "Temperature", "Rainfall", "Predicted_Price"]
    st.dataframe(
        future_subset[display_cols].tail(5).style.format(
            {
                "Temperature": "{:.2f} °C",
                "Rainfall": "{:.2f} mm",
                "Predicted_Price": "₹ {:,.2f}",
            }
        )
    )


if __name__ == "__main__":
    main()
