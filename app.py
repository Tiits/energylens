import streamlit as st
import pandas as pd

from src.utils.data_loader import load_heapo_raw, preprocess_heapo
from src.models.forecast import fit_predict_prophet
from src.viz.plots import plot_history, plot_forecast

# Application title
st.set_page_config(page_title="EnergyLens", layout="wide")
st.title("EnergyLens: Regional Electricity Consumption Forecast")

# Sidebar configuration
st.sidebar.header("Configuration")

# Data path
data_path = st.sidebar.text_input(
    "HEAPO archive path", "data/raw/heapo_data.zip"
)

# Loading data
with st.spinner("Loading and preprocessing data..."):
    try:
        df_raw = load_heapo_raw(data_path)
        df = preprocess_heapo(df_raw)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

# Determine overall available period
date_min = df['ds'].dt.date.min()
date_max = df['ds'].dt.date.max()

# Select analysis period
dates = st.sidebar.date_input(
    "Select date range", [date_min, date_max],
    min_value=date_min, max_value=date_max
)
if len(dates) != 2:
    st.sidebar.error("Please select start and end dates.")
    st.stop()
start_date, end_date = dates

# Select household
households = sorted(df['household'].unique())
selected_household = st.sidebar.selectbox(
    "Select household ID", households
)

# Filter data for the selected period and household
mask = (
    (df['ds'].dt.date >= start_date) &
    (df['ds'].dt.date <= end_date) &
    (df['household'] == selected_household)
)
df_selected = df.loc[mask]
if df_selected.empty:
    st.warning("No data for chosen filters. Please adjust period or household.")
    st.stop()

# Display historical series
st.subheader(f"Historical Consumption for Household {selected_household}")
fig_hist = plot_history(df_selected, title="Historical Electricity Consumption")
st.plotly_chart(fig_hist, use_container_width=True)

# Forecast parameters
st.sidebar.header("Forecast Parameters")
horizon_days = st.sidebar.slider(
    "Forecast horizon (days)", min_value=1, max_value=30, value=7
)

# Run button
if st.sidebar.button("Run Forecast"):
    periods = horizon_days * 96  # 96 intervals per day at 15-min frequency
    with st.spinner("Training model and predicting..."):
        model, forecast = fit_predict_prophet(
            df_selected[['ds', 'y']],
            periods=periods,
            freq='15min',
            weekly_seasonality=True,
            daily_seasonality=False,
            yearly_seasonality=True,
            changepoint_prior_scale=0.05
        )
    # Display forecasts
    st.subheader("Forecasted Consumption")
    fig_fc = plot_forecast(model, forecast)
    st.plotly_chart(fig_fc, use_container_width=True)

    # Display forecast data
    st.subheader("Forecast Data (last 10 entries)")
    df_fc_tail = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(10)
    df_fc_tail = df_fc_tail.rename(
        columns={
            'ds': 'Date',
            'yhat': 'Prediction',
            'yhat_lower': 'Lower CI',
            'yhat_upper': 'Upper CI'
        }
    )
    st.dataframe(df_fc_tail)

    # Display components
    with st.expander("Show Forecast Components"):
        st.write("Prophet decomposition plots")
        model.plot_components(forecast)
else:
    st.info("Adjust parameters and click 'Run Forecast' to start.")
