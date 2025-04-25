import streamlit as st
import pandas as pd

from sarima_4800 import run_sarima_4800
from sarima_hw_5500 import run_sarima_hw_5500
from xgb_lstm_5700 import run_xgb_lstm_5700
from xgb_lstm_6000 import run_xgb_lstm_6000
from arima_rf_5500cfr import run_arima_rf_5500cfr



st.set_page_config(page_title="Coal Price Forecasting", layout="wide")
st.title("🧮 Coal Price Forecasting App")

# Upload Excel File
uploaded_file = st.file_uploader("Upload Coal Historical Excel File", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file, sheet_name="2024")
    st.success("✅ Data successfully loaded!")

    # Model Selector
    model_choice = st.selectbox(
        "Select Forecasting Model:",
        (
            "SARIMA (4800kcal)",
            "SARIMA + Holt-Winters (5500kcal FOB)",
            "XGBoost + LSTM (5700kcal)",
            "XGBoost + LSTM (6000kcal)",
            "ARIMA + Random Forest (5500kcal CFR)",
        )
    )

    if st.button("Run Forecast"):
        st.write("---")  # Separator

        # Call selected model's function
        if model_choice == "SARIMA (4800kcal)":
            run_sarima_4800(df, st)
        elif model_choice == "SARIMA + Holt-Winters (5500kcal FOB)":
            run_sarima_hw_5500(df, st)
        elif model_choice == "XGBoost + LSTM (5700kcal)":
            run_xgb_lstm_5700(df, st)
        elif model_choice == "XGBoost + LSTM (6000kcal)":
            run_xgb_lstm_6000(df, st)
        elif model_choice == "ARIMA + Random Forest (5500kcal CFR)":
            run_arima_rf_5500cfr(df, st)
else:
    st.info(" Please upload the Excel file to begin.")
