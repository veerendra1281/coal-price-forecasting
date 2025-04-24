import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX

def run_arima_rf_5500cfr(df, st):
    try:
        # Clean data
        df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
        df = df[["date_and_time", "5500kcal_cfr_price"]].dropna()
        df["date_and_time"] = pd.to_datetime(df["date_and_time"])
        df.set_index("date_and_time", inplace=True)
        ts_data = df["5500kcal_cfr_price"].astype(float)

        # === ARIMA Model ===
        arima_model = SARIMAX(ts_data, order=(1,1,1), seasonal_order=(1,1,1,12))
        arima_result = arima_model.fit(disp=False)
        arima_forecast = arima_result.get_forecast(steps=30)
        arima_preds = arima_forecast.predicted_mean

        # === Random Forest Model ===
        rf_df = ts_data.to_frame()
        for lag in range(1, 8):
            rf_df[f"lag_{lag}"] = rf_df["5500kcal_cfr_price"].shift(lag)
        rf_df.dropna(inplace=True)

        X = rf_df.drop("5500kcal_cfr_price", axis=1)
        y = rf_df["5500kcal_cfr_price"]
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        # Generate RF forecast
        last_known = X.iloc[-1:].values
        rf_preds = []
        for _ in range(30):
            pred = model.predict(last_known)[0]
            rf_preds.append(pred)
            last_known = np.roll(last_known, -1)
            last_known[0, -1] = pred

        # === Streamlit Output ===
        st.subheader("ARIMA + Random Forest (5500kcal CFR)")
        
        # Metrics
        st.write("**Model Metrics:**")
        st.write("ARIMA Forecast for next 30 days generated")
        st.write("Random Forest Forecast for next 30 days generated")

        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(ts_data.index, ts_data.values, label="Historical Data")
        ax.plot(pd.date_range(ts_data.index[-1], periods=30, freq="D"), 
                arima_preds, label="ARIMA Forecast", marker='x')
        ax.plot(pd.date_range(ts_data.index[-1], periods=30, freq="D"), 
                rf_preds, label="RF Forecast", marker='s')
        ax.set_title("5500kcal CFR Price Forecast")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {str(e)}")