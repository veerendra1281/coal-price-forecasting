import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def run_sarima_hw_5500(df, st):
    try:
        # Clean data
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        df['date_and_time'] = pd.to_datetime(df['date_and_time'], errors='coerce')
        df.sort_values('date_and_time', inplace=True)
        df.fillna(method='ffill', inplace=True)
        df['5500kcalfob_price'] = pd.to_numeric(df['5500kcal_fob_price'], errors='coerce')
        
        # Prepare time series data
        ts_data = df[['date_and_time', '5500kcalfob_price']].dropna()
        ts_data.set_index('date_and_time', inplace=True)

        # Train-test split
        forecast_horizon = 7
        train = ts_data.iloc[:-forecast_horizon]
        test = ts_data.iloc[-forecast_horizon:]

        # SARIMA model
        sarima_model = SARIMAX(train, order=(1,1,1), seasonal_order=(0,1,1,7),
                            enforce_stationarity=False, enforce_invertibility=False)
        sarima_fit = sarima_model.fit(disp=False)
        sarima_forecast = sarima_fit.forecast(steps=forecast_horizon)

        # Holt-Winters model
        hw_model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=7)
        hw_fit = hw_model.fit()
        hw_forecast = hw_fit.forecast(steps=forecast_horizon)

        # Metrics
        sarima_rmse = mean_squared_error(test, sarima_forecast, squared=False)
        sarima_r2 = r2_score(test, sarima_forecast)
        hw_rmse = mean_squared_error(test, hw_forecast, squared=False)
        hw_r2 = r2_score(test, hw_forecast)

        # Display results in Streamlit
        st.subheader("SARIMA + Holt-Winters (5500kcal FOB)")
        st.write(f"**SARIMA RMSE:** {sarima_rmse:.2f}")
        st.write(f"**SARIMA R²:** {sarima_r2:.2f}")
        st.write(f"**Holt-Winters RMSE:** {hw_rmse:.2f}")
        st.write(f"**Holt-Winters R²:** {hw_r2:.2f}")

        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(test.index, test.values, label='Actual', marker='o')
        ax.plot(test.index, sarima_forecast.values, label='SARIMA Forecast', marker='x')
        ax.plot(test.index, hw_forecast.values, label='Holt-Winters Forecast', marker='s')
        ax.set_title('5500kcal FOB Price Forecast')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}")