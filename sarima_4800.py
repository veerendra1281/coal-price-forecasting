
# sarima_4800.py

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, r2_score

def run_sarima_4800(df, st):
    try:
        # Clean data
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        df['date_and_time'] = pd.to_datetime(df['date_and_time'], errors='coerce')
        df.sort_values('date_and_time', inplace=True)
        df.fillna(method='ffill', inplace=True)

        ts_data = df[['date_and_time', '4800kcal_price']].dropna()
        ts_data.set_index('date_and_time', inplace=True)

        # Train-test split
        forecast_horizon = 7
        train = ts_data.iloc[:-forecast_horizon]
        test = ts_data.iloc[-forecast_horizon:]

        # SARIMA model
        model = SARIMAX(train, order=(1, 1, 1), seasonal_order=(0, 1, 1, 7),
                        enforce_stationarity=False, enforce_invertibility=False)
        results = model.fit(disp=False)
        forecast = results.forecast(steps=forecast_horizon)

        # Metrics
        rmse = mean_squared_error(test, forecast, squared=False)
        r2 = r2_score(test, forecast)

        # Output
        st.subheader("Forecasting Results (SARIMA - 4800kcal)")
        st.write(f"**RMSE:** {rmse:.2f}")
        st.write(f"**R² Score:** {r2:.2f}")

        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(test.index, test.values, label='Actual', marker='o')
        ax.plot(test.index, forecast.values, label='SARIMA Forecast', marker='x')
        ax.set_title('4800kcal Price Forecast (SARIMA)')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error: {e}")
