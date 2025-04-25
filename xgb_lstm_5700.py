import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

def run_xgb_lstm_5700(df, st):
    try:
        # Clean data
        df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['5700kcal'] = pd.to_numeric(df['5700kcal'], errors='coerce')
        df = df.dropna(subset=['date', '5700kcal'])
        df.set_index('date', inplace=True)

        # XGBoost Forecasting
        xgb_data = df[['5700kcal']].copy()
        for lag in range(1, 8):
            xgb_data[f'lag_{lag}'] = xgb_data['5700kcal'].shift(lag)
        xgb_data = xgb_data.dropna()

        train_xgb = xgb_data.iloc[:-30]
        test_xgb = xgb_data.iloc[-30:]

        X_train = train_xgb.drop(columns='5700kcal')
        y_train = train_xgb['5700kcal']
        X_test = test_xgb.drop(columns='5700kcal')
        y_test = test_xgb['5700kcal']

        xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1)
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)

        # LSTM Forecasting
        scaler = MinMaxScaler()
        scaled_prices = scaler.fit_transform(df[['5700kcal']])

        def create_sequences(data, seq_length):
            X, y = [], []
            for i in range(seq_length, len(data)):
                X.append(data[i - seq_length:i])
                y.append(data[i])
            return np.array(X), np.array(y)

        sequence_length = 10
        X_lstm, y_lstm = create_sequences(scaled_prices, sequence_length)
        X_train_lstm = X_lstm[:-30]
        y_train_lstm = y_lstm[:-30]
        X_test_lstm = X_lstm[-30:]
        y_test_lstm = y_lstm[-30:]

        model = Sequential([
            LSTM(50, activation='relu', input_shape=(sequence_length, 1)),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        model.fit(X_train_lstm, y_train_lstm, epochs=50, batch_size=8, verbose=0)
        lstm_pred_scaled = model.predict(X_test_lstm)
        lstm_pred = scaler.inverse_transform(lstm_pred_scaled)
        y_test_lstm_inv = scaler.inverse_transform(y_test_lstm)

        # Metrics
        xgb_mse = mean_squared_error(y_test, xgb_pred)
        xgb_rmse = np.sqrt(xgb_mse)

        lstm_mse = mean_squared_error(y_test_lstm_inv, lstm_pred)
        lstm_rmse = np.sqrt(lstm_mse)

        # Streamlit Output
        st.subheader("XGBoost + LSTM (5700kcal)")
        st.write(f"**XGBoost RMSE:** {xgb_rmse:.2f}")
        st.write(f"**LSTM RMSE:** {lstm_rmse:.2f}")

        # Plot
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(y_test.index, y_test.values, label="Actual", marker='o')
        ax.plot(y_test.index, xgb_pred, label="XGBoost Forecast", marker='x')
        ax.plot(y_test.index, lstm_pred.flatten(), label="LSTM Forecast", marker='s')
        ax.set_title("5700kcal Price Forecast")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error in XGB-LSTM model: {str(e)}")
