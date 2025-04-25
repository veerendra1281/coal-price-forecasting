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
        # Clean column names
        df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

        # Show available columns (for debugging)
        st.write("📋 Uploaded columns:", df.columns.tolist())

        # Handle date column flexibility
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
        elif 'date_and_time' in df.columns:
            df['date'] = pd.to_datetime(df['date_and_time'], errors='coerce')
        else:
            st.error("❌ Date column not found. Please include 'date' or 'date_and_time'.")
            return

        # Automatically detect the '5700kcal' column
        price_col = next((col for col in df.columns if '5700' in col), None)
        if not price_col:
            st.error("❌ Column for '5700kcal' not found in uploaded data.")
            return

        df[price_col] = pd.to_numeric(df[price_col], errors='coerce')
        df = df.dropna(subset=['date', price_col])
        df.set_index('date', inplace=True)

        # XGBoost Forecasting
        xgb_data = df[[price_col]].copy()
        for lag in range(1, 8):
            xgb_data[f'lag_{lag}'] = xgb_data[price_col].shift(lag)
        xgb_data = xgb_data.dropna()

        train_xgb = xgb_data.iloc[:-30]
        test_xgb = xgb_data.iloc[-30:]

        X_train = train_xgb.drop(columns=price_col)
        y_train = train_xgb[price_col]
        X_test = test_xgb.drop(columns=price_col)
        y_test = test_xgb[price_col]

        xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1)
        xgb_model.fit(X_train, y_train)
        xgb_pred = xgb_model.predict(X_test)

        # LSTM Forecasting
        scaler = MinMaxScaler()
        scaled_prices = scaler.fit_transform(df[[price_col]])

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
        st.error(f"❌ Error in XGB-LSTM model: {str(e)}")
