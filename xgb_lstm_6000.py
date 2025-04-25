import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

def run_xgb_lstm_6000(df, st):
    try:
        # Clean data
        df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
        df['date_and_time'] = pd.to_datetime(df['date_and_time'], errors='coerce')
        df['6000kcal_price'] = pd.to_numeric(df['6000kcal_price'], errors='coerce')
        df = df.dropna(subset=['date_and_time', '6000kcal_price'])
        df.set_index('date_and_time', inplace=True)

        # XGBoost Forecasting
        xgb_df = df[['6000kcal_price']].copy()
        for lag in range(1, 8):
            xgb_df[f'lag_{lag}'] = xgb_df['6000kcal_price'].shift(lag)
        xgb_df = xgb_df.dropna()

        train_xgb = xgb_df.iloc[:-30]
        test_xgb = xgb_df.iloc[-30:]

        X_train_xgb = train_xgb.drop(columns=['6000kcal_price'])
        y_train_xgb = train_xgb['6000kcal_price']
        X_test_xgb = test_xgb.drop(columns=['6000kcal_price'])
        y_test_xgb = test_xgb['6000kcal_price']

        xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1)
        xgb_model.fit(X_train_xgb, y_train_xgb)
        xgb_preds = xgb_model.predict(X_test_xgb)

        # LSTM Forecasting
        scaler = MinMaxScaler()
        scaled_prices = scaler.fit_transform(df[['6000kcal_price']])

        def create_sequences(data, seq_len):
            X, y = [], []
            for i in range(seq_len, len(data)):
                X.append(data[i - seq_len:i])
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
        lstm_preds_scaled = model.predict(X_test_lstm)
        lstm_preds = scaler.inverse_transform(lstm_preds_scaled)

        # Metrics
    xgb_mse = mean_squared_error(y_test, xgb_pred)
    xgb_rmse = np.sqrt(xgb_mse)

    lstm_mse = mean_squared_error(y_test_lstm_inv, lstm_pred)
    lstm_rmse = np.sqrt(lstm_mse)


        # Streamlit Output
        st.subheader("XGBoost + LSTM (6000kcal)")
        st.write(f"**XGBoost RMSE:** {xgb_rmse:.2f}")
        st.write(f"**LSTM RMSE:** {lstm_rmse:.2f}")

        # Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(test_xgb.index, y_test_xgb.values, label='Actual', marker='o')
        ax.plot(test_xgb.index, xgb_preds, label='XGBoost Forecast', marker='x')
        ax.plot(test_xgb.index, lstm_preds.flatten(), label='LSTM Forecast', marker='s')
        ax.set_title("6000kcal Price Forecast")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error in XGB-LSTM model: {str(e)}")