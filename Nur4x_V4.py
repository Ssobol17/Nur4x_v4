"""
Forex Prediction App - Master Copy
Version: 1.4
Last Updated: December 26, 2024
"""

import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import plotly.graph_objs as go
from datetime import timedelta
import logging
import ta

class ForexPredictionApp:
    def __init__(self):
        self.version = "1.1"
        self.lookback_period = 50
        self.feature_columns = [
            'Close', 'SMA', 'EMA', 'RSI', 'MACD', 
            'BB_upper', 'BB_lower', 'ATR', 'OBV'
        ]
        self.scaler = MinMaxScaler()
        self.model = self.build_model()
        self.valid_pairs = {
            'EURUSD': 'EUR/USD',
            'GBPUSD': 'GBP/USD',
            'USDJPY': 'USD/JPY',
            'AUDUSD': 'AUD/USD',
            'USDCAD': 'USD/CAD'
        }
        self.valid_intervals = {'1d': '1 day', '1h': '1 hour', '15m': '15 minutes'}
        self.model_path = 'best_model.h5'
        self.setup_logging()

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def validate_pair(self, pair):
        if not isinstance(pair, str):
            raise ValueError("Forex pair must be a string")
        pair = pair.upper().strip()
        if pair not in self.valid_pairs:
            raise ValueError(f"Invalid pair. Supported pairs: {', '.join(self.valid_pairs.keys())}")
        return pair

        if not isinstance(start_date, (pd.Timestamp, pd.DatetimeIndex, str)):
            raise ValueError("Invalid start_date format")
        if not isinstance(end_date, (pd.Timestamp, pd.DatetimeIndex, str)):
            raise ValueError("Invalid end_date format")

        start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
end_date:raise ValueError("Sta)
                          
Error(f"Date range must be at least {min_days} days for reliable predictions")

        max_lookback = pmestamp.now() - timedelta(days=365*5)
        if start_date < max_lookback:
            raise ValueError("Historical data limited to last 5 years")

        return start_date, end_date


    def validate_interval(self, interval):
        if interval not in self.valid_intervals:
            raise ValueError(f"Invalid interval. Supported intervals: {', '.join(self.valid_intervals.keys())}")
        return interval

    def validate_data(self, data):
        if data is None or data.empty:
            raise ValueError("No data received from API")
        
        min_required = self.lookback_period * 2
        if len(data) < min_required:
            raise ValueError(f"Insufficient data points. Need at least {min_required} points for analysis")
        
        missing_values = data[['Close']].isnull().sum().sum()
        if missing_values > len(data) * 0.1:
            raise ValueError(f"Too many missing values: {missing_values} points")
        return True

    def build_model(self):
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(self.lookback_period, len(self.feature_columns))),
            Dropout(0.3),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32, return_sequences=False),
            Dropout(0.3),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(optimizer=optimizer, loss='huber')
        return model

    #@st.cache_data(ttl=3600)
    def cached_fetch_data(pair, start_date, end_date, interval='1d'):
        """Separate cached function for data fetching"""
        return yf.download(
            f"{pair[0:3]}{pair[3:6]}=X",
            start=start_date,
            end=end_date,
            interval=interval
    )
    def fetch_data(self, pair, start_date, end_date, interval='1d'):
        """Removed @st.cache_data decorator and implemented direct fetch"""
        try:
            pair = self.validate_pair(pair)
            start_date, end_date = self.validate_dates(start_date, end_date)
            interval = self.validate_interval(interval)

            symbol = f"{pair[0:3]}{pair[3:6]}=X"
            self.logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")
            
            data = yf.download(symbol, start=start_date, end=end_date, interval=interval)
            self.validate_data(data)
            return self.add_technical_indicators(data)

        except ValueError as e:
            st.error(str(e))
            self.logger.error(f"Validation error: {str(e)}")
            return None
        except Exception as e:
            st.error("An unexpected error occurred while fetching data")
            self.logger.error(f"Unexpected error in fetch_data: {str(e)}")
            return None

    def add_technical_indicators(self, data):
        try:
            data['SMA'] = ta.trend.sma_indicator(data['Close'], window=14)
            data['EMA'] = ta.trend.ema_indicator(data['Close'], window=14)
            data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
            
            macd = ta.trend.MACD(data['Close'])
            data['MACD'] = macd.macd()
            
            bollinger = ta.volatility.BollingerBands(data['Close'])
            data['BB_upper'] = bollinger.bollinger_hband()
            data['BB_lower'] = bollinger.bollinger_lband()
            
            data['ATR'] = ta.volatility.average_true_range(data['High'], data['Low'], data['Close'])
            data['OBV'] = ta.volume.on_balance_volume(data['Close'], data['Volume'])
            
            data.fillna(method='bfill', inplace=True)
            data.fillna(method='ffill', inplace=True)
            
            return data
        except Exception as e:
            st.error(f"Error adding technical indicators: {e}")
            self.logger.error(f"Error in technical indicators: {str(e)}")
            return data

    def prepare_data(self, data):
        try:
            if data is None:
                raise ValueError("No data available for preparation")

            missing_features = [col for col in self.feature_columns if col not in data.columns]
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")

            scaled_data = self.scaler.fit_transform(data[self.feature_columns])
            X, y = [], []
            
            for i in range(self.lookback_period, len(scaled_data)):
                X.append(scaled_data[i - self.lookback_period:i])
                y.append(scaled_data[i, 0])

            if len(X) == 0:
                raise ValueError("Insufficient data points after preparation")

            return np.array(X), np.array(y)
        except Exception as e:
            st.error(f"Error preparing data: {str(e)}")
            self.logger.error(f"Error in prepare_data: {str(e)}")
            return None, None

    def train_model(self, X_train, y_train, epochs=50, batch_size=32):
        try:
            tscv = TimeSeriesSplit(n_splits=5)
            
            early_stopping = EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            )
            
            checkpoint = ModelCheckpoint(
                self.model_path,
                monitor='val_loss',
                save_best_only=True
            )
            
            return self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=0.2,
                callbacks=[early_stopping, checkpoint],
                verbose=1
            )
        except Exception as e:
            st.error(f"Error during training: {e}")
            self.logger.error(f"Error in model training: {str(e)}")
            return None

    def make_prediction(self, X):
        try:
            predictions = self.model.predict(X)
            return self.scaler.inverse_transform(predictions)
        except Exception as e:
            st.error(f"Error making predictions: {e}")
            self.logger.error(f"Error in predictions: {str(e)}")
            return None

def main():
    st.title("Forex Pair Prediction")

    app = ForexPredictionApp()
    st.sidebar.text(f"Version: {app.version}")

    pair = st.sidebar.text_input("Forex Pair (e.g., EURUSD)", value="EURUSD")
    start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
    end_date = st.sidebar.date_input("End Date", value=pd.to_datetime("2023-12-31"))
    interval = st.sidebar.selectbox("Interval", ["1d", "1h", "15m"], index=0)

    data = app.fetch_data(pair, start_date, end_date, interval)
    if data is not None:
        st.write("### Historical Data", data)

        X, y = app.prepare_data(data)
        if X is not None and y is not None:
            split_idx = int(len(X) * 0.8)
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            history = app.train_model(X_train, y_train)
            if history:
                st.write("### Training History")
                st.line_chart(pd.DataFrame(history.history['loss']), height=200)

            predictions = app.make_prediction(X_test)
            if predictions is not None:
                st.write("### Predictions vs Actual")
                comparison = pd.DataFrame({
                    'Actual': app.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten(),
                    'Predicted': predictions.flatten()
                })
                st.line_chart(comparison)

                st.write("### Next 5 Days Prediction")
                recent_data = X[-1].reshape(1, app.lookback_period, len(app.feature_columns))
                future_predictions = []

                for _ in range(5):
                    next_prediction = app.make_prediction(recent_data)[0, 0]
                    future_predictions.append(next_prediction)
                    next_input = np.append(recent_data[0, 1:], [[next_prediction]], axis=0)
                    recent_data = next_input.reshape(1, app.lookback_period, len(app.feature_columns))

                prediction_dates = pd.date_range(start=pd.to_datetime("today"), periods=5).strftime('%Y-%m-%d')
                predicted_data = pd.DataFrame({'Date': prediction_dates, 'Predicted Close': future_predictions})
                st.write(predicted_data)


if __name__ == "__main__":
    main()