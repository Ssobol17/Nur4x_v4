import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from datetime import timedelta
import logging
import ta
from typing import Tuple, Optional, Dict, List
import os

import plotly.graph_objs as go


@st.cache_data
def cached_fetch_data(pair: str, start_date: str, end_date: str, interval: str = '1d') -> pd.DataFrame:
    """Cached function for fetching data from yfinance."""
    symbol = f"{pair[0:3]}{pair[3:6]}=X"
    return yf.download(
        symbol,
        start=start_date,
        end=end_date,
        interval=interval
    )


class ForexPredictionApp:
    def fetch_data(self, pair: str, start_date: str, end_date: str, interval: str = '1d') -> Optional[pd.DataFrame]:
        """
        Fetch and validate forex data.

        Args:
            pair: Currency pair (e.g., 'EURUSD')
            start_date: Start date for data fetch
            end_date: End date for data fetch
            interval: Data interval ('1d', '1h', '15m', '5m')

        Returns:
            Optional[pd.DataFrame]: Processed DataFrame or None if error occurs
        """
        try:
            # Validate inputs
            pair = self.validate_pair(pair)
            start_date, end_date = self.validate_dates(start_date, end_date)
            interval = self.validate_interval(interval)

            symbol = f"{pair[0:3]}{pair[3:6]}=X"
            self.logger.info(f"Fetching data for {symbol} from {start_date} to {end_date}")

            # Use cached data fetching
            data = cached_fetch_data(pair, start_date, end_date, interval)

            # Validate the fetched data
            if self.validate_data(data):
                # Add technical indicators
                processed_data = self.add_technical_indicators(data)

                # Verify all required features are present
                missing_features = [col for col in self.feature_columns if col not in processed_data.columns]
                if missing_features:
                    raise ValueError(f"Missing required features after processing: {missing_features}")

                return processed_data

            return None

        except ValueError as e:
            self.logger.error(f"Validation error in fetch_data: {str(e)}")
            st.error(str(e))
            return None

        except Exception as e:
            self.logger.error(f"Unexpected error in fetch_data: {str(e)}")
            st.error("An unexpected error occurred while fetching data. Please check the logs for details.")
            return None

    """A Streamlit application for predicting Forex currency pair movements."""

    def __init__(self):
        self.version = "1.1.2"
        self.lookback_period = 50
        self.feature_columns = [
            'Close', 'SMA', 'EMA', 'RSI', 'MACD',
            'BB_upper', 'BB_lower', 'ATR', 'OBV',
            'ROC', 'VWAP'  # Added new technical indicators
        ]
        self.scaler = MinMaxScaler()
        self.model = None  # Initialize model only when needed
        self.valid_pairs = {
            'EURUSD': 'EUR/USD',
            'GBPUSD': 'GBP/USD',
            'USDJPY': 'USD/JPY',
            'AUDUSD': 'AUD/USD',
            'USDCAD': 'USD/CAD',
            'EURGBP': 'EUR/GBP',  # Added more currency pairs
            'EURJPY': 'EUR/JPY',
            'GBPJPY': 'GBP/JPY'
        }
        self.valid_intervals = {
            '1d': '1 day',
            '1h': '1 hour',
            '15m': '15 minutes',
            '5m': '5 minutes'  # Added more granular timeframe
        }
        self.model_dir = 'models'
        os.makedirs(self.model_dir, exist_ok=True)
        self.setup_logging()

    def setup_logging(self) -> None:
        """Configure logging with rotating file handler."""
        log_file = 'forex_prediction.log'
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(file_handler)

    def get_model_path(self, pair: str, interval: str) -> str:
        """Generate unique model path for each pair/interval combination."""
        return os.path.join(self.model_dir, f'model_{pair}_{interval}.h5')

    def load_or_build_model(self, pair: str, interval: str) -> None:
        """Load existing model or build new one if not exists."""
        model_path = self.get_model_path(pair, interval)
        try:
            if os.path.exists(model_path):
                self.model = load_model(model_path)
                self.logger.info(f"Loaded existing model from {model_path}")
            else:
                self.model = self.build_model()
                self.logger.info(f"Built new model for {pair} {interval}")
        except Exception as e:
            self.logger.error(f"Error loading/building model: {str(e)}")
            self.model = self.build_model()

    def build_model(self) -> Sequential:
        """Build LSTM model with improved architecture."""
        model = Sequential([
            LSTM(128, return_sequences=True, input_shape=(self.lookback_period, len(self.feature_columns))),
            Dropout(0.3),
            LSTM(64, return_sequences=True),
            Dropout(0.3),
            LSTM(32, return_sequences=False),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1)
        ])

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        model.compile(
            optimizer=optimizer,
            loss='huber',
            metrics=['mae', 'mse']  # Added metrics for better monitoring
        )
        return model

    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Enhanced technical indicator calculation with error handling."""
        try:
            # Existing indicators
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

            # New indicators
            data['ROC'] = ta.momentum.roc(data['Close'], window=12)
            data['VWAP'] = (data['Close'] * data['Volume']).cumsum() / data['Volume'].cumsum()

            # Handle missing values more robustly
            for col in data.columns:
                if data[col].isnull().any():
                    data[col] = data[col].fillna(method='ffill').fillna(method='bfill')

            return data
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {str(e)}")
            raise

    def validate_pair(self, pair: str) -> str:
        """Validate the forex pair input."""
        if not isinstance(pair, str):
            raise ValueError("Forex pair must be a string")
        pair = pair.upper().strip()
        if pair not in self.valid_pairs:
            raise ValueError(f"Invalid pair. Supported pairs: {', '.join(self.valid_pairs.keys())}")
        return pair

    def validate_dates(self, start_date: str, end_date: str) -> Tuple[pd.Timestamp, pd.Timestamp]:
        """Validate the date range for data fetching."""
        try:
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)

            if start_date >= end_date:
                raise ValueError("Start date must be before end date")

            min_days = self.lookback_period + 10
            if (end_date - start_date).days < min_days:
                raise ValueError(f"Date range must be at least {min_days} days for reliable predictions")

            max_lookback = pd.Timestamp.now() - timedelta(days=365 * 5)
            if start_date < max_lookback:
                raise ValueError("Historical data limited to last 5 years")

            return start_date, end_date

        except Exception as e:
            raise ValueError(f"Date validation error: {str(e)}")

    def validate_interval(self, interval: str) -> str:
        """Validate the time interval."""
        if interval not in self.valid_intervals:
            raise ValueError(f"Invalid interval. Supported intervals: {', '.join(self.valid_intervals.keys())}")
        return interval

    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate the fetched data."""
        if data is None or data.empty:
            raise ValueError("No data received from API")

        min_required = self.lookback_period * 2
        if len(data) < min_required:
            raise ValueError(f"Insufficient data points. Need at least {min_required} points for analysis")

        missing_values = data[['Close']].isnull().sum().sum()
        if missing_values > len(data) * 0.1:
            raise ValueError(f"Too many missing values: {missing_values} points")
        return True

    def prepare_prediction_interval(self, predictions: np.ndarray, confidence: float = 0.95) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate prediction intervals using historical error distribution."""
        z_score = stats.norm.ppf((1 + confidence) / 2)
        std_dev = np.std(predictions)
        lower_bound = predictions - (z_score * std_dev)
        upper_bound = predictions + (z_score * std_dev)
        return lower_bound, upper_bound

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """Calculate and return model performance metrics."""
        predictions = self.model.predict(X_test)
        metrics = {
            'mae': mean_absolute_error(y_test, predictions),
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'r2': r2_score(y_test, predictions)
        }
        return metrics

    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def get_market_sentiment(self, pair: str) -> Dict[str, float]:
        """Analyze market sentiment using news headlines and social media."""
        # Placeholder for sentiment analysis implementation
        return {
            'bullish_probability': 0.6,
            'bearish_probability': 0.4,
            'confidence_score': 0.8
        }

    def prepare_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for model training and prediction."""
        try:
            if data is None:
                raise ValueError("No data available for preparation")

            missing_features = [col for col in self.feature_columns if col not in data.columns]
            if missing_features:
                raise ValueError(f"Missing features: {missing_features}")

            # Scale the features
            feature_data = data[self.feature_columns].values
            scaled_data = self.scaler.fit_transform(feature_data)

            X, y = [], []
            for i in range(self.lookback_period, len(scaled_data)):
                X.append(scaled_data[i - self.lookback_period:i])
                y.append(scaled_data[i, 0])  # 0 index corresponds to Close price

            if len(X) == 0:
                raise ValueError("Insufficient data points after preparation")

            return np.array(X), np.array(y)

        except Exception as e:
            self.logger.error(f"Error in prepare_data: {str(e)}")
            raise

    def make_prediction(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the trained model."""
        try:
            if self.model is None:
                raise ValueError("Model not initialized. Please train the model first.")

            predictions = self.model.predict(X)

            # Create a dummy array with the same number of features as the original data
            dummy_array = np.zeros((len(predictions), len(self.feature_columns)))
            dummy_array[:, 0] = predictions.flatten()  # Put predictions in the first column

            # Inverse transform the entire dummy array
            inverse_transformed = self.scaler.inverse_transform(dummy_array)

            # Return only the first column (Close price predictions)
            return inverse_transformed[:, 0]

        except Exception as e:
            self.logger.error(f"Error in make_prediction: {str(e)}")
            raise


def main():
    st.set_page_config(page_title="Forex Prediction Pro", layout="wide")

    # Add CSS for better styling
    st.markdown("""
        <style>
        .stButton>button {
            width: 100%;
            margin-top: 1rem;
        }
        .metric-card {
            background-color: #f0f2f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

    app = ForexPredictionApp()

    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Prediction", "Analysis", "Settings"])

    with tab1:
        st.title("Forex Pair Prediction")

        col1, col2 = st.columns(2)

        with col1:
            pair = st.selectbox("Select Forex Pair", list(app.valid_pairs.keys()))
            interval = st.selectbox("Select Interval", list(app.valid_intervals.keys()))

        with col2:
            start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
            end_date = st.date_input("End Date", value=pd.to_datetime("2023-12-31"))

        if st.button("Generate Prediction"):
            with st.spinner("Processing data..."):
                try:
                    data = app.fetch_data(pair, start_date, end_date, interval)
                    if data is not None:
                        X, y = app.prepare_data(data)

                        # Display interactive charts
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=data['Close'],
                            mode='lines',
                            name='Historical Price'
                        ))
                        st.plotly_chart(fig)

                        # Show predictions with confidence intervals
                        predictions = app.make_prediction(X)
                        lower_bound, upper_bound = app.prepare_prediction_interval(predictions)

                        metrics = app.evaluate_model(X[-len(predictions):], y[-len(predictions):])

                        # Display metrics in cards
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("MAE", f"{metrics['mae']:.4f}")
                        with col2:
                            st.metric("RMSE", f"{metrics['rmse']:.4f}")
                        with col3:
                            st.metric("R² Score", f"{metrics['r2']:.4f}")

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

    with tab2:
        st.header("Technical Analysis")
        # Add technical analysis components here

    with tab3:
        st.header("Settings")
        # Add settings components here


if __name__ == "__main__":
    main()
