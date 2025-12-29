"""
Data Engine for the Market Interaction-Based Performance Predictor.

This script is responsible for fetching, processing, and featurizing real-time
stock data. It's designed to be the foundational data layer for the GAT model.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from typing import List, Tuple, Optional

# Use Streamlit's caching to prevent re-fetching data too frequently.
# The data will be cached for 60 seconds (ttl=60), meaning the yfinance API
# is only called once per minute at most, respecting API rate limits and
# speeding up the app.
@st.cache_data(ttl=60)
def fetch_live_data(tickers: List[str]) -> Optional[pd.DataFrame]:
    """
    Fetches historical intraday data (1-minute interval) from Yahoo Finance.

    The "educational" aspect here is understanding that for high-frequency
    analysis, we need granular data. Daily data is too coarse to capture
    the "right now" dynamics of the market.

    Args:
        tickers (List[str]): A list of stock ticker symbols to fetch.

    Returns:
        Optional[pd.DataFrame]: 
            A DataFrame containing the 1-minute closing prices for each ticker.
            The index is a DatetimeIndex. Returns None if data fetching fails.
    """
    try:
        # We fetch data for the last 7 days to ensure we have enough historical
        # context even after filtering for market hours or data gaps.
        data = yf.download(
            tickers,
            period="7d",
            interval="1m",
            auto_adjust=True,  # Automatically adjusts for splits and dividends
            progress=False     # Suppress the download progress bar in the console
        )['Close']

        # --- Data Cleaning: Handling Missing Values ---
        # Intraday data can be messy. It often has NaNs (Not a Number) due
        # to market halts, low liquidity, or API errors. We must clean this.
        # 'ffill' (forward-fill) propagates the last valid observation forward.
        data = data.ffill()
        # 'bfill' (backward-fill) fills any remaining NaNs at the start of the series.
        data = data.bfill()

        if data.empty:
            st.warning("yfinance returned an empty dataframe. Markets may be closed.")
            return None

        return data
    except Exception as e:
        st.error(f"Failed to fetch live data from yfinance: {e}")
        return None

def calculate_live_features(data: pd.DataFrame, lookback_minutes: int) -> pd.DataFrame:
    """
    Engineers features from the raw 1-minute price data.

    The "educational" aspect is feature engineering. A model rarely uses raw
    prices. Instead, we create features that describe the *behavior* of the
    price, such as its momentum and volatility. These features provide a
    richer signal for the model to learn from.

    Args:
        data (pd.DataFrame): DataFrame of 1-minute prices.
        lookback_minutes (int): The rolling window in minutes for calculations.

    Returns:
        pd.DataFrame: A dataframe with the latest engineered features for each ticker.
    """
    # We only need the most recent 'lookback_minutes' of data for our features.
    recent_data = data.tail(lookback_minutes)

    # --- Feature 1: Log Returns ---
    # The Math: We use log returns, calculated as log(price_t / price_{t-1}).
    # Why? Log returns are normalized and additive over time, which are
    # properties that are highly beneficial for statistical and ML models.
    # It represents the continuous compounding return.
    log_returns = np.log(recent_data / recent_data.shift(1))

    # --- Feature 2: Short-Term Momentum ---
    # The Math: (last_price / first_price) - 1 over the lookback window.
    # Why? This captures the overall trend or "drift" of the stock in the
    # recent past. A positive value indicates an upward trend.
    momentum = (recent_data.iloc[-1] / recent_data.iloc[0]) - 1

    # --- Feature 3: Short-Term Volatility ---
    # The Math: The standard deviation of the 1-minute log returns.
    # Why? This measures the magnitude of price fluctuation. High volatility
    # means the stock is making large price swings and is considered riskier.
    volatility = log_returns.std()

    # Combine all features into a single DataFrame.
    feature_df = pd.DataFrame({
        'Momentum': momentum,
        'Volatility': volatility,
        'Last_Return': log_returns.iloc[-1]
    })

    # Final cleaning step to handle any potential divisions by zero or other
    # calculation errors that might result in infinity or NaN values.
    feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    feature_df.fillna(0, inplace=True)

    return feature_df

def create_dynamic_graph(data: pd.DataFrame, lookback_minutes: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates a dynamic graph representation of the market based on correlation.

    The "educational" aspect is graph construction. This is how we model the
    "interaction" between stocks. We hypothesize that stocks that move together
    (are highly correlated) are related and should influence each other's
    predicted performance.

    Args:
        data (pd.DataFrame): DataFrame of 1-minute prices.
        lookback_minutes (int): Rolling window in minutes for correlation.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - correlation_matrix: The full matrix of recent correlation values.
            - adjacency_matrix: A binary matrix representing the graph structure.
    """
    # Again, we focus on the most recent slice of data.
    recent_data = data.tail(lookback_minutes)
    log_returns = np.log(recent_data / recent_data.shift(1)).dropna()

    # --- Correlation Matrix: The "Blueprint" of the Graph ---
    # The Math: We calculate the Pearson correlation coefficient for the log
    # returns of every pair of stocks. A value of +1 means they move
    # perfectly together; -1 means they move perfectly opposite.
    correlation_matrix = log_returns.corr()

    # --- Adjacency Matrix: The "Structure" of the Graph ---
    # The Math: This is a binary matrix derived from the correlation matrix.
    # If the absolute correlation between two stocks is above a certain
    # threshold, we set the corresponding entry to 1 (an "edge" exists),
    # otherwise 0. This tells the model which stocks are "connected."
    adjacency_matrix = (correlation_matrix.abs() > 0.5).astype(int)

    # A stock cannot be its own neighbor in this model.
    np.fill_diagonal(adjacency_matrix.values, 0)
    
    # Final cleaning
    correlation_matrix.fillna(0, inplace=True)

    return correlation_matrix, adjacency_matrix
