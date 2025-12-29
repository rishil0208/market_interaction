"""
Data Engine for fetching and processing REAL-TIME stock data.

Handles live, intraday data retrieval from yfinance and feature engineering
for the high-frequency interaction model.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from typing import List, Tuple

# Use Streamlit's caching to prevent re-fetching data too frequently.
# The data will be cached for 60 seconds (ttl=60).
@st.cache_data(ttl=60)
def fetch_live_data(tickers: List[str]) -> pd.DataFrame:
    """
    Fetches historical intraday data (1-minute interval) for a list of tickers
    for the last 5 days.

    Args:
        tickers (List[str]): List of stock tickers.

    Returns:
        pd.DataFrame: A DataFrame with 'Close' prices for each ticker.
                      Returns None if data fetching fails.
    """
    try:
        data = yf.download(
            tickers,
            period="5d",
            interval="1m",
            auto_adjust=True,
            progress=False  # Suppress yfinance progress bar in logs
        )['Close']
        
        # Forward-fill NaNs which can occur during market halts or for illiquid stocks
        data = data.ffill()
        # Then back-fill any remaining NaNs at the beginning of the series
        data = data.bfill()

        if data.empty:
            st.warning("yfinance returned an empty dataframe. Markets may be closed.")
            return None

        return data
    except Exception as e:
        st.error(f"Failed to fetch live data from yfinance: {e}")
        return None

def get_live_features(data: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """
    Calculates dynamic features from the last N minutes of data.
    These features capture short-term momentum and volatility.

    Args:
        data (pd.DataFrame): DataFrame of 1-minute prices.
        lookback (int): The rolling window in minutes for calculations.

    Returns:
        pd.DataFrame: A dataframe with the latest features for each ticker.
    """
    recent_data = data.tail(lookback)
    
    # 1. Log Returns (captures minute-to-minute change)
    log_returns = np.log(recent_data / recent_data.shift(1))

    # 2. Short-Term Momentum (total return over the lookback period)
    momentum = (recent_data.iloc[-1] / recent_data.iloc[0]) - 1

    # 3. Short-Term Volatility (std dev of 1-min returns)
    volatility = log_returns.std()

    feature_df = pd.DataFrame({
        'Momentum_60m': momentum,
        'Volatility_60m': volatility,
        'Last_1m_Return': log_returns.iloc[-1]
    })
    
    # Handle potential NaNs/Infs from calculations
    feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    feature_df.fillna(0, inplace=True)

    return feature_df


def get_live_correlation_matrix(data: pd.DataFrame, lookback: int, threshold: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates a dynamic adjacency and weighted correlation matrix based on
    the most recent N minutes of 1-minute log returns.

    Args:
        data (pd.DataFrame): DataFrame of 1-minute prices.
        lookback (int): Rolling window in minutes for correlation.
        threshold (float): Correlation value above which an edge is created.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - correlation_matrix: Full matrix of recent correlation values.
            - adjacency_matrix: A square matrix where A_ij = 1 if corr(i, j) > threshold.
    """
    recent_data = data.tail(lookback)
    log_returns = np.log(recent_data / recent_data.shift(1)).dropna()
    
    # Calculate the correlation matrix on the most recent data
    correlation_matrix = log_returns.corr()
    
    # Create adjacency matrix based on the threshold
    adjacency_matrix = (correlation_matrix.abs() > threshold).astype(int)
    
    # Ensure the diagonal is 0 (a stock has no self-loop)
    np.fill_diagonal(adjacency_matrix.values, 0)
    
    # Replace NaNs in correlation matrix with 0
    correlation_matrix.fillna(0, inplace=True)
    
    return correlation_matrix, adjacency_matrix