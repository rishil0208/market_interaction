"""
Data Engine for fetching and processing stock data.

Handles data retrieval from yfinance and feature engineering,
including individual and relational features.
"""
import yfinance as yf
import pandas as pd
import numpy as np
from typing import List, Tuple

def fetch_data(tickers: List[str], period: str = "1y") -> pd.DataFrame:
    """
    Fetches historical daily price data for a list of tickers.

    Args:
        tickers (List[str]): List of stock tickers.
        period (str): The time period for which to fetch data (e.g., "1y", "6mo").

    Returns:
        pd.DataFrame: A DataFrame with 'Adj Close' prices for each ticker.
    """
    data = yf.download(tickers, period=period, auto_adjust=True)['Close']
    data = data.dropna(axis=1, how='any') # Drop columns with any NaNs
    return data

def calculate_features(data: pd.DataFrame, lookback: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Calculates individual stock features: Log Returns, Volatility, and RSI.

    Args:
        data (pd.DataFrame): DataFrame of historical prices.
        lookback (int): The rolling window size for calculations.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            - Log Returns
            - Rolling Volatility
            - Relative Strength Index (RSI)
    """
    # 1. Log Returns: Captures momentum
    log_returns = np.log(data / data.shift(1)).dropna()

    # 2. Rolling Volatility: Captures risk/uncertainty
    volatility = log_returns.rolling(window=lookback).std().dropna()

    # 3. Relative Strength Index (RSI): Captures overbought/oversold conditions
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=lookback).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=lookback).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.dropna()

    # Align dataframes to the same index
    last_common_date = min(log_returns.index[-1], volatility.index[-1], rsi.index[-1])
    log_returns = log_returns.loc[:last_common_date]
    volatility = volatility.loc[:last_common_date]
    rsi = rsi.loc[:last_common_date]

    return log_returns, volatility, rsi

def get_adjacency_matrix(data: pd.DataFrame, lookback: int, threshold: float) -> pd.DataFrame:
    """
    Creates a dynamic adjacency matrix based on rolling correlation.
    This matrix represents the "Market Interaction Graph," where an edge
    exists if the correlation between two stocks exceeds a threshold.

    This directly satisfies the "Interaction" and "Dynamic Graph" requirements.

    Args:
        data (pd.DataFrame): DataFrame of historical prices.
        lookback (int): Rolling window for correlation calculation.
        threshold (float): Correlation value above which an edge is created.

    Returns:
        pd.DataFrame: A square matrix where A_ij = 1 if corr(i, j) > threshold, else 0.
    """
    # Calculate rolling correlation on log returns for more stable relationships
    log_returns = np.log(data / data.shift(1))
    rolling_corr = log_returns.rolling(window=lookback).corr().unstack()
    
    # Get the most recent correlation matrix
    latest_corr = rolling_corr.iloc[-1].unstack()
    
    # Create adjacency matrix based on the threshold
    adjacency = (latest_corr > threshold).astype(int)
    
    # Ensure the diagonal is 0 (a stock has no self-loop in this context)
    np.fill_diagonal(adjacency.values, 0)
    
    return adjacency
