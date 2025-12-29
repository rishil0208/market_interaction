"""
Data Processor for the Portfolio-Grade Financial Analytics Platform.

This module is the heart of the data engine. It implements the "Three Pillars"
of data required for a realistic financial model:
1. Price Action (Log Returns, Volatility)
2. Technical Momentum (RSI, Bollinger Bands)
3. Market Sentiment (News Headline Analysis)
"""
import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from ta import add_all_ta_features
from ta.utils import dropna
from textblob import TextBlob
from typing import List, Tuple, Optional

@st.cache_data(ttl=60)
def fetch_market_data(tickers: List[str]) -> Optional[pd.DataFrame]:
    """
    Fetches 1-minute intraday data for a list of stock tickers.

    Financial Concept: For high-frequency analysis and real-time decision
    making, we need granular data. 1-minute "candles" provide a detailed view
    of price, volume, and volatility within a trading day, which is essential
    for capturing short-term market dynamics.

    Args:
        tickers: A list of stock ticker symbols.

    Returns:
        A multi-index DataFrame containing Open, High, Low, Close, and Volume
        data for each ticker, or None if the API call fails.
    """
    try:
        data = yf.download(
            tickers,
            period="5d",       # Fetch 5 days of data to ensure enough history
            interval="1m",     # The crucial 1-minute interval
            auto_adjust=True,
            progress=False,
            group_by='ticker'  # Organize data by ticker
        )
        if data.empty:
            return None
        return data
    except Exception:
        return None

def add_technical_indicators(df: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """
    Adds technical momentum indicators (RSI, Bollinger Bands) to the data.

    Financial Concept: Technical indicators are mathematical calculations based
    on historical price, volume, or open interest data. They are used to
    forecast price direction.
    - RSI (Relative Strength Index): A momentum oscillator that measures the
      speed and change of price movements. RSI values > 70 indicate an asset
      is overbought; values < 30 indicate it is oversold.
    - Bollinger Bands: A volatility indicator. The bands widen when volatility
      increases and narrow when it decreases. Prices are considered high when
      above the upper band and low when below the lower band.

    Args:
        df: The raw market data DataFrame.
        tickers: The list of tickers.

    Returns:
        The DataFrame enriched with technical indicator columns for each ticker.
    """
    df_copy = df.copy()
    for ticker in tickers:
        if ticker in df_copy.columns.get_level_values(0):
            # Calculate all technical analysis features for the specific ticker's sub-frame
            ta_df = add_all_ta_features(
                df_copy[ticker], open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True
            )
            
            # Identify the new columns created by the 'ta' library
            new_cols = [col for col in ta_df.columns if col not in df_copy[ticker].columns]
            
            # Add each new column back to the main multi-index DataFrame
            for col in new_cols:
                df_copy[(ticker, col)] = ta_df[col]
                
    # Forward-fill any remaining NaNs just in case
    return df_copy.ffill()

@st.cache_data(ttl=1800)  # Cache sentiment for 30 minutes
def get_sentiment(tickers: List[str]) -> pd.DataFrame:
    """
    Fetches the latest news for each ticker and calculates a sentiment score.

    Financial Concept: Market sentiment, or the overall attitude of investors
    toward a particular security, can be a significant driver of price
    movement, especially in the short term. We quantify this by analyzing the
    language used in news headlines.

    Args:
        tickers: A list of stock ticker symbols.

    Returns:
        A DataFrame with tickers and their corresponding sentiment scores.
    """
    sentiment_scores = []
    for ticker_symbol in tickers:
        try:
            ticker_obj = yf.Ticker(ticker_symbol)
            # yfinance .news is a list of dicts
            news = ticker_obj.news
            if not news:
                sentiment_scores.append({'ticker': ticker_symbol, 'sentiment': 0.0})
                continue

            # Analyze the sentiment of each headline and average the scores
            polarity_sum = 0
            for item in news:
                headline = item['title']
                blob = TextBlob(headline)
                polarity_sum += blob.sentiment.polarity  # Polarity is between -1 and 1

            avg_sentiment = polarity_sum / len(news)
            sentiment_scores.append({'ticker': ticker_symbol, 'sentiment': avg_sentiment})
        except Exception:
            # If news for a ticker fails, assign a neutral score
            sentiment_scores.append({'ticker': ticker_symbol, 'sentiment': 0.0})

    return pd.DataFrame(sentiment_scores).set_index('ticker')

def get_feature_matrix(
    full_data: pd.DataFrame,
    sentiment_data: pd.DataFrame,
    tickers: List[str],
    lookback_minutes: int
) -> pd.DataFrame:
    """
    Constructs the final feature matrix for the GAT model.

    This function combines all "Three Pillars" of data into a single,
    normalized matrix that the neural network can understand.

    Args:
        full_data: The complete DataFrame with prices and technical indicators.
        sentiment_data: The DataFrame with sentiment scores.
        tickers: The list of stock tickers.
        lookback_minutes: The rolling window for calculating features.

    Returns:
        A DataFrame where rows are tickers and columns are the final features.
    """
    latest_features = []
    for ticker in tickers:
        if ticker in full_data.columns.get_level_values(0):
            recent_data = full_data[ticker].tail(lookback_minutes)
            
            # Pillar 1: Price Action
            log_returns = np.log(recent_data['Close'] / recent_data['Close'].shift(1))
            volatility = log_returns.std()
            
            # Pillar 2: Technical Momentum
            latest_rsi = recent_data['momentum_rsi'].iloc[-1]
            
            # Pillar 3: Market Sentiment
            sentiment = sentiment_data.loc[ticker, 'sentiment']
            
            latest_features.append({
                'ticker': ticker,
                'Volatility': volatility,
                'RSI': latest_rsi,
                'Sentiment': sentiment,
                'Last_Return': log_returns.iloc[-1]
            })

    feature_df = pd.DataFrame(latest_features).set_index('ticker')
    feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return feature_df.fillna(0)

def create_dynamic_graph(
    price_data: pd.DataFrame,
    lookback_minutes: int,
    threshold: float
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates a dynamic graph representation of the market based on correlation.

    Financial Concept: Correlation is a measure of how two securities move in
    relation to each other. By creating a graph based on high correlation, we
    are building a model of the market's "structure" at a given moment. This
    allows our GAT model to learn how influence propagates through this structure.

    Args:
        price_data: DataFrame of 1-minute closing prices.
        lookback_minutes: The rolling window in minutes for correlation calculation.
        threshold: The correlation value above which a graph edge is created.

    Returns:
        A tuple containing the full correlation matrix and the binary adjacency matrix.
    """
    recent_prices = price_data.tail(lookback_minutes)
    log_returns = np.log(recent_prices / recent_prices.shift(1)).dropna()
    
    correlation_matrix = log_returns.corr()
    adjacency_matrix = (correlation_matrix.abs() > threshold).astype(int)
    
    np.fill_diagonal(adjacency_matrix.values, 0)
    correlation_matrix.fillna(0, inplace=True)
    
    return correlation_matrix, adjacency_matrix
