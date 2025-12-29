# 📈 QuantPlatform: Market Interaction-Based Performance Predictor

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25%2B-red?style=for-the-badge&logo=streamlit)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![Status](https://img.shields.io/badge/Status-Active-green?style=for-the-badge)]()

---

**A portfolio-grade financial analytics platform that predicts short-term stock performance by modeling the market as a dynamic graph and leveraging a Graph Attention Network (GAT).**

<br>

> **Note:** Please add a GIF of the running application here! A GIF is the best way to showcase this project on your portfolio.

<p align="center">
  <!-- ![DEMO_GIF_PLACEHOLDER](link_to_your_gif.gif) -->
  *Your GIF Here*
</p>

## 📋 Project Overview

This platform moves beyond traditional single-asset analysis by embracing a core market principle: **stocks do not move in a vacuum.** It hypothesizes that the performance of a stock is heavily influenced by the real-time behavior of its peers, sector leaders, and the broader market sentiment.

To model this, the application constructs a dynamic graph of the market every 60 seconds and uses a Graph Attention Network (GAT) to learn the complex, weighted relationships between assets. The result is a "Bloomberg-style" dashboard that not only provides predictions but, more importantly, **explains the reasoning behind them.**

## ✨ Key Features

- **Three-Pillar Data Engine:** The model's predictions are based on a rich feature set derived from:
    1.  **Price Action:** 1-minute intraday candlesticks, log returns, and rolling volatility.
    2.  **Technical Momentum:** RSI and Bollinger Bands calculated using the `ta` library.
    3.  **Live Market Sentiment:** Real-time news headlines are fetched and analyzed for sentiment polarity using `TextBlob`.
- **Interpretable AI Model:** A **Graph Attention Network (GAT)** built from scratch with PyTorch identifies the most influential stocks in the network and provides its "attention weights" as a basis for explaining its predictions.
- **"Bloomberg-Style" UI:** A professional, dark-mode dashboard built with Streamlit that auto-refreshes every 60 seconds.
- **Advanced Visualizations:**
    - Live-updating **Candlestick Charts** with overlaid Bollinger Bands.
    - A dynamic **Market Interaction Network** graph where edge thickness and color represent the live correlation strength between stocks.
- **🎓 Educational "University Mode":** A unique sidebar toggle that reveals the mathematical formulas and financial concepts behind every chart and prediction, making this a powerful learning tool.

## 🏗️ System Architecture

The platform operates on a 60-second cycle, following a clear data pipeline from ingestion to visualization:

1.  **Data Ingestion (`data_processor.py`):**
    - Fetches 1-minute OHLCV data from `yfinance`.
    - Fetches news headlines from `yfinance`.

2.  **Feature Engineering (`data_processor.py`):**
    - **Pillar 1 (Price):** Calculates Log Returns and Volatility.
    - **Pillar 2 (Momentum):** Calculates RSI and Bollinger Bands.
    - **Pillar 3 (Sentiment):** Calculates sentiment polarity scores from news.
    - These features are combined into a feature matrix.

3.  **Graph Construction (`data_processor.py`):**
    - Calculates a 60-minute rolling correlation matrix on log returns.
    - Creates a graph adjacency matrix where an edge `(A, B)` exists if `corr(A, B)` > threshold.

4.  **Model Inference (`gat_model.py`):**
    - The feature matrix and adjacency matrix are fed into the GAT.
    - The model outputs a "performance score" for each stock and the attention matrix that explains its reasoning.

5.  **Frontend Rendering (`app.py`):**
    - The Streamlit dashboard visualizes the scores, attention weights, and dynamic graph.
    - The entire process repeats automatically.

## 🚀 Getting Started

### Prerequisites
- Python 3.8 - 3.11
- Git

### 1. Clone the Repository
```bash
git clone <YOUR_REPOSITORY_URL>
cd <repository_folder>
```

### 2. Critical Setup: Download Language Corpora
The sentiment analysis feature requires a one-time download of the `textblob` language corpora.
```bash
python -m textblob.download_corpora
```

### 3. Install Dependencies
All required packages are listed in `requirements.txt`.
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
streamlit run app.py
```
The application will open in your web browser and begin fetching live data. Please be patient on the first run, as it may take a moment to collect the initial data.
