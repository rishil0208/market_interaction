# Market Interaction-Based Performance Predictor

## 📖 Overview

This project is a Python application that predicts the relative performance ranking of a basket of tech stocks for the next trading day. It uses a graph-based model to explicitly account for inter-company relationships, satisfying the core requirement that stocks should not be treated in isolation.

The application features a modern, professional frontend built with Streamlit, providing clear, interpretable signals and visualizations.

---

## ✨ Features

- **Interaction-Based Modeling**: Implements a simplified Graph Attention Network (GAT) to model how companies influence each other's performance.
- **Dynamic Graph Construction**: Builds a daily "market interaction map" using a rolling correlation matrix to adapt to changing market dynamics.
- **High Interpretability**: The model's "attention weights" are used to explain *why* a prediction was made, identifying the key influencing companies.
- **Interactive Dashboard**: A clean, dark-themed Streamlit UI for visualizing predictions, market graphs, and model explanations.
- **Data-Driven Visuals**: Uses Plotly and NetworkX to render an interactive node-link diagram of the market interaction graph.

---

## 🛠️ Tech Stack

- **Backend & Modeling**: Python, PyTorch, Pandas, NumPy
- **Data Source**: `yfinance` for live and historical stock data.
- **Dashboard**: Streamlit
- **Visualizations**: Plotly, NetworkX

---

## ⚙️ Methodology

The model's predictive power comes from how it represents the market as a dynamic graph:

1.  **Feature Engineering**: For each stock, standard features like Log Returns, Volatility, and RSI are calculated.
2.  **Graph Construction**: A **Rolling Correlation Matrix** (30-day lookback) is computed. If the correlation between two companies exceeds a threshold (e.g., > 0.6), a weighted "edge" is created between them in an adjacency matrix. This forms the dynamic graph.
3.  **Interaction Scoring**: An `InteractionScorer` (a simplified GAT) processes the graph. Each stock (node) aggregates information from its neighbors, weighted by an attention mechanism. The attention weights signify the degree of influence one stock has on another.
4.  **Prediction & Ranking**: The model outputs a final "performance score" for each stock, which is used to generate the next-day relative performance ranking. The attention weights are surfaced in the UI to explain the results.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Git

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/rishil0208/market_interaction.git
    cd market_interaction
    ```

2.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Application

1.  **Launch the Streamlit dashboard:**
    ```bash
    streamlit run app.py
    ```
    Or, if the `streamlit` command is not in your PATH:
    ```bash
    python -m streamlit run app.py
    ```

2.  Open your web browser and navigate to the local URL provided by Streamlit.

