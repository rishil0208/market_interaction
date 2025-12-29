# Market Interaction-Based Performance Predictor (Real-Time Edition)

## 📖 Overview

This project is an educational, real-time dashboard that predicts short-term stock performance by modeling the market as a dynamic graph. It is designed not only to provide predictions but also to **teach the user** the fundamental concepts behind its operation, such as graph neural networks and attention mechanisms.

The dashboard simulates a "trading floor" experience, fetching live 1-minute data from Yahoo Finance and automatically updating every 60 seconds.

---

## 🏛️ Architecture Overview

The application follows a clear, linear data flow from data source to final prediction, designed for educational clarity:

1.  **Data Engine (`data_engine.py`)**:
    *   Fetches the last 7 days of 1-minute interval stock data from the `yfinance` API.
    *   Calculates dynamic features for each stock based on the last 60 minutes of data (e.g., momentum, volatility).
    *   Constructs a **Dynamic Graph** by calculating a 60-minute rolling correlation matrix. An "edge" exists between two stocks if their correlation exceeds a set threshold.

2.  **Model Builder (`model_builder.py`)**:
    *   The feature set and graph structure are fed into a **Graph Attention Network (GAT)** built with PyTorch.
    *   The GAT calculates a "performance score" for each stock by learning how to weight the influence of its neighbors.
    *   Crucially, the model outputs both the **final scores** and the **attention weights**, which represent the "focus" of the model and provide a basis for explaining its decisions.

3.  **Dashboard (`app.py`)**:
    *   The Streamlit frontend orchestrates the process, refreshing every 60 seconds.
    *   It calls the data engine and model builder to get fresh predictions.
    *   It visualizes all outputs for the user:
        *   The dynamic graph shows which stocks are currently correlated.
        *   The attention heatmap shows the raw "focus" of the GAT model.
        *   A leaderboard ranks the stocks based on their predicted short-term performance.
    *   An educational sidebar explains the key concepts to the user in plain English.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Git

### Installation & Setup

1.  **Clone the repository:**
    ```bash
    git clone <YOUR_REPOSITORY_URL>
    cd <repository_folder>
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

2.  Open your web browser and navigate to the local URL provided by Streamlit. The dashboard will automatically start fetching data and updating.