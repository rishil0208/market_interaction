"""
Configuration file for the Real-Time Market Interaction Monitor.
"""

# --- Ticker Basket ---
# A curated list of 10 volatile tech stocks for intraday monitoring.
TICKERS = [
    'NVDA',  # NVIDIA
    'AMD',   # Advanced Micro Devices
    'AAPL',  # Apple
    'MSFT',  # Microsoft
    'GOOGL', # Alphabet
    'META',  # Meta Platforms
    'TSLA',  # Tesla
    'AMZN',  # Amazon
    'INTC',  # Intel
    'QCOM'   # Qualcomm
]

# --- Modeling Hyperparameters ---
# Lookback window in minutes for calculating rolling features and correlations.
LOOKBACK_MINUTES = 60

# Correlation threshold for establishing a "live" edge in the interaction graph.
CORRELATION_THRESHOLD = 0.5 # Lowered for intraday data as correlations are weaker

# --- UI & Theming ---
# Custom CSS for a professional, "trading floor" look.
CUSTOM_CSS = """
    <style>
        /* Main page background */
        .main {
            background-color: #0A0A0A;
        }
        /* Custom metric card styling for the top ticker tape */
        .stMetric {
            background-color: #1E1E1E;
            border-radius: 10px;
            padding: 10px;
            border: 1px solid #333;
        }
        /* Center align metric labels */
        .stMetric-label {
             font-size: 0.9em;
             font-weight: bold;
             text-transform: uppercase;
        }
        /* Style the metric values */
        .stMetric-value {
             font-size: 1.5em !important;
        }
        /* Style the metric delta (price change) */
        .stMetric-delta {
            font-size: 1.1em !important;
            font-weight: bold;
        }
    </style>
"""