"""
Configuration file for the Market Interaction-Based Performance Predictor.
"""

# --- Ticker Basket ---
# A curated list of 12 highly interconnected tech companies.
TICKERS = [
    'AAPL',  # Apple
    'MSFT',  # Microsoft
    'GOOGL', # Alphabet
    'AMZN',  # Amazon
    'NVDA',  # NVIDIA
    'META',  # Meta Platforms
    'TSM',   # Taiwan Semiconductor
    'AVGO',  # Broadcom
    'ORCL',  # Oracle
    'CRM',   # Salesforce
    'AMD',   # Advanced Micro Devices
    'QCOM',  # Qualcomm
]

# --- Modeling Hyperparameters ---
# Lookback window for calculating rolling features and correlations.
LOOKBACK_DAYS = 30

# Correlation threshold for establishing an "edge" in the interaction graph.
# A value of 0.6 means a strong positive linear relationship is required.
CORRELATION_THRESHOLD = 0.6

# --- UI & Theming ---
# Custom CSS for a professional, modern look.
# Inspired by financial terminals and cyberpunk aesthetics.
CUSTOM_CSS = """
    <style>
        /* Main page background */
        .main {
            background-color: #0A0A0A;
        }
        /* Bigger, bolder titles */
        h1 {
            font-size: 2.5em;
            font-weight: 700;
            letter-spacing: -1px;
            border-bottom: 2px solid #00f900;
            padding-bottom: 10px;
        }
        /* Custom metric card styling */
        .stMetric {
            background-color: #1E1E1E;
            border-radius: 10px;
            padding: 15px;
            border: 1px solid #333;
        }
        /* Green/Red metric delta colors */
        .stMetric .st-ax {
            color: #FAFAFA !important;
        }
        /* Center align metric labels */
        .stMetric-label {
             text-align: center;
             font-weight: bold;
             text-transform: uppercase;
        }
        /* Center align metric values */
        .stMetric-value {
             text-align: center;
             font-size: 2.2em !important;
        }
    </style>
"""
