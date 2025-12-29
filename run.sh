#!/bin/bash
# This script installs the necessary Python packages and launches the Streamlit application.

echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "Installation complete. Launching the Real-Time Market Monitor..."
streamlit run app.py
