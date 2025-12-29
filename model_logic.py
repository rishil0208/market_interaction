"""
Core modeling logic for the Real-Time Market Interaction Monitor.

This file implements the same interpretable graph-based scoring algorithm
but is now fed with high-frequency features derived from intraday data
to generate short-term performance predictions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from typing import Tuple

class InteractionScorer:
    """
    A simplified, attention-based graph model to rank stocks based on
    high-frequency intraday data.

    It computes a 'short-term performance score' for each stock by aggregating
    information from its correlated neighbors, weighted by an attention
    mechanism that captures "right now" market dynamics.
    """
    def __init__(self, feature_matrix: pd.DataFrame, adjacency_matrix: pd.DataFrame):
        """
        Initializes the scorer.

        Args:
            feature_matrix (pd.DataFrame): Rows are stocks, columns are real-time
                                           features (e.g., 60m Momentum, Volatility).
            adjacency_matrix (pd.DataFrame): The live market interaction graph.
        """
        if feature_matrix.empty or adjacency_matrix.empty:
            raise ValueError("Feature or adjacency matrix cannot be empty.")
            
        self.tickers = feature_matrix.index.tolist()
        
        # Convert pandas DataFrames to PyTorch tensors
        self.features = self._normalize_features(
            torch.tensor(feature_matrix.values, dtype=torch.float32)
        )
        self.adjacency = torch.tensor(adjacency_matrix.values, dtype=torch.float32)

        # Define a simple linear layer for feature transformation
        # This layer learns to weigh the importance of different features (momentum vs. volatility)
        self.W = nn.Linear(self.features.shape[1], self.features.shape[1], bias=False)
        torch.nn.init.xavier_uniform_(self.W.weight) # Initialize weights

    def _normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """Standardize features to have zero mean and unit variance."""
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True)
        # Add a small epsilon to prevent division by zero
        return (features - mean) / (std + 1e-8)

    def calculate_scores(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Executes the core logic to compute scores and influence weights.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                - A DataFrame with final short-term performance scores for each ticker.
                - A DataFrame representing the attention/influence matrix.
        """
        # --- Step 1: Feature Transformation ---
        h = self.W(self.features)

        # --- Step 2: Attention Mechanism ---
        # Pairwise dot product attention to see how features align.
        attention_scores = h @ h.T
        attention_scores = F.leaky_relu(attention_scores)

        # --- Step 3: Masking & Normalization (Softmax) ---
        # We only care about attention from *currently* correlated stocks.
        mask = self.adjacency == 0
        attention_scores.masked_fill_(mask, -1e9)
        
        # Normalize to get attention weights.
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # --- Step 4: Aggregation ---
        # The final score for each node is a weighted sum of its neighbors' features.
        final_scores = attention_weights @ h

        # --- Step 5: Final Ranking Score ---
        # Collapse the feature dimension to a single score per stock.
        ranking_scores = final_scores.sum(dim=1).detach().numpy()

        # Create human-readable DataFrames for the UI
        score_df = pd.DataFrame({
            'ticker': self.tickers,
            'score': ranking_scores
        }).sort_values('score', ascending=False).set_index('ticker')

        attention_df = pd.DataFrame(
            attention_weights.detach().numpy(),
            index=self.tickers,
            columns=self.tickers
        )

        return score_df, attention_df