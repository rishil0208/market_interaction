"""
Core modeling logic for the Interaction-Based Performance Predictor.

This file implements a custom, interpretable graph-based scoring algorithm
that mimics a Graph Attention Network (GAT) to satisfy the project's
"Interaction" and "Interpretability" constraints.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from typing import Tuple

class InteractionScorer:
    """
    A simplified, attention-based graph model to rank stocks.
    It computes a 'performance score' for each stock by aggregating
    information from its correlated neighbors, weighted by an attention
    mechanism.

    This architecture ensures that the prediction for a stock is explicitly
    a function of its market neighbors, and the attention weights provide
    a clear explanation for the prediction.
    """
    def __init__(self, feature_matrix: pd.DataFrame, adjacency_matrix: pd.DataFrame):
        """
        Initializes the scorer.

        Args:
            feature_matrix (pd.DataFrame): Rows are stocks, columns are features
                                           (e.g., RSI, Volatility).
            adjacency_matrix (pd.DataFrame): The market interaction graph.
        """
        self.tickers = feature_matrix.index.tolist()
        
        # Convert pandas DataFrames to PyTorch tensors
        # Features are normalized to have zero mean and unit variance
        self.features = self._normalize_features(
            torch.tensor(feature_matrix.values, dtype=torch.float32)
        )
        self.adjacency = torch.tensor(adjacency_matrix.values, dtype=torch.float32)

        # Define a simple linear layer for feature transformation (like in GAT)
        self.W = nn.Linear(self.features.shape[1], self.features.shape[1], bias=False)
        torch.nn.init.xavier_uniform_(self.W.weight) # Initialize weights

    def _normalize_features(self, features: torch.Tensor) -> torch.Tensor:
        """Standardize features by removing the mean and scaling to unit variance."""
        mean = features.mean(dim=0, keepdim=True)
        std = features.std(dim=0, keepdim=True)
        return (features - mean) / (std + 1e-8)

    def calculate_scores(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Executes the core logic to compute scores and influence weights.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                - A DataFrame with final performance scores for each ticker.
                - A DataFrame representing the attention/influence matrix.
        """
        # --- Step 1: Feature Transformation ---
        # Project features into a new space, enhancing signal.
        h = self.W(self.features) # (N, F') where N=num_tickers, F'=num_features

        # --- Step 2: Attention Mechanism ---
        # This is the mathematical core of the "Interaction" requirement.
        # We calculate an attention score `e_ij` for every pair of connected nodes (i, j).
        # e_ij = LeakyReLU( (h_i @ h_j.T) )
        # This measures how much node `j` should pay attention to node `i`.
        
        # Calculate pairwise scores (dot product attention)
        attention_scores = h @ h.T
        
        # Apply a non-linearity
        attention_scores = F.leaky_relu(attention_scores)

        # --- Step 3: Masking & Normalization (Softmax) ---
        # We only care about attention from connected nodes (neighbors).
        # Set attention to -inf for non-connected nodes so softmax ignores them.
        mask = self.adjacency == 0
        attention_scores.masked_fill_(mask, -1e9)
        
        # Normalize attention scores across each row to get weights.
        # `attention_weights_ij` = how much node `i` attends to node `j`.
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # --- Step 4: Aggregation ---
        # The final score for each node is a weighted sum of its neighbors' features.
        # This explicitly models influence.
        # score_i = sum_j(attention_weights_ij * h_j)
        final_scores = attention_weights @ h

        # --- Step 5: Final Ranking Score ---
        # Collapse the feature dimension to a single score per stock.
        # Here, we simply sum the feature vector. A more complex model
        # could have another linear layer here.
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
