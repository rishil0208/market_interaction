"""
Graph Attention Network (GAT) Model for Financial Market Prediction.

This script defines the GAT architecture using PyTorch. The model is designed
to be interpretable, allowing us to extract "attention weights" which explain
which neighboring stocks were most influential in a prediction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class GATLayer(nn.Module):
    """
    A single layer of a Graph Attention Network (GAT).

    Financial Concept: In a network of stocks, not all relationships are
    equally important. A GAT layer learns to assign different levels of
    "attention" to different neighbors when making a prediction for a specific
    stock. If NVDA and AMD are highly correlated, the model might learn that
    to predict NVDA's next move, it should pay high attention to what AMD is
    doing right now. This is more advanced than a simple Graph Convolutional
    Network (GCN) which treats all neighbors equally.
    """
    def __init__(self, in_features: int, out_features: int):
        super(GATLayer, self).__init__()
        # This is the main linear transformation applied to node features.
        # It's a learnable weight matrix (W) that projects features into a
        # new space, allowing the model to capture more complex patterns.
        self.W = nn.Linear(in_features, out_features, bias=False)
        
        # This is the learnable parameter vector 'a' for the attention mechanism.
        # It's implemented as a small feed-forward network that takes the
        # concatenated features of two nodes and outputs a single score.
        self.a = nn.Linear(2 * out_features, 1, bias=False)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        The forward pass of the GAT layer.

        Args:
            h: Input node features (Shape: [Number of Stocks, Input Features]).
            adj: The adjacency matrix of the graph (Shape: [N, N]).

        Returns:
            The new node features (embeddings) after one layer of attention-
            based aggregation (Shape: [N, Output Features]).
            
        Note: This layer computes the new features 'h_prime'. The attention
        weights themselves are an intermediate product computed inside. The
        GATModel orchestrating this layer will extract them.
        """
        # 1. Apply the linear transformation to all nodes.
        Wh = self.W(h)  # Shape: [N, out_features]

        # 2. Compute attention scores (e_ij) for all pairs of nodes.
        # This creates all possible pairs [Wh_i || Wh_j] to feed into the
        # attention mechanism.
        Wh1 = Wh.unsqueeze(1).repeat(1, Wh.size(0), 1)
        Wh2 = Wh.unsqueeze(0).repeat(Wh.size(0), 1, 1)
        final_Wh = torch.cat([Wh1, Wh2], dim=-1)
        
        # Apply the attention head 'a' to get the raw attention scores.
        e = F.leaky_relu(self.a(final_Wh).squeeze(-1))

        # 3. Masking and Normalization (Softmax)
        # We mask out the attention scores for non-connected nodes by setting
        # them to a very large negative number, so they become zero after softmax.
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        # The softmax function turns the raw scores into a probability
        # distribution, giving us our final attention weights (alpha_ij).
        attention_weights = F.softmax(attention, dim=1)

        # 4. Aggregate neighbor features using the attention weights.
        # This is the core of the GAT: the new feature of a node is a
        # weighted sum of its neighbors' features.
        h_prime = torch.matmul(attention_weights, Wh)
        
        # We store the attention weights on the layer object so we can retrieve
        # them later for explainability.
        self.attention_weights = attention_weights

        return h_prime

class GATModel(nn.Module):
    """
    The full GAT model.

    This class orchestrates the GAT layers. For this educational project,
    we use a single GAT layer for maximum interpretability.
    """
    def __init__(self, in_features: int, hidden_features: int):
        super(GATModel, self).__init__()
        self.gat_layer = GATLayer(in_features, hidden_features)
        
        # An output layer to map the final node embeddings to a single score.
        self.output_layer = nn.Linear(hidden_features, 1)

    def forward(self, features: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        The forward pass for the entire model.

        Args:
            features: The input node features from the data processor.
            adj: The graph's adjacency matrix.

        Returns:
            A 1D tensor of final performance scores for each stock.
        """
        # Pass data through the GAT layer.
        h_prime = self.gat_layer(features, adj)
        
        # Apply a non-linearity.
        h_prime = F.elu(h_prime)
        
        # Get the final scores from the output layer.
        final_scores = self.output_layer(h_prime).squeeze(-1)
        
        return final_scores

    def get_attention_weights(self) -> torch.Tensor:
        """
        Extracts the attention weights from the GAT layer.

        This is the "get_attention_weights" method requested, providing a clean
        interface to access the model's "focus" for explainability in the UI.

        Returns:
            The attention weights matrix (Shape: [N, N]).
        """
        return self.gat_layer.attention_weights
