"""
Model Builder for the Market Interaction-Based Performance Predictor.

This script defines the Graph Attention Network (GAT) using PyTorch. The GAT
is the "brain" of the operation, designed to learn and model the complex
interactions between stocks in our graph.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class GATLayer(nn.Module):
    """
    A single layer of a Graph Attention Network.

    The "educational" aspect is understanding what "attention" means in a graph.
    Imagine you are a stock (a node) in a network. You want to predict your own
    future movement. You could just look at your own historical data, but it's
    better to see what your neighbors are doing.

    But which neighbors are most important? A GAT layer learns to assign
    "attention scores" to your neighbors. If NVIDIA is your neighbor, and it's
    highly predictive of your movement, the GAT will learn to pay a lot of
    "attention" to it. The final prediction for you is a weighted average of
    your neighbors' features, where the weights are these learned attention scores.
    """
    def __init__(self, in_features: int, out_features: int):
        """
        Args:
            in_features (int): Number of input features for each node.
            out_features (int): Number of output features for each node.
        """
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # --- The Core of the GAT Layer ---
        # We define a single linear transformation that will be applied to every
        # node's feature vector. This is a learnable weight matrix (W).
        # The Math: h_i' = W * h_i
        # This projects the input features into a higher-dimensional space to
        # allow the model to learn more complex patterns.
        self.W = nn.Linear(in_features, out_features, bias=False)
        
        # We also define the learnable vector 'a' for the attention mechanism.
        # It's a simple single-layer feed-forward network.
        self.a = nn.Linear(2 * out_features, 1, bias=False)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The forward pass of the GAT layer.

        Args:
            h (torch.Tensor): The input node features. Shape: (N, in_features)
                              where N is the number of nodes (stocks).
            adj (torch.Tensor): The adjacency matrix of the graph. Shape: (N, N)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - The new node features (embeddings). Shape: (N, out_features)
                - The attention weights matrix. Shape: (N, N)
        """
        # 1. Apply the linear transformation to all nodes.
        Wh = self.W(h)  # Shape: (N, out_features)

        # 2. Compute attention scores (e_ij) for all pairs of nodes.
        # This is the most complex part. We want to compute a score for every
        # edge in the graph.
        
        # First, we create all possible pairs of node features.
        # Wh_i is repeated N times row-wise, Wh_j is repeated N times column-wise
        Wh1 = Wh.unsqueeze(1).repeat(1, Wh.size(0), 1) # Shape: (N, N, out_features)
        Wh2 = Wh.unsqueeze(0).repeat(Wh.size(0), 1, 1) # Shape: (N, N, out_features)
        
        # Concatenate them to form pairs [Wh_i || Wh_j]
        final_Wh = torch.cat([Wh1, Wh2], dim=-1) # Shape: (N, N, 2 * out_features)
        
        # Apply the learnable attention vector 'a' and a LeakyReLU activation.
        # The Math: e_ij = LeakyReLU(a^T * [W*h_i || W*h_j])
        e = F.leaky_relu(self.a(final_Wh).squeeze(-1)) # Shape: (N, N)

        # 3. Masking and Normalization (Softmax)
        # We only want to compute attention for nodes that are actually connected.
        # We create a mask where we set the attention score to a very large
        # negative number for non-adjacent nodes.
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        # The softmax function then turns these scores into a probability
        # distribution. The large negative numbers become ~0.
        # The Math: alpha_ij = softmax_j(e_ij)
        attention_weights = F.softmax(attention, dim=1) # Shape: (N, N)

        # 4. Aggregate neighbor features
        # The new feature for each node is a weighted sum of its neighbors'
        # transformed features.
        # The Math: h_i' = sigma(sum_{j in N_i} alpha_ij * W * h_j)
        h_prime = torch.matmul(attention_weights, Wh) # Shape: (N, out_features)

        return h_prime, attention_weights

class GATModel(nn.Module):
    """A simple GAT model with one GAT layer."""
    def __init__(self, in_features: int, out_features: int):
        super(GATModel, self).__init__()
        self.gat_layer = GATLayer(in_features, out_features)

    def forward(self, features: torch.Tensor, adj: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the entire model.

        Args:
            features (torch.Tensor): The input node features.
            adj (torch.Tensor): The adjacency matrix.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - A 1D tensor of final performance scores for each node.
                - The attention weights matrix from the GAT layer.
        """
        # Pass data through the GAT layer.
        h_prime, attention_weights = self.gat_layer(features, adj)

        # For our final prediction, we'll simply sum the output features
        # to get a single "performance score" for each stock.
        # A more complex model could have more layers or a different output head.
        final_scores = torch.sum(h_prime, dim=1)
        
        return final_scores, attention_weights
