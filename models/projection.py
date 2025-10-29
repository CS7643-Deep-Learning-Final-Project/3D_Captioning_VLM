"""
projection.py
--------------
Defines the projection module that bridges visual encoder and language decoder.
Responsible for transforming encoder embeddings into decoder-compatible space.
"""

import torch
import torch.nn as nn


class ProjectionLayer(nn.Module):
    """
    Projection network to align encoder output with decoder input space.
    Handles dimension matching and feature transformation.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 512):
        """
        Initialize projection layers with optional hidden dimension.

        Args:
            input_dim (int): Dimension of encoder output embeddings.
            output_dim (int): Target dimension matching decoder input.
            hidden_dim (int): Intermediate dimension for nonlinear mapping.
        """
        super().__init__()
        # TODO: define transformation layers (e.g., Linear + Activation)
        # Example: self.proj = nn.Sequential(...)
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Linear -> Activation -> Linear
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Project encoder features to decoder-compatible space.

        Args:
            x (torch.Tensor): Input tensor from encoder of shape (B, input_dim).

        Returns:
            torch.Tensor: Projected features of shape (B, output_dim),
                          suitable for conditioning the language model.
        """
        # Responsibilities:
        # - Apply linear or MLP projection.
        # - Normalize or activate features if needed.
        # - Optionally support dropout or layer normalization.
        
        # (B, input_dim) -> (B, output_dim)
        return self.proj(x)


__all__ = ["ProjectionLayer"]