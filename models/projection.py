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

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 512, dropout: float = 0.1):
        """
        Initialize projection layers with optional hidden dimension.

        Args:
            input_dim (int): Dimension of encoder output embeddings.
            output_dim (int): Target dimension matching decoder input.
            hidden_dim (int): Intermediate dimension for nonlinear mapping.
        """
        super().__init__()        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Linear -> Activation -> Linear
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
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

        if x.ndim == 2:
          return self.proj(x)
        

        # x.ndim == 3
        if x.ndim == 3:
          # (B, T, D) -> flatten -> proj -> reshape
          B, T, D = x.shape
          y = self.proj(x.reshape(B * T, D))
          return y.reshape(B, T, self.output_dim)


        


__all__ = ["ProjectionLayer"]