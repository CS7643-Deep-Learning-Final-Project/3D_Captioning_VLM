"""
projection.py
--------------
Defines the projection module that bridges visual encoder and language decoder.
Responsible for transforming encoder embeddings into decoder-compatible space.
"""

import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model

class ProjectionLayer(nn.Module):
    """
    Projection network to align encoder output with decoder input space.
    Handles dimension matching and feature transformation.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        prefix_tokens: int = 1,
    ):
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
        self.prefix_tokens = max(1, int(prefix_tokens))
        
        # Linear -> Activation -> Linear (expanded to produce prefix_tokens embeddings)
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim * self.prefix_tokens),
        )
        self.norm = nn.LayerNorm(output_dim)
        
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["0", "3"]
        )
        print(self.proj)
        # for param in self.proj.parameters():
        #     param.requires_grad = False
        self.proj = get_peft_model(self.proj, peft_config)

    def forward(self, x: torch.Tensor):
        """
        Project encoder features to decoder-compatible space.

        Args:
            x (torch.Tensor): Input tensor from encoder of shape (B, input_dim).

        Returns:
            torch.Tensor: Projected features of shape (B, prefix_tokens, output_dim)
                          (or higher rank when processing sequences), suitable for
                          conditioning the language model with multiple prefix tokens.
        """
        # Responsibilities:
        # - Apply linear or MLP projection.
        # - Normalize or activate features if needed.
        # - Optionally support dropout or layer normalization.
        
        # (B, input_dim) -> (B, output_dim)

        if x.ndim == 2:
            y = self.proj(x)  # (B, output_dim * prefix_tokens)
            y = y.view(x.size(0), self.prefix_tokens, self.output_dim)
            return self.norm(y)

        if x.ndim == 3:
            # (B, T, D) -> flatten -> proj -> reshape -> normalize
            B, T, D = x.shape
            y = self.proj(x.reshape(B * T, D))
            y = y.view(B, T, self.prefix_tokens, self.output_dim)
            return self.norm(y)

        raise ValueError(f"Unsupported input rank {x.ndim} for ProjectionLayer")

__all__ = ["ProjectionLayer"]
