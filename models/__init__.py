"""
models/__init__.py
------------------
Defines the end-to-end 3D captioning model.
Integrates the encoder, projection layer, and decoder into a unified architecture.
"""

import torch
import torch.nn as nn
from typing import Any, Dict, List, Optional

from .encoders import EncoderFactory, BaseEncoder
from .projection import ProjectionLayer
from .decoders import GPT2Decoder


class CaptionModel(nn.Module):
    """
    End-to-end 3D captioning model combining encoder, projection, and decoder.
    Supports both training (with teacher forcing) and inference (caption generation).
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the complete model architecture based on configuration.

        Args:
            config (Dict[str, Any]): Experiment or model configuration dictionary.
                Expected keys:
                    - model.encoder_type: str, e.g. "pointbert" or "dgcnn"
                    - model.output_dim: int, encoder output dimension
                    - model.hidden_dim: int, optional projection hidden size
                    - model.decoder_name: str, e.g. "gpt2"
                    - model.freeze_encoder: bool
        """
        super().__init__()
        # Responsibilities:
        # - Initialize encoder via EncoderFactory
        self.encoder = EncoderFactory.create_encoder(config['encoder_type'], config['output_dim'])
        # - Initialize projection layer (input_dim=encoder_dim, output_dim=decoder_dim)
        self.projection = ProjectionLayer(input_dim=config['output_dim'], output_dim=config['embed_dim'])
        # - Initialize decoder (e.g., GPT2Decoder)
        self.decoder = GPT2Decoder(model_name=config['decoder_name'])
        # - Handle optional freezing of encoder weights

    def forward(self, point_clouds: torch.Tensor, caption: Optional[List[str]] = None) -> Any:
        """
        Complete forward pass for training and inference.

        Args:
            point_clouds (torch.Tensor): Input point clouds of shape (B, N, 3).
            caption (Optional[List[str]]): Ground-truth captions (for training).
        
        Returns:
            Any: Model output (e.g., logits, loss dict, or generated text depending on mode).
        """
        # Responsibilities:
        # - Pass point_clouds through encoder → embeddings
        f = self.encoder(point_clouds)
        # - Pass embeddings through projection → decoder space
        f = self.projection(f)
        # - If captions are provided: run decoder forward() for training
        # - If no caption: run decoder.generate() for inference
        if caption:
            return self.decoder(f, caption)
        else:
            return self.decoder(f)

    def generate(self, point_clouds: torch.Tensor, **gen_kwargs) -> List[str]:
        """
        Generate captions for given 3D inputs (inference only).

        Args:
            point_clouds (torch.Tensor): Input point clouds of shape (B, N, 3).
            **gen_kwargs: Additional generation parameters for the decoder (e.g., beam size).

        Returns:
            List[str]: Generated captions for each sample.
        """
        # Responsibilities:
        # - Compute visual embeddings
        # - Run decoder.generate() with projected features
        pass

    def freeze_encoder(self) -> None:
        """
        Optionally freeze encoder weights during training to reduce compute and overfitting.
        """
        # Example:
        # for param in self.encoder.parameters():
        #     param.requires_grad = False
        pass


__all__ = ["CaptionModel"]
