"""
encoders.py
------------
Defines abstract and concrete encoder classes for 3D captioning.
Supports modular integration of multiple 3D vision backbones (e.g., DGCNN, Point-BERT).
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Any


class BaseEncoder(ABC, nn.Module):
    """
    Abstract base class for all 3D encoders.
    Ensures consistent interface for different encoder architectures.
    """

    def __init__(self):
        """Initialize the base encoder."""
        super().__init__()

    @abstractmethod
    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """
        Process input point cloud and return feature embeddings.

        Args:
            point_cloud (torch.Tensor): Input tensor of shape (B, N, 3 or more).

        Returns:
            torch.Tensor: Encoded feature representation of shape (B, D).
        """
        pass

    @abstractmethod
    def get_output_dim(self) -> int:
        """
        Return output feature dimension of the encoder.
        Used for configuring downstream projection or language models.

        Returns:
            int: Dimension of the output embedding (D).
        """
        pass


class DGCNNEncoder(BaseEncoder):
    """
    DGCNN-based encoder using Dynamic Graph CNN for point cloud feature extraction.
    Can optionally load pretrained weights from ShapeNet for better initialization.
    """

    def __init__(self, output_dim: int = 768, k: int = 20, pretrained: bool = True):
        """
        Initialize the DGCNN encoder.

        Args:
            output_dim (int): Dimension of the output embedding.
            k (int): Number of nearest neighbors for graph construction.
            pretrained (bool): Whether to load pretrained weights.
        """
        super().__init__()
        # Define model layers and parameters here (to be implemented by vision team)
        pass

    def load_pretrained_weights(self) -> None:
        """
        Load weights pretrained on ShapeNet or other 3D classification datasets.
        Useful for faster convergence and better generalization.
        """
        pass

    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """
        Extract features using the DGCNN architecture.

        Args:
            point_cloud (torch.Tensor): Input tensor of shape (B, N, 3).

        Returns:
            torch.Tensor: Encoded feature embeddings of shape (B, D).
        """
        pass

    def get_output_dim(self) -> int:
        """Return the output embedding dimension."""
        return self.output_dim


class PointBERTEncoder(BaseEncoder):
    """
    Point-BERT encoder using Transformer architecture with masked point modeling.
    Provides semantically rich features through self-supervised pretraining.
    """

    def __init__(self, output_dim: int = 768, pretrained: bool = True, freeze_backbone: bool = True):
        """
        Initialize the Point-BERT encoder.

        Args:
            output_dim (int): Dimension of the encoder output features.
            pretrained (bool): Whether to load pretrained model weights.
            freeze_backbone (bool): If True, freeze backbone parameters during training.
        """
        super().__init__()
        # Define model loading and initialization here (to be implemented by vision team)
        pass

    def forward(self, point_cloud: torch.Tensor) -> torch.Tensor:
        """
        Extract global representation using Point-BERT.

        Args:
            point_cloud (torch.Tensor): Input tensor of shape (B, N, 3).

        Returns:
            torch.Tensor: Global feature embedding of shape (B, D),
                          typically using the [CLS] token representation.
        """
        pass

    def get_output_dim(self) -> int:
        """Return the output embedding dimension."""
        return self.output_dim


class EncoderFactory:
    """
    Factory class for creating encoder instances based on configuration.
    Simplifies model selection and initialization.
    """

    @staticmethod
    def create_encoder(encoder_type: str, output_dim: int = 768, **kwargs) -> BaseEncoder:
        """
        Create encoder instance of the specified type.

        Args:
            encoder_type (str): Encoder architecture type ('dgcnn' or 'pointbert').
            output_dim (int): Output embedding dimension.
            **kwargs: Additional arguments passed to encoder constructor.

        Returns:
            BaseEncoder: Instantiated encoder object.
        """
        encoder_type = encoder_type.lower()
        if encoder_type == "dgcnn":
            return DGCNNEncoder(output_dim=output_dim, **kwargs)
        elif encoder_type == "pointbert":
            return PointBERTEncoder(output_dim=output_dim, **kwargs)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")


__all__ = ["BaseEncoder", "DGCNNEncoder", "PointBERTEncoder", "EncoderFactory"]
