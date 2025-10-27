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
from models.dgcnn import DGCNN
import torch.nn.functional as F

class BaseEncoder(ABC, nn.Module):
    """
    Abstract base class for all 3D encoders.
    Ensures consistent interface for different encoder architectures.
    """

    def __init__(self):
        """Initialize the base encoder."""
        super().__init__()

    @abstractmethod
    def forward(self, point_cloud: torch.Tensor):
        """
        Process input point cloud and return feature embeddings.
        Args:
            point_cloud (torch.Tensor): Input tensor of shape (B, N, 3 or more).
        Returns:
            torch.Tensor: Encoded feature representation of shape (B, D).
        """
        pass

    @abstractmethod
    def get_output_dim(self):
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

    def __init__(self, output_dim: int = 2048, k: int = 20, pretrained: bool = True, dropout: float = 0.0):
        """
        Initialize the DGCNN encoder.
        Args:
            k (int): Number of nearest neighbors for graph construction.
            pretrained (bool): Whether to load pretrained weights.
        """
        super().__init__()
        self.k = k
        self.output_dim = output_dim
        self.act = nn.LeakyReLU(0.2, inplace=True)

        # EdgeConv layers: each processes edge features [x_j - x_i ; x_i]
        # Input channel sizes are fixed (2*C), matching the original DGCNN design.
        self.conv1 = nn.Conv2d(6, 64, kernel_size=1, bias=False)   # 2*3 = 6 input channels (xyz)
        self.bn1   = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(128, 64, kernel_size=1, bias=False)   # 2*64 = 128
        self.bn2   = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(128, 128, kernel_size=1, bias=False)  # 2*64 = 128
        self.bn3   = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(256, 256, kernel_size=1, bias=False)  # 2*128 = 256
        self.bn4   = nn.BatchNorm2d(256)

        # After concatenating multi-scale features (64+64+128+256 = 512),
        # apply a 1×1 Conv to get 1024-D global feature maps.
        self.conv5 = nn.Conv1d(512, 1024, kernel_size=1, bias=False)
        self.bn5   = nn.BatchNorm1d(1024)

        if pretrained:
            self.load_pretrained_weights()
    
    @staticmethod
    def knn(x, k):
        """
        Compute k-nearest neighbors for each point in the batch.
        Args:
            x: (B, C, N) input features.
        Returns:
            idx: (B, N, k) indices of nearest neighbors.
        """
        B, C, N = x.shape
        xt = x.transpose(2, 1)                   # (B, N, C)
        inner = -2 * torch.matmul(xt, x)         # pairwise distance term
        xx = (x ** 2).sum(dim=1, keepdim=True)   # squared norms
        dist = -xx.transpose(2, 1) - inner - xx  # full pairwise distance matrix
        k = min(k, N)
        idx = dist.topk(k=k, dim=-1)[1]          # take top-k nearest neighbors
        return idx
    
    @staticmethod
    def get_graph_feature(x, k, idx=None):
        """
        Construct edge features [x_j - x_i ; x_i] for all points.
        Args:
            x:   (B, C, N)
            k:   number of neighbors
            idx: (optional) precomputed neighbor indices
        Returns:
            edge features of shape (B, 2C, N, k)
        """
        B, C, N = x.shape
        if idx is None:
            idx = DGCNNEncoder.knn(x, k)

        device = x.device
        idx_base = torch.arange(B, device=device).view(-1, 1, 1) * N
        idx = (idx + idx_base).view(-1)

        x_t = x.transpose(2, 1).contiguous()        # (B, N, C)
        neighbors = x_t.view(B * N, C)[idx, :].view(B, N, k, C)
        x_center  = x_t.unsqueeze(2).expand(-1, -1, k, -1)
        edge = torch.cat((neighbors - x_center, x_center), dim=3)  # (B, N, k, 2C)
        return edge.permute(0, 3, 1, 2).contiguous()               # (B, 2C, N, k)

    def load_pretrained_weights(self):
        """
        Load weights pretrained on ShapeNet or other 3D classification datasets.
        Useful for faster convergence and better generalization.
        """
        try:
            checkpoint = torch.hub.load_state_dict_from_url(
                'https://github.com/antao97/dgcnn.pytorch/raw/master/pretrained/model.cls.1024.t7',
                map_location='cpu'
            )
            state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}
            self.load_state_dict(state_dict, strict=False)
            print("Successfully loaded pretrained DGCNN weights")
            
        except Exception as e:
            print(f"Could not load pretrained weights: {e}")

    def forward(self, point_cloud: torch.Tensor):
        """
        Extract features using the DGCNN architecture.
        Args:
            point_cloud (torch.Tensor): Input tensor of shape (B, N, C).
        Returns:
            torch.Tensor: Encoded feature embeddings of shape (B, D).
        """
        # Only use xyz coordinates for graph construction (as in the original DGCNN)
        x = point_cloud[..., :3].transpose(1, 2).contiguous()  # (B, 3, N)

        # ---- EdgeConv Block 1 ----
        e1 = self.get_graph_feature(x, self.k)
        f1 = self.act(self.bn1(self.conv1(e1))).max(dim=-1)[0]  # (B, 64, N)

        # ---- EdgeConv Block 2 ----
        e2 = self.get_graph_feature(f1, self.k)
        f2 = self.act(self.bn2(self.conv2(e2))).max(dim=-1)[0]  # (B, 64, N)

        # ---- EdgeConv Block 3 ----
        e3 = self.get_graph_feature(f2, self.k)
        f3 = self.act(self.bn3(self.conv3(e3))).max(dim=-1)[0]  # (B, 128, N)

        # ---- EdgeConv Block 4 ----
        e4 = self.get_graph_feature(f3, self.k)
        f4 = self.act(self.bn4(self.conv4(e4))).max(dim=-1)[0]  # (B, 256, N)

        # Concatenate multi-scale features and reduce to 1024-D
        feat = torch.cat((f1, f2, f3, f4), dim=1)               # (B, 512, N)
        feat = self.act(self.bn5(self.conv5(feat)))             # (B, 1024, N)

        # Global pooling (max + avg) → 2048-D vector
        f_max = F.adaptive_max_pool1d(feat, 1).squeeze(-1)
        f_avg = F.adaptive_avg_pool1d(feat, 1).squeeze(-1)
        global_feat = torch.cat([f_max, f_avg], dim=1)          # (B, 2048)
        
        if global_feat.size(1) != self.output_dim:
            print(f"Warning: Output dimension is 2048, but output_dim was set to {self.output_dim}")

        return global_feat

    def get_output_dim(self):
        """Return the output embedding dimension."""
        return self.output_dim


class PointBERTEncoder(BaseEncoder):
    """
    Point-BERT encoder using Transformer architecture with masked point modeling.
    Provides semantically rich features through self-supervised pretraining.
    """

    def __init__(self, output_dim: int = 2048, pretrained: bool = True, freeze_backbone: bool = True):
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

    def forward(self, point_cloud: torch.Tensor):
        """
        Extract global representation using Point-BERT.
        Args:
            point_cloud (torch.Tensor): Input tensor of shape (B, N, 3).
        Returns:
            torch.Tensor: Global feature embedding of shape (B, D),
                          typically using the [CLS] token representation.
        """
        pass

    def get_output_dim(self):
        """Return the output embedding dimension."""
        return self.output_dim


class EncoderFactory:
    """
    Factory class for creating encoder instances based on configuration.
    Simplifies model selection and initialization.
    """

    @staticmethod
    def create_encoder(encoder_type: str, output_dim: int = 2048, **kwargs):
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
