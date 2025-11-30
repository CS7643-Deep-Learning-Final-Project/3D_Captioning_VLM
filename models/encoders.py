"""
encoders.py
------------
Defines abstract and concrete encoder classes for 3D captioning.
Supports modular integration of multiple 3D vision backbones (e.g., DGCNN, Point-BERT).
"""
import os
import sys
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Any
import torch.nn.functional as F
import importlib.util
from importlib import import_module

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
    Can optionally load pretrained weights from ABO for better initialization.
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
        
        if output_dim != 2048:
            raise ValueError(f"DGCNNEncoder output_dim must be 2048, got {output_dim}")
        
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
    Load weights pretrained on ABO or other 3D classification datasets.
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

try:
    this_file = os.path.abspath(__file__)
except NameError:
    this_file = os.path.abspath(os.path.join(os.getcwd(), "models", "encoders.py"))
_REPO_ROOT = os.path.dirname(os.path.dirname(this_file))
POINTBERT_ROOT = os.path.join(_REPO_ROOT, "Point-BERT")
POINTBERT_CFG = os.path.join(POINTBERT_ROOT, "cfgs", "Mixup_models", "Point-BERT.yaml")
POINTBERT_DVAE_CKPT = os.path.join(POINTBERT_ROOT, "dVAE.pth")
POINTBERT_BERT_CKPT = os.path.join(POINTBERT_ROOT, "Point-BERT.pth")
for path in (_REPO_ROOT, POINTBERT_ROOT):
    resolved = os.path.abspath(path)
    if os.path.isdir(resolved) and resolved not in sys.path:
        sys.path.insert(0, resolved)

if "Point_BERT" not in sys.modules:
    init_file = os.path.join(POINTBERT_ROOT, "__init__.py")
    spec = importlib.util.spec_from_file_location("Point_BERT", init_file, submodule_search_locations=[POINTBERT_ROOT])
    module = importlib.util.module_from_spec(spec)
    sys.modules["Point_BERT"] = module
    if spec.loader is not None:
        spec.loader.exec_module(module)

from importlib import import_module

cfg_from_yaml_file = import_module("Point_BERT.utils.config").cfg_from_yaml_file
build_model_from_cfg = import_module("Point_BERT.models.build").build_model_from_cfg
_PointBERT_module = import_module("Point_BERT.models.Point_BERT")
_PointBERT = _PointBERT_module.Point_BERT
_checkpoint_utils = import_module("Point_BERT.utils.checkpoint")
_missing_msg = getattr(_checkpoint_utils, "get_missing_parameters_message", lambda keys: str(keys))
_unexpected_msg = getattr(_checkpoint_utils, "get_unexpected_parameters_message", lambda keys: str(keys))


def _load_pointbert_checkpoint(backbone: nn.Module, ckpt_path: str):
    """Fallback loader mirroring Point-BERT's original helper."""
    ckpt = torch.load(ckpt_path, map_location="cpu")
    base_model = ckpt.get("base_model", ckpt)
    processed = {}
    for key, value in base_model.items():
        key = key.replace("module.", "")
        processed[key] = value
        if key.startswith("transformer_q") and not key.startswith("transformer_q.cls_head"):
            processed[key[len("transformer_q."):]] = value
        elif key.startswith("base_model"):
            processed[key[len("base_model."):]] = value
    incompatible = backbone.load_state_dict(processed, strict=False)
    if getattr(incompatible, "missing_keys", None):
        print("[PointBERTEncoder] Missing keys:\n" + _missing_msg(incompatible.missing_keys))
    if getattr(incompatible, "unexpected_keys", None):
        print("[PointBERTEncoder] Unexpected keys:\n" + _unexpected_msg(incompatible.unexpected_keys))

class PointBERTEncoder(BaseEncoder):
    """
    Point-BERT encoder using Transformer architecture with masked point modeling.
    Provides semantically rich features through self-supervised pretraining.
    """

    def __init__(
        self,
        output_dim: int | None = None,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        cfg_path: str | None = None,
        dvae_ckpt: str | None = None,
        pointbert_ckpt: str | None = None,
    ):
        super().__init__()
        # ------- 1. Parse configuration -------
        cfg_path = cfg_path or POINTBERT_CFG
        cfg = cfg_from_yaml_file(cfg_path)      # Returns an EasyDict configuration
        model_cfg = cfg.model                   # Model sub-config consumed by the Point-BERT backbone

        # Point-BERT expects a valid dVAE checkpoint path during _prepare_dvae
        dvae_ckpt = dvae_ckpt or POINTBERT_DVAE_CKPT
        model_cfg.dvae_config.ckpt = dvae_ckpt

        # ------- 2. Instantiate Point-BERT backbone -------
        # build_model_from_cfg dispatches to the registered Point_BERT class via model_cfg.NAME
        self.backbone: _PointBERT = build_model_from_cfg(model_cfg)

        # ------- 3. Load pretrained Point-BERT weights -------
        if pretrained:
            pointbert_ckpt = pointbert_ckpt or POINTBERT_BERT_CKPT
            if not os.path.isfile(pointbert_ckpt):
                raise FileNotFoundError(f"Point-BERT checkpoint not found: {pointbert_ckpt}")
            # Use the helper provided by the official Point_BERT implementation
            loader_fn = getattr(self.backbone, "load_model_from_ckpt", None)
            loaded = False
            if callable(loader_fn):
                loader_fn(pointbert_ckpt)
                loaded = True
            else:
                base_cls = type(self.backbone)
                loader_fn = getattr(base_cls, "load_model_from_ckpt", None)
                if callable(loader_fn):
                    loader_fn(self.backbone, pointbert_ckpt)
                    loaded = True
                else:
                    class_loader = getattr(_PointBERT, "load_model_from_ckpt", None)
                    if callable(class_loader):
                        class_loader(self.backbone, pointbert_ckpt)
                        loaded = True
            if not loaded:
                _load_pointbert_checkpoint(self.backbone, pointbert_ckpt)
            print(f"[PointBERTEncoder] Loaded pretrained weights from {pointbert_ckpt}")

        # ------- 4. Optionally freeze backbone and record output dimensionality -------
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
            self.backbone.eval()

        # CLS token after cls_head becomes the geometric embedding dimension exposed by this encoder
        self.output_dim = self.backbone.transformer_q.cls_dim
        if output_dim is not None and output_dim != self.output_dim:
            # Keep dimensional reconciliation in the Projection module instead of forcing a local linear layer
            print(
                f"[PointBERTEncoder] Note: backbone cls_dim={self.output_dim}, "
                f"but PointBERTEncoder received output_dim={output_dim}."
            )

    def forward(self, point_cloud: torch.Tensor):
        """
        Extract global representation using Point-BERT.
        Args:
            point_cloud (torch.Tensor): Input tensor of shape (B, N, 3).
        Returns:
            torch.Tensor: Global feature embedding of shape (B, D),
                          typically using the [CLS] token representation.
        """
    device = next(self.backbone.parameters()).device
    pts_xyz = point_cloud[..., :3].contiguous()
    cls_feat = self.backbone.forward_eval(pts_xyz.to(device))   # (B, cls_dim)
        if torch.isnan(cls_feat).any():
            print("[PointBERTEncoder] Warning: forward pass produced NaNs")
        else:
            print(f"[PointBERTEncoder] Forward pass OK (batch={cls_feat.size(0)}, dim={cls_feat.size(1)})")
        return cls_feat

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
