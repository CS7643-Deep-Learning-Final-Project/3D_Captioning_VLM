# main.py
import argparse
import os
import random
from typing import Any, Dict, Tuple
import numpy as np
import torch
import yaml
from data import DataModule
from models import CaptionModel
from training.trainer import Trainer
from evaluation import CaptionEvaluator

# ---- utils ----
def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _select_device(pref: str = "auto"):
    if pref.lower() == "cpu":
        return torch.device("cpu")
    if pref.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(pref if ":" in pref else "cuda:0")
        print("[warn] CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    # auto
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --- 1. Load configuration ---
def load_config(config_path: str):
    """Load training configuration from YAML file."""
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


# --- 2. Environment setup ---
def setup_environment(config: Dict[str, Any]):
    """Initialize device, data module, and model based on configuration."""
    # Responsibilities:
    # - Select device (CUDA if available)
    # - Initialize DataModule and prepare dataloaders
    # - Build CaptionModel from config
    """Initialize device, data module, and model based on configuration."""
    seed = int(config.get("seed", 42))
    _set_seed(seed)

    device = _select_device(config.get("device", "auto"))

    # DataModule
    data_cfg = config.get("data", {})
    if hasattr(DataModule, "from_config"):
        datamodule = DataModule.from_config(data_cfg)
    else:
        datamodule = DataModule(**data_cfg) if isinstance(data_cfg, dict) else DataModule()
    # allow either .setup() or .prepare_data()/.setup(stage="fit")
    if hasattr(datamodule, "setup"):
        try:
            datamodule.setup(stage="fit")
        except TypeError:
            datamodule.setup()
    elif hasattr(datamodule, "prepare_data"):
        datamodule.prepare_data()

    # Model
    model_cfg = config.get("model", {})
    if hasattr(CaptionModel, "from_config"):
        model = CaptionModel.from_config(model_cfg)
    else:
        model = CaptionModel(**model_cfg) if isinstance(model_cfg, dict) else CaptionModel()

    model.to(device)
    return device, datamodule, model


# --- 3. Main execution pipeline ---
def main():
    os.chdir(os.path.dirname(__file__))
    """Main execution function coordinating the full training pipeline."""
    # 1. Load config
    # 2. Setup environment (device, data, model)
    # 3. Initialize Trainer and Evaluator
    # 4. Run training with periodic validation and checkpointing
    """Main execution function coordinating the full training pipeline."""
    parser = argparse.ArgumentParser(description="Train captioning model")
    parser.add_argument(
        "-c", "--config", default="configs/train_config.yaml", help="Path to YAML config"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    device, datamodule, model = setup_environment(config)

    # Dataloaders
    if hasattr(datamodule, "get_dataloaders"):
        train_loader, val_loader = datamodule.get_dataloaders()
    else:
        # fallbacks
        train_loader = getattr(datamodule, "train_dataloader")()
        val_loader = getattr(datamodule, "val_dataloader")()

    # Trainer
    # API per your example: Trainer(model, train_loader, val_loader, config, device)
    trainer = Trainer(model, train_loader, val_loader, config, device)

    # Evaluator (allow kwargs if provided)
    eval_cfg = config.get("evaluator", {})
    try:
        evaluator = CaptionEvaluator(**eval_cfg) if isinstance(eval_cfg, dict) else CaptionEvaluator()
    except TypeError:
        evaluator = CaptionEvaluator()

    trainer.train(evaluator)
    
if __name__ == "__main__":
    main()