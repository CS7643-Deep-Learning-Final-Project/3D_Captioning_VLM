"""Utility script to visualize point clouds alongside generated captions.

Loads a trained checkpoint, samples a handful of items from the requested split,
plots their point clouds, and overlays both the generated and reference captions.
"""

import argparse
import random
import textwrap
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import torch
import yaml

from data.data_loader import Cap3DDataset
from models import CaptionModel


def load_config(path: Path) -> dict:
    with path.open("r") as fh:
        return yaml.safe_load(fh) or {}


def select_device(preference: str) -> torch.device:
    pref = preference.lower()
    if pref == "cpu":
        return torch.device("cpu")
    if pref.startswith("cuda") and torch.cuda.is_available():
        return torch.device(pref if ":" in pref else "cuda:0")
    if pref == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def build_dataset(data_cfg: dict, split: str, max_samples: int | None) -> Cap3DDataset:
    cfg = dict(data_cfg) if data_cfg is not None else {}
    cfg.setdefault("profile_io", False)
    cfg.setdefault("populate_cache", False)
    if max_samples is not None:
        cfg["max_samples"] = max_samples
    return Cap3DDataset(
        hf_repo=cfg.get("hf_repo", "tiange/Cap3D"),
        hf_file=cfg.get("hf_file", "Cap3D_automated_ShapeNet.csv"),
        split=split,
        point_cloud_size=cfg.get("point_cloud_size", 1024),
        tokenizer=None,
        profile_io=bool(cfg.get("profile_io", False)),
        profile_every=int(cfg.get("profile_every", 50)),
        use_cache=bool(cfg.get("use_cache", False)),
        cache_dir=cfg.get("cache_dir"),
        populate_cache=bool(cfg.get("populate_cache", False)),
        max_samples=max_samples,
    )


def load_model(model_cfg: dict, checkpoint: Path, device: torch.device) -> CaptionModel:
    model = CaptionModel(model_cfg)
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def choose_indices(total: int, count: int, seed: int | None) -> List[int]:
    count = min(count, total)
    rng = random.Random(seed)
    if count <= 0:
        return []
    if total <= count:
        return list(range(total))
    return rng.sample(range(total), count)


def format_caption(text: str, width: int = 48) -> str:
    wrapped = textwrap.fill(text.strip(), width=width)
    return wrapped if wrapped else "<empty>"


def visualize(args: argparse.Namespace) -> None:
    config_path = Path(args.config).expanduser()
    checkpoint_path = Path(args.checkpoint).expanduser()

    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    config = load_config(config_path)
    data_cfg = config.get("data", {})
    model_cfg = config.get("model", {})
    train_cfg = config.get("training", {})

    device = select_device(args.device)

    dataset = build_dataset(data_cfg, split=args.split, max_samples=args.max_samples)
    if len(dataset) == 0:
        raise RuntimeError(f"No samples available for split '{args.split}'.")

    model = load_model(model_cfg, checkpoint_path, device)

    indices = choose_indices(len(dataset), args.num_examples, args.seed)
    if not indices:
        print("No indices selected; nothing to visualize.")
        return

    fig, axes = plt.subplots(
        1,
        len(indices),
        subplot_kw={"projection": "3d"},
        figsize=(6 * len(indices), 6),
        constrained_layout=True,
    )
    if len(indices) == 1:
        axes = [axes]

    gen_kwargs = {
        "max_length": train_cfg.get("max_length", 128),
        "num_beams": train_cfg.get("num_beams", 3),
    }

    with torch.no_grad():
        for ax, idx in zip(axes, indices):
            sample = dataset[idx]
            pts = sample["point_clouds"].to(device)
            reference = sample.get("caption", "")

            predictions = model.generate(point_clouds=pts.unsqueeze(0), **gen_kwargs)
            generated = predictions[0] if isinstance(predictions, list) else str(predictions)

            pts_np = pts.cpu().numpy()
            ax.scatter(pts_np[:, 0], pts_np[:, 1], pts_np[:, 2], s=1.0, alpha=0.75)
            ax.set_axis_off()

            title = (
                f"Sample #{idx}\nGenerated:\n{format_caption(generated)}\n\n"
                f"Reference:\n{format_caption(reference)}"
            )
            ax.set_title(title, fontsize=9)

    output_path = Path(args.output).expanduser() if args.output else None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        print(f"Visualization saved to {output_path}")
    else:
        plt.show()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize point clouds with generated captions")
    parser.add_argument("--config", default="configs/training_config.yaml", help="Path to YAML config")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--num-examples", type=int, default=3, help="Number of examples to render")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on dataset samples for faster loading")
    parser.add_argument("--device", default="auto", help="Computation device: auto|cpu|cuda[:idx]|mps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for selecting samples")
    parser.add_argument("--output", default=None, help="Optional path to save figure instead of showing it")
    return parser.parse_args()


if __name__ == "__main__":
    visualize(parse_args())
