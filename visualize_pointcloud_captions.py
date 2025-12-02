"""Utility script to visualize point clouds alongside generated captions.

Loads a trained checkpoint, samples a handful of items from the requested split,
plots their point clouds, and overlays both the generated and reference captions.
"""

import argparse
import html
import math
import random
import textwrap
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
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


def build_dataset(
    data_cfg: dict,
    split: str,
    max_samples: int | None,
    point_cloud_size: int | None,
) -> Cap3DDataset:
    cfg = dict(data_cfg) if data_cfg is not None else {}
    cfg["profile_io"] = False  # avoid verbose I/O logs when visualizing
    cfg["populate_cache"] = False  # never pre-populate the entire split here
    if max_samples is not None:
        cfg["max_samples"] = max_samples
    if point_cloud_size is not None:
        cfg["point_cloud_size"] = point_cloud_size
    return Cap3DDataset(
        hf_repo=cfg.get("hf_repo", "tiange/Cap3D"),
    hf_file=cfg.get("hf_file", "Cap3D_automated_ABO.csv"),
        split=split,
        point_cloud_size=cfg.get("point_cloud_size", 2048),
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


def format_caption_html(text: str, width: int = 48) -> str:
    formatted = format_caption(text, width)
    return "<br>".join(html.escape(line) for line in formatted.splitlines())


def save_interactive_html(path: Path, plots: List[dict], cols_per_row: int = 5) -> None:
    try:
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
    except ImportError as exc:
        raise RuntimeError(
            "Plotly is required to export interactive visualizations. Install it with 'pip install plotly'."
        ) from exc

    if not plots:
        raise ValueError("No plots available to render.")

    cols = max(1, cols_per_row)
    rows = math.ceil(len(plots) / cols)

    if cols > 1:
        max_spacing = max(0.0, (1.0 / (cols - 1)) - 1e-3)
        horizontal_spacing = min(0.12, max_spacing)
    else:
        horizontal_spacing = 0.0

    fig = make_subplots(
        rows=rows,
        cols=cols,
        specs=[[{"type": "scene"} for _ in range(cols)] for _ in range(rows)],
        horizontal_spacing=horizontal_spacing,
        vertical_spacing=0.66 if rows > 1 else 0.0,
    )

    for plot_idx, plot in enumerate(plots):
        row = (plot_idx // cols) + 1
        col = (plot_idx % cols) + 1
        coords = plot["coords"]
        colors = plot["colors"]
        if colors is None:
            marker = {"size": 2, "opacity": 0.75, "color": "#1f77b4"}
        else:
            color_arr = (np.clip(colors, 0.0, 1.0) * 255).astype(np.uint8)
            color_strings = [f"rgb({r},{g},{b})" for r, g, b in color_arr]
            marker = {"size": 2, "opacity": 0.75, "color": color_strings}

        fig.add_trace(
            go.Scatter3d(
                x=coords[:, 0],
                y=coords[:, 1],
                z=coords[:, 2],
                mode="markers",
                marker=marker,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

        scene_name = f"scene{'' if plot_idx == 0 else plot_idx + 1}"
        fig.layout[scene_name].update(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
        )

        domain = fig.layout[scene_name].domain
        if hasattr(domain, "x"):
            start_x, end_x = domain.x
        else:
            start_x, end_x = domain
        if hasattr(domain, "y"):
            start_y, end_y = domain.y
        else:
            start_y, end_y = (1.0, 1.0)
        center_x = 0.5 * (start_x + end_x)
        caption_y = max(-0.55, start_y - 0.08)

        fig.add_annotation(
            text=plot["title_html"],
            x=center_x,
            y=caption_y,
            xref="paper",
            yref="paper",
            showarrow=False,
            align="left",
            font=dict(size=10),
        )

    fig.update_layout(
        margin=dict(l=0, r=0, t=24, b=180),
        showlegend=False,
        height=500,
        width=360 * min(cols, len(plots)),
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(path, include_plotlyjs="cdn")


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

    sample_cap = args.max_samples if args.max_samples is not None else args.num_examples
    dataset = build_dataset(
        data_cfg,
        split=args.split,
        max_samples=sample_cap,
        point_cloud_size=args.point_cloud_size,
    )
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
        figsize=(3.0 * len(indices), 3.0),
        constrained_layout=True,
    )
    if len(indices) == 1:
        axes = [axes]

    gen_kwargs = {
        "max_length": train_cfg.get("max_length", 128),
        "num_beams": train_cfg.get("num_beams", 3),
    }

    plot_records: List[dict] = []

    with torch.no_grad():
        for ax, idx in zip(axes, indices):
            sample = dataset[idx]
            pts = sample["point_clouds"].to(device)
            reference = sample.get("caption", "")

            predictions = model.generate(point_clouds=pts.unsqueeze(0), **gen_kwargs)
            generated = predictions[0] if isinstance(predictions, list) else str(predictions)

            pts_np = pts.cpu().numpy()
            coords = pts_np[:, :3]
            colors = None
            rgb = None
            if pts_np.shape[1] >= 9:
                # xyznxnynzrgb
                rgb = pts_np[:, 6:9]
            elif pts_np.shape[1] >= 6:
                # xyzrgb
                rgb = pts_np[:, 3:6]

            if rgb is not None:
                if rgb.max() > 1.0:
                    rgb = rgb / 255.0
                colors = np.clip(rgb, 0.0, 1.0)

            scatter_kwargs = {"s": 1.0, "alpha": 0.75}
            if colors is not None:
                scatter_kwargs["c"] = colors
            else:
                scatter_kwargs["color"] = "#1f77b4"
            
            ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], **scatter_kwargs)
            ax.set_axis_off()

            title = (
                f"Sample #{idx}\nGenerated:\n{format_caption(generated)}\n\n"
                f"Reference:\n{format_caption(reference)}"
            )
            ax.set_title(title, fontsize=9)

            plot_records.append(
                {
                    "coords": coords,
                    "colors": colors,
                    "title_html": (
                        f"Sample #{idx}<br>"
                        f"<b>Generated:</b><br>{format_caption_html(generated)}<br><br>"
                        f"<b>Reference:</b><br>{format_caption_html(reference)}"
                    ),
                }
            )

    output_path = Path(args.output).expanduser() if args.output else None
    if output_path:
        interactive_path = output_path.with_suffix(".html")
    else:
        interactive_path = Path.cwd() / "interactive_visualization.html"

    should_show = output_path is None

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        print(f"Visualization saved to {output_path}")
    save_interactive_html(interactive_path, plot_records)
    print(f"Interactive visualization saved to {interactive_path}")
    if should_show:
        plt.show()
    else:
        plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize point clouds with generated captions")
    parser.add_argument("--config", default="configs/training_config.yaml", help="Path to YAML config")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="Dataset split")
    parser.add_argument("--num-examples", type=int, default=3, help="Number of examples to render")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on dataset samples for faster loading")
    parser.add_argument(
        "--point-cloud-size",
        type=int,
        default=None,
        help="Number of points to sample from each point cloud (overrides config)",
    )
    parser.add_argument("--device", default="auto", help="Computation device: auto|cpu|cuda[:idx]|mps")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for selecting samples")
    parser.add_argument("--output", default=None, help="Optional path to save figure instead of showing it")
    return parser.parse_args()


if __name__ == "__main__":
    visualize(parse_args())
