import os
import time
import torch
from data.data_loader import Cap3DDataset, DataModule

# Optional: faster HF downloads
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")


def smoke_single_item():
    print("=== Single item load ===")
    ds = Cap3DDataset(
        hf_repo="tiange/Cap3D",
        hf_file="Cap3D_automated_ABO.csv",
        split="train",
        point_cloud_size=512,  # smaller for quick test
    )

    if not len(ds):
        raise ValueError("Dataset has zero samples — check CSV path or repo access.")

    t0 = time.time()
    sample = ds[0]
    dt = time.time() - t0

    pts = sample["points"]
    print(f"Caption: {sample['caption'][:80]}...")
    print(f"Points shape={tuple(pts.shape)}, dtype={pts.dtype}, load_time={dt:.2f}s")
    print(f"XYZ range: min={pts[:, :3].min():.3f}, max={pts[:, :3].max():.3f}, mean_norm≈{pts[:, :3].norm(dim=1).mean():.3f}")


def smoke_dataloader():
    print("\n=== DataLoader batch ===")
    cfg = {
        "data": {
            "hf_repo": "tiange/Cap3D",
            "hf_file": "Cap3D_automated_ABO.csv",
            "split_train": "train",
            "split_val": "val",
            "point_cloud_size": 512,
            "batch_size": 2,
            "num_workers": 0,  # safer for first test
        }
    }

    dm = DataModule(cfg)
    train_loader, _ = dm.get_dataloaders()

    t0 = time.time()
    batch = next(iter(train_loader))
    dt = time.time() - t0

    pts, caps = batch["points"], batch["caption"]
    print(f"Batch: points={tuple(pts.shape)} (B,N,F), load_time={dt:.2f}s")
    print(f"Caption[0]: {caps[0][:80]}...")


if __name__ == "__main__":
    try:
        smoke_single_item()
        smoke_dataloader()
        print("\nSmoke tests passed.")
    except Exception as e:
        print(f"\nSmoke test failed: {e}")
        raise