import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, Optional
from huggingface_hub import hf_hub_download
import pandas as pd
import pickle
import hashlib
import zipfile, io, os

class Cap3DDataset(Dataset):
    """
    HF-only Cap3D dataset loader.
    Pulls captions (CSV) and references to point clouds directly from Hugging Face.
    """

    def __init__(
        self,
        hf_repo: str = "tiange/Cap3D",
        hf_file: str = "Cap3D_automated_ShapeNet.csv",
        split: str = "train",
        point_cloud_size: int = 1024,
        tokenizer: Optional[Any] = None,
        config: Dict[str, Any] = {}
    ):
        """
        Args:
            hf_repo: HF dataset repo id.
            hf_file: Data file inside the repo (e.g., CSV listing uids & captions).
            split: Dataset split name (you can emulate splits via filters/slices).
            point_cloud_size: Number of points to sample per object.
            tokenizer: Tokenizer for captions.
        """
        self.hf_repo = hf_repo
        self.hf_file = hf_file
        self.split = split
        self.point_cloud_size = point_cloud_size
        self.tokenizer = tokenizer
        self.zip_cache = {}
        self.config = config
        self.load_data()
    
    def _get_zip_path(self, zip_name: str):
        if zip_name not in self.zip_cache:
            self.zip_cache[zip_name] = hf_hub_download(
                repo_id=self.hf_repo,
                filename=f"PointCloud_zips_ShapeNet/{zip_name}",
                repo_type="dataset",
                cache_dir= self.config.get('cache_dir', None),
            )
            print("Downloaded zip path:", self.zip_cache[zip_name])
        return self.zip_cache[zip_name]

    def _uid_bucket(self, uid: str):
        """Deterministic [0,1) bucket per UID, stable across runs/machines."""
        h = hashlib.md5(str(uid).encode("utf-8")).hexdigest()[:8]
        return int(h, 16) / 2**32

    def load_data(self):
        """
        Load captions and assign them to the single zip shard (compressed_pcs_00.zip).
        """
        csv_path = hf_hub_download(repo_id=self.hf_repo,
                                filename=self.hf_file,
                                repo_type="dataset",
                                cache_dir= self.config.get('cache_dir', None)
                                )
        df = pd.read_csv(csv_path, header=None, names=["uid", "caption"], dtype=str)

        # single zip file
        zip_name = "compressed_pcs_00.zip"

        all_samples = []
        for _, r in df.iterrows():
            uid = r["uid"]
            member = f"{uid}.ply"  # file name inside the zip
            all_samples.append({
                "uid": uid,
                "caption": r["caption"],
                "zip": zip_name,
                "member": member,
                "_bucket": self._uid_bucket(uid),
            })

        self.samples = self._filter_split(all_samples)
        print(f"Split '{self.split}': {len(self.samples)} / {len(all_samples)} usable samples")
        
        if len(self.samples) == 0:
            raise ValueError(f"No samples found for split '{self.split}'")

        total_original = len(all_samples)
        print(f"Split distribution: {len(self.samples)}/{total_original} ({len(self.samples)/total_original:.1%}) for '{self.split}'")

    def _filter_split(self, all_samples):
        """
        Stable 80/10/10 by per-UID bucket: [0,.8)->train, [.8,.9)->val, [.9,1)->test.
        """
        if self.split == "all":
            return [{k: v for k, v in s.items() if k != "_bucket"} for s in all_samples]

        def keep(b):
            if self.split == "train": return b < 0.8
            if self.split == "val":   return 0.8 <= b < 0.9
            if self.split == "test":  return b >= 0.9
            raise ValueError(f"Unknown split: {self.split}")

        out = [ {k: v for k, v in s.items() if k != "_bucket"}
                for s in all_samples if keep(s["_bucket"]) ]
        return out

    def preprocess_point_cloud(self, points: np.ndarray):
        """
        Center to centroid, scale to unit sphere, and sample/pad to a fixed size.
        Assumes points shape (N, 3[+f]); only XYZ are normalized; extra features are kept.
        """
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError(f"Expected (N, 3[+f]) array, got {points.shape}")

        pts = points.astype(np.float32, copy=True)
        xyz = pts[:, :3]

        # Handle NaNs/Infs gracefully
        mask = np.all(np.isfinite(xyz), axis=1)
        if not np.any(mask):
            raise ValueError("All points are non-finite.")
        pts = pts[mask]
        xyz = pts[:, :3]

        # Normalize: center and scale to unit sphere
        centroid = xyz.mean(axis=0, keepdims=True)
        xyz -= centroid
        scale = np.linalg.norm(xyz, axis=1).max()
        scale = scale if scale > 0 else 1.0
        xyz /= scale
        pts[:, :3] = xyz

        # Sample/pad to fixed size
        n = pts.shape[0]
        m = self.point_cloud_size
        idx = np.random.choice(n, m, replace=(n < m))

        return pts[idx]

    def __getitem__(self, idx: int):
        """
        Fetch one sample from the HF dataset (streaming or cached),
        load the referenced point cloud, preprocess, tokenize caption.

        Returns:
            {
              'points': torch.FloatTensor (N, 3[+f]),
              'caption': str,
              'tokens': Optional[torch.LongTensor]
            }
        """
        try:
            s = self.samples[idx]
            zip_path = self._get_zip_path(s["zip"])
            
            with zipfile.ZipFile(zip_path) as zf:
                names = set(zf.namelist())
                member = s["member"]
                if member not in names:
                    member = f"ShapeNet_pcs/{member}"
                    if member not in names:
                        raise FileNotFoundError(f"{s['member']} not found in {zip_path}")

                with zf.open(member) as f:
                    from plyfile import PlyData
                    v = PlyData.read(io.BytesIO(f.read()))["vertex"].data

            cols = [c for c in ("x","y","z","nx","ny","nz","red","green","blue") if c in v.dtype.names]
            if not {"x","y","z"}.issubset(cols):
                raise ValueError("PLY missing XYZ coordinates.")
            pts = np.column_stack([v[c] for c in cols]).astype(np.float32)

            pts = self.preprocess_point_cloud(pts)
            sample = {"point_clouds": torch.from_numpy(pts), "caption": s["caption"]}

            return sample
        
        except Exception as e:
            raise RuntimeError(f"Failed uid={s.get('uid')} member={s.get('member')}: {e}")
    
    def __len__(self):
        return len(self.samples)

def cap3d_collate(batch, tokenizer=None, max_length=64):
    points = torch.stack([b["point_clouds"] for b in batch])  # (B, N, F)
    caps   = [b["caption"] for b in batch]
    out = {"point_clouds": points, "caption": caps}

    if tokenizer is not None:
        tok = tokenizer(
            caps,
            return_tensors="pt",
            truncation=True,
            padding=True,  # pad to longest caption in batch
            max_length=max_length
        )
        out.update(tok)  # adds 'input_ids' and 'attention_mask'

    return out

class DataModule:
    """
    Data module handling train/val splits and dataloader creation.
    Provides easy access to both training and validation data.
    """

    def __init__(self, config: Dict[str, Any], tokenizer: Optional[Any] = None):
        """
        Expect config['data'] to include:
            - hf_repo, hf_file, split_train, split_val, point_cloud_size,
              batch_size, num_workers
        """
        self.cfg = config
        self.tokenizer = tokenizer
        self.train_dataset = None
        self.val_dataset = None

    def setup_datasets(self):
        """
        Create train and validation dataset instances.

        Responsibilities:
            - Instantiate Cap3DDataset for 'train' and 'val'.
            - Apply consistent preprocessing and tokenizer settings.
        """
        d = self.cfg.get("data", {})
        shared = dict(
            hf_repo=d.get("hf_repo", "tiange/Cap3D"),
            hf_file=d.get("hf_file", "Cap3D_automated_ShapeNet.csv"),
            point_cloud_size=d.get("point_cloud_size", 1024),
            tokenizer=self.tokenizer,
            config = self.cfg
        )
        self.train_dataset = Cap3DDataset(split=d.get("split_train", "train"), **shared)
        self.val_dataset   = Cap3DDataset(split=d.get("split_val", "val"), **shared)

    def get_dataloaders(self):
        """
        Return train and validation dataloaders with specified batch size.

        Responsibilities:
            - Wrap datasets with PyTorch DataLoader.
            - Configure batch size, shuffle, and num_workers.
        """
        if self.train_dataset is None or self.val_dataset is None:
            self.setup_datasets()

        d = self.cfg.get("data", {})
        bs = d.get("batch_size", 16)
        nw = d.get("num_workers", 0)
        pin = torch.cuda.is_available()
        persist = bool(nw > 0) and d.get("persistent_workers", True)
        prefetch = d.get("prefetch_factor", 4 if nw > 0 else None)

        collate = (lambda b: cap3d_collate(b, tokenizer=self.tokenizer, max_length=64))

        train_loader = DataLoader(
            self.train_dataset, batch_size=bs, shuffle=True,
            num_workers=nw, pin_memory=pin, persistent_workers=persist,
            prefetch_factor=prefetch,
            drop_last=True, collate_fn=collate
        )
        val_loader = DataLoader(
            self.val_dataset, batch_size=bs, shuffle=False,
            num_workers=nw, pin_memory=pin, persistent_workers=persist,
            prefetch_factor=prefetch,
            drop_last=False, collate_fn=collate
        )
        return train_loader, val_loader
