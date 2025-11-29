import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, Optional
from huggingface_hub import hf_hub_download
import pandas as pd
import pickle
import hashlib
import zipfile, io, os
import time
import re
from pathlib import Path

class Cap3DDataset(Dataset):
    """
    HF-only Cap3D dataset loader.
    Pulls captions (CSV) and references to point clouds directly from Hugging Face.
    """

    def __init__(
        self,
        hf_repo: str = "tiange/Cap3D",
        hf_file: str = "Cap3D_automated_ABO.csv",
        split: str = "train",
        point_cloud_size: int = 2048,
        tokenizer: Optional[Any] = None,
        profile_io: bool = False,
        profile_every: int = 50,
        use_cache: bool = False,
        cache_dir: Optional[str] = None,
        populate_cache: bool = False,
        max_samples: Optional[int] = None
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
        self.profile_io = profile_io
        self.profile_every = max(1, int(profile_every))
        self.use_cache = bool(use_cache)
        self.max_samples = int(max_samples) if max_samples is not None else None
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else None
        self.populate_cache_flag = bool(populate_cache)
        self._building_cache = False
        if self.use_cache:
            if self.cache_dir is None:
                self.cache_dir = Path("cache/pointclouds")
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.load_data()
        if self.use_cache and self.populate_cache_flag:
            self.populate_cache()
    
    def _get_zip_path(self, zip_name: str):
        if zip_name not in self.zip_cache:
            t0 = time.perf_counter() if self.profile_io else None
            self.zip_cache[zip_name] = hf_hub_download(
                repo_id=self.hf_repo,
                filename=f"PointCloud_zips_ABO/{zip_name}",
                repo_type="dataset"
            )
            if self.profile_io:
                elapsed = time.perf_counter() - t0 if t0 is not None else 0.0
                print(f"[Cap3DDataset] downloaded {zip_name} in {elapsed:.2f}s -> {self.zip_cache[zip_name]}")
            else:
                print("Downloaded zip path:", self.zip_cache[zip_name])
        return self.zip_cache[zip_name]

    def _cache_path(self, uid: str) -> Optional[Path]:
        if not self.use_cache or self.cache_dir is None:
            return None
        fname = f"{uid}_n{self.point_cloud_size}.pt"
        return self.cache_dir / fname

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
                                repo_type="dataset")
        df = pd.read_csv(csv_path, header=None, names=["uid", "category", "caption"], dtype=str)

        # single zip file
        zip_name = "compressed_pcs_00.zip"

        all_samples = []
        for _, r in df.iterrows():
            uid = r["uid"]
            caption = (r["caption"] or "").strip()
            if caption:
                pieces = re.split(r"(?<=[.!?])\s+", caption, maxsplit=1)
                caption = pieces[0].strip()
            member = f"{uid}.ply"  # file name inside the zip
            all_samples.append({
                "uid": uid,
                "caption": caption,
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
        if self.max_samples is not None and len(self.samples) > self.max_samples:
            original = len(self.samples)
            self.samples = self.samples[: self.max_samples]
            print(
                f"Split '{self.split}': truncating to {len(self.samples)} of {original} samples per max_samples setting"
            )

    def _filter_split(self, all_samples):
        """
        Stable 80/10/10 by per-UID bucket: [0,.8)->train, [.8,.9)->val, [.9,1)->test.
        """
        split_name = (self.split or "").strip().lower()

        if split_name == "all":
            return [{k: v for k, v in s.items() if k != "_bucket"} for s in all_samples]

        def keep(b):
            if split_name == "train":
                return b < 0.8
            if split_name == "val":
                return 0.8 <= b < 0.9
            if split_name == "test":
                return b >= 0.9
            raise ValueError(
                f"Unknown split '{self.split}'. Expected 'train', 'val', or 'test'."
            )

        out = [ {k: v for k, v in s.items() if k != "_bucket"}
                for s in all_samples if keep(s["_bucket"]) ]
        return out

    def preprocess_point_cloud(self, points: np.ndarray):
        """
        Validate and sample/pad a point cloud without altering its original scale.
        Assumes points shape (N, 3[+f]); extra features are kept.
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

        # Sample/pad to fixed size without modifying coordinates
        n = pts.shape[0]
        if n == 0:
            raise ValueError("Point cloud has no valid points after filtering.")

        m = self.point_cloud_size
        replace = n < m
        idx = np.random.choice(n, m, replace=replace)
        return pts[idx]

    def _load_and_preprocess(self, idx: int, sample_meta: Dict[str, Any], enable_profile: bool) -> torch.Tensor:
        try:
            t_start = time.perf_counter() if enable_profile else None
            zip_path = self._get_zip_path(sample_meta["zip"])
            t_after_zip = time.perf_counter() if enable_profile else None

            with zipfile.ZipFile(zip_path) as zf:
                names = set(zf.namelist())
                member = sample_meta["member"]
                if member not in names:
                    member = f"ABO_pcs/{member}"
                    if member not in names:
                        raise FileNotFoundError(f"{sample_meta['member']} not found in {zip_path}")

                with zf.open(member) as f:
                    data_bytes = f.read()

            t_after_read = time.perf_counter() if enable_profile else None

            from plyfile import PlyData
            v = PlyData.read(io.BytesIO(data_bytes))["vertex"].data

            cols = [c for c in ("x","y","z","nx","ny","nz","red","green","blue") if c in v.dtype.names]
            if not {"x","y","z"}.issubset(cols):
                raise ValueError("PLY missing XYZ coordinates.")
            pts = np.column_stack([v[c] for c in cols]).astype(np.float32)

            pts = self.preprocess_point_cloud(pts)
            pts_tensor = torch.from_numpy(pts)
            t_after_preprocess = time.perf_counter() if enable_profile else None

            if enable_profile and t_start is not None:
                zip_time = (t_after_zip - t_start) if (t_after_zip is not None) else 0.0
                read_time = (t_after_read - (t_after_zip or t_start)) if t_after_read is not None else 0.0
                preprocess_time = (t_after_preprocess - (t_after_read or t_after_zip or t_start)) if t_after_preprocess is not None else 0.0
                total_time = (t_after_preprocess - t_start) if (t_after_preprocess is not None) else 0.0
                print(
                    f"[Cap3DDataset] idx={idx} zip={os.path.basename(zip_path)} "
                    f"open={zip_time:.3f}s read={read_time:.3f}s preprocess={preprocess_time:.3f}s total={total_time:.3f}s"
                )

            return pts_tensor
        except Exception as e:
            raise RuntimeError(f"Failed uid={sample_meta.get('uid')} member={sample_meta.get('member')}: {e}")

    def __getitem__(self, idx: int):
        """Fetch a sample, using on-disk cache when enabled."""
        sample_meta = self.samples[idx]
        enable_profile = self.profile_io and (idx % self.profile_every == 0)
        cache_path = self._cache_path(sample_meta["uid"])

        if cache_path is not None and cache_path.exists():
            try:
                t0 = time.perf_counter() if enable_profile else None
                pts_tensor = torch.load(cache_path, map_location="cpu", weights_only=True)
                if enable_profile and t0 is not None:
                    elapsed = time.perf_counter() - t0
                    print(f"[Cap3DDataset] idx={idx} cache_hit {cache_path.name} load={elapsed:.3f}s")
                return {"point_clouds": pts_tensor, "caption": sample_meta["caption"]}
            except Exception as cache_err:
                if enable_profile:
                    print(f"[Cap3DDataset] cache load failed for {cache_path}: {cache_err}; regenerating")
                try:
                    cache_path.unlink()
                except OSError:
                    pass

        cache_missing = cache_path is not None and (not cache_path.exists())
        cache_miss_is_error = (
            cache_missing and self.use_cache and self.populate_cache_flag and not self._building_cache
        )

        if cache_miss_is_error:
            raise RuntimeError(
                "Cache miss detected during training even though populate_cache=True."
            )

        pts_tensor = self._load_and_preprocess(idx, sample_meta, enable_profile)

        if cache_path is not None:
            try:
                tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
                torch.save(pts_tensor, tmp_path)
                tmp_path.replace(cache_path)
            except Exception as cache_err:
                if enable_profile:
                    print(f"[Cap3DDataset] cache save failed for {cache_path}: {cache_err}")

        return {"point_clouds": pts_tensor, "caption": sample_meta["caption"]}

    def populate_cache(self, max_items: Optional[int] = None) -> None:
        if not self.use_cache or self.cache_dir is None:
            print("[Cap3DDataset] populate_cache skipped: caching disabled.")
            return

        total = len(self.samples)
        limit = total if max_items is None else min(total, max_items)
        print(f"[Cap3DDataset] Populating cache for {limit}/{total} samples in '{self.split}' split...")
        t0 = time.perf_counter()
        progress_interval = max(1, limit // 10)
        self._building_cache = True
        try:
            for idx in range(limit):
                sample_meta = self.samples[idx]
                cache_path = self._cache_path(sample_meta["uid"])
                if cache_path is not None and cache_path.exists():
                    if idx % progress_interval == 0:
                        pct = (idx / max(1, limit)) * 100.0
                        print(f"  - {idx}/{limit} cached ({pct:.1f}%) (hit)")
                    continue
                sample = self.__getitem__(idx)
                if idx % progress_interval == 0:
                    pct = (idx / max(1, limit)) * 100.0
                    print(f"  - {idx}/{limit} cached ({pct:.1f}%)")
                del sample
        finally:
            self._building_cache = False
        elapsed = time.perf_counter() - t0
        print(f"[Cap3DDataset] Cache population complete in {elapsed:.2f}s")
    
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
        self.cfg = config if config is not None else {}
        if isinstance(self.cfg, dict) and "data" in self.cfg:
            self.data_cfg = self.cfg.get("data", {})
        elif isinstance(self.cfg, dict):
            self.data_cfg = self.cfg
        else:
            self.data_cfg = {}
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
        d = self.data_cfg
        shared = dict(
            hf_repo=d.get("hf_repo", "tiange/Cap3D"),
            hf_file=d.get("hf_file", "Cap3D_automated_ABO.csv"),
            point_cloud_size=d.get("point_cloud_size", 2048),
            tokenizer=self.tokenizer,
            profile_io=bool(d.get("profile_io", False)),
            profile_every=int(d.get("profile_every", 50)),
            use_cache=bool(d.get("use_cache", False)),
            cache_dir=d.get("cache_dir"),
            populate_cache=bool(d.get("populate_cache", False)),
        )
        base_limit = d.get("max_samples")
        train_limit = d.get("train_max_samples", base_limit)
        val_limit = d.get("val_max_samples", base_limit)

        self.train_dataset = Cap3DDataset(
            split=d.get("split_train", "train"), max_samples=train_limit, **shared
        )
        self.val_dataset = Cap3DDataset(
            split=d.get("split_val", "val"), max_samples=val_limit, **shared
        )

    def get_dataloaders(self):
        """
        Return train and validation dataloaders with specified batch size.

        Responsibilities:
            - Wrap datasets with PyTorch DataLoader.
            - Configure batch size, shuffle, and num_workers.
        """
        if self.train_dataset is None or self.val_dataset is None:
            self.setup_datasets()

        d = self.data_cfg
        bs = d.get("batch_size", 16)
        nw = d.get("num_workers", 0)
        pin = torch.cuda.is_available()
        persist = bool(nw > 0)

        collate = (lambda b: cap3d_collate(b, tokenizer=self.tokenizer, max_length=64))

        train_loader = DataLoader(
            self.train_dataset, batch_size=bs, shuffle=True,
            num_workers=nw, pin_memory=pin, persistent_workers=persist,
            drop_last=True, collate_fn=collate
        )
        val_loader = DataLoader(
            self.val_dataset, batch_size=bs, shuffle=False,
            num_workers=nw, pin_memory=pin, persistent_workers=persist,
            drop_last=False, collate_fn=collate
        )
        return train_loader, val_loader
