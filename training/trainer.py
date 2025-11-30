"""
trainer.py
-----------
Defines the Trainer class that manages end-to-end training and validation loops.
Handles checkpointing, progress tracking, and metric evaluation.
"""

import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math
from typing import Any, Dict, List, Union, Optional
# from torch.profiler import profile, ProfilerActivity


class Trainer:
    """
    Training manager handling training loops, validation, and model checkpointing.
    Implements standard training procedures with progress tracking.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict[str, Any],
        device: torch.device
    ):
        """
        Initialize trainer with model, data, and training configuration.
        Args:
            model (nn.Module): CaptionModel combining encoder, projection, and decoder.
            train_loader (DataLoader): Training dataset loader.
            val_loader (DataLoader): Validation dataset loader.
            config (Dict[str, Any]): Experiment/training configuration dictionary.
            device (torch.device): Device to run model on (e.g., 'cuda' or 'cpu').
        """
        # Responsibilities:
        # - Move model to device
        # - Initialize optimizer, scheduler, and loss function from config
        # - Store loaders and training parameters
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        train_cfg = config.get("training", {})
        eval_cfg = config.get("evaluation", {})
        
        # core training hyperparameters
        self.lr = train_cfg.get("learning_rate", 1e-4)
        self.epochs = train_cfg.get("num_epochs", 10)
        self.gen_max_length = train_cfg.get("max_length", 128)
        self.eval_every = 1  # evaluate once per epoch by default
        self.metrics = eval_cfg.get("metrics", ["cider"])

        # fixed or optional params (could also move to YAML later)
        self.optimizer_name = train_cfg.get("optimizer", "adamw")
        self.weight_decay = train_cfg.get("weight_decay", 0.01)
        self.scheduler_name = train_cfg.get("scheduler", "cosine")
        self.step_size = train_cfg.get("step_size", 1)
        self.gamma = train_cfg.get("gamma", 0.9)
        self.grad_accum_steps = train_cfg.get("grad_accum_steps", 1)
        self.max_norm = train_cfg.get("max_norm", 1.0)
        self.pad_token_id = train_cfg.get("pad_token_id", -100)
        self.gen_num_beams = train_cfg.get("num_beams", 3)
        self.save_top_k = train_cfg.get("save_top_k", 3)
        self.main_metric = eval_cfg.get("main_metric", "cider")
        self.use_amp = bool(train_cfg.get("amp", True) and torch.cuda.is_available())
        self.log_timing = bool(train_cfg.get("log_timing", False))
        self._sync_cuda_timing = self.log_timing and hasattr(self.device, "type") and self.device.type == "cuda"
        # establish metric priority order for best-trial reporting (cider first, then bertscore)
        priority_seed = ["cider", "bertscore", self.main_metric]
        seen = set()
        self.metric_priority: List[str] = []
        for metric in priority_seed:
            if metric and metric not in seen:
                self.metric_priority.append(metric)
                seen.add(metric)
    
        # initialize components
        self.optimizer = self._build_optimizer(self.optimizer_name, self.lr, self.weight_decay)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.pad_token_id)
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        steps_per_epoch = max(1, math.ceil(len(self.train_loader) / max(1, self.grad_accum_steps)))
        self.total_steps = self.epochs * steps_per_epoch
        warmup_fraction = float(train_cfg.get("warmup_fraction", 0.05))
        candidate_warmup = int(round(self.total_steps * warmup_fraction))
        if train_cfg.get("warmup_steps") is not None:
            candidate_warmup = int(train_cfg.get("warmup_steps"))
        candidate_warmup = max(0, min(self.total_steps, candidate_warmup))
        if candidate_warmup >= self.total_steps and self.total_steps > 0:
            candidate_warmup = max(1, self.total_steps - 1)
        self.warmup_steps = candidate_warmup
        self.scheduler = self._build_scheduler(self.scheduler_name, self.warmup_steps, self.step_size, self.gamma)
        
        self._per_step_scheduler = isinstance(self.scheduler, torch.optim.lr_scheduler.LambdaLR)
        self._per_epoch_scheduler = isinstance(self.scheduler, torch.optim.lr_scheduler.StepLR)
    
        self.best_scores: List[float] = [] # track top-k checkpoints
        self.saved_ckpts: List[str] = [] # track top-k checkpoints
        self.config = config  # keep YAML for checkpoint metadata


    def _build_optimizer(self, name: str, lr: float, weight_decay: float):
        name = name.lower()
        if name == "adamw":
            return torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        if name == "adam":
            return torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        if name == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        raise ValueError(f"Unknown optimizer '{name}'")
    
    
    def _build_scheduler(self, name: str, warmup_steps: int, step_size: int, gamma: float):
        name = name.lower()
        if name == "none":
            return None
        if name == "step":
            return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        if name == "cosine":
            total = max(1, self.total_steps)
            def lr_lambda(current_step):
                # linear warmup
                if warmup_steps > 0 and current_step < warmup_steps:
                    return float(current_step + 1) / float(warmup_steps)
                # cosine decay from 1 -> 0
                progress = (current_step - warmup_steps) / float(max(1, total - warmup_steps))
                progress = min(max(progress, 0.0), 1.0)
                return 0.5 * (1.0 + math.cos(math.pi * progress))
            return torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        raise ValueError(f"Unknown scheduler '{name}'")
    
    
    def _move_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        out = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                out[k] = v.to(self.device, non_blocking=True)
            elif isinstance(v, (list, tuple)):
                # keep text/reference lists on CPU
                out[k] = v
            else:
                out[k] = v
        return out
    
    
    def _compute_loss_from_outputs(self, outputs: Any, batch: Dict[str, Any]) -> torch.Tensor:
        # Prefer model-provided loss if available
        if isinstance(outputs, dict) and "loss" in outputs and outputs["loss"] is not None:
            return outputs["loss"]

        # Otherwise, try CE over logits & labels
        logits = None
        if isinstance(outputs, dict) and "logits" in outputs:
            logits = outputs["logits"]
        elif isinstance(outputs, (list, tuple)) and len(outputs) > 0 and torch.is_tensor(outputs[0]):
            logits = outputs[0]

        labels = batch.get("labels", None)
        if logits is None or labels is None:
            raise RuntimeError("Cannot compute loss: need either outputs['loss'] or (logits, labels).")

        # logits: [B, T, V], labels: [B, T]
        B, T, V = logits.shape
        loss = self.loss_fn(logits.reshape(B * T, V), labels.reshape(B * T))
        return loss
    
    
    def train_epoch(self, epoch: int):
        """
        Execute one training epoch and return average loss.
        Args:
            epoch (int): Current epoch index.
        Returns:
            float: Average training loss across all batches.
        """
        # Responsibilities:
        # - Set model to train mode
        # - Iterate through training DataLoader
        # - Compute loss and backpropagate
        # - Update optimizer and scheduler
        # - Track loss for progress display
        self.model.train()
        running = 0.0
        count = 0
        accum = max(1, self.grad_accum_steps)
        self.optimizer.zero_grad(set_to_none=True)
        prev_time = time.perf_counter()
        data_time_sum = 0.0
        device_time_sum = 0.0
        compute_time_sum = 0.0
        iter_time_sum = 0.0

        # --- train_epoch ---
        # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for step, batch in enumerate(self.train_loader, start=1):
            if self.log_timing:
                iter_start = time.perf_counter()
                data_time = iter_start - prev_time
            else:
                iter_start = None
                data_time = 0.0

            if self.log_timing:
                device_start = time.perf_counter()
            batch = self._move_to_device(batch)
            if self.log_timing:
                device_time = time.perf_counter() - device_start
            else:
                device_time = 0.0

            if self.log_timing:
                compute_start = time.perf_counter()
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                outputs = self.model.forward(**{k: v for k, v in batch.items() if k != "references"})
                loss = self._compute_loss_from_outputs(outputs, batch) / accum

            self.scaler.scale(loss).backward()
            do_step = (step % accum == 0) or (step == len(self.train_loader))
            if do_step:
                if self.max_norm and self.max_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                # step only per-step schedulers here
                if self._per_step_scheduler:
                    self.scheduler.step()

            if self.log_timing:
                if self._sync_cuda_timing:
                    torch.cuda.synchronize(self.device)
                compute_end = time.perf_counter()
                compute_time = compute_end - compute_start
                iter_total = compute_end - iter_start
                prev_time = compute_end
                data_time_sum += data_time
                device_time_sum += device_time
                compute_time_sum += compute_time
                iter_time_sum += iter_total
            else:
                prev_time = time.perf_counter()

            running += loss.item() * accum
            count += 1
            log_every = int(self.config.get("training", {}).get("log_every", 50))
            if step % log_every == 0:
                lr = self.optimizer.param_groups[0]["lr"]
                msg = (
                    f"[Epoch {epoch} | Step {step}/{len(self.train_loader)}] "
                    f"loss={running / count:.4f} lr={lr:.2e}"
                )
                if self.log_timing and step > 0:
                    denom = float(step)
                    msg += (
                        f" avg_data={data_time_sum / denom:.3f}s"
                        f" avg_to_device={device_time_sum / denom:.3f}s"
                        f" avg_compute={compute_time_sum / denom:.3f}s"
                        f" avg_iter={iter_time_sum / denom:.3f}s"
                    )
                print(msg)

        # print("PRINTING ACG")
        # print(prof.key_averages().table())
        if self._per_epoch_scheduler: # step only per-epoch schedulers here
            self.scheduler.step()
        return running / max(1, count)
    

    def validate(self, evaluator: 'CaptionEvaluator'):
        """
        Run validation on entire validation set and return metric scores.
        Args:
            evaluator (CaptionEvaluator): Evaluation helper handling BLEU/CIDEr/etc.
        Returns:
            Dict[str, float]: Dictionary of validation metrics and scores.
        """
        # Responsibilities:
        # - Set model to eval mode
        # - Disable gradient computation
        # - Generate captions for validation samples
        # - Use evaluator to compute scores (e.g., BLEU, CIDEr)
        self.model.eval()
        predictions: List[str] = []
        references_all: List[List[str]] = []

        for batch in self.val_loader:
            batch = self._move_to_device(batch)

            point_clouds = batch.get("point_clouds")
            if point_clouds is None or not torch.is_tensor(point_clouds):
                raise RuntimeError("Validation requires 'point_clouds' tensor in batch for generation.")

            # Generate captions (expects model.generate to return List[str] or List[List[int]] decoded internally)
            gen = self.model.generate(
                point_clouds=point_clouds,
                max_length=self.gen_max_length,
                num_beams=self.gen_num_beams,
            )

            if isinstance(gen, (list, tuple)) and len(gen) > 0 and isinstance(gen[0], str):
                preds = list(gen)
            else:
                # If the model returns token IDs, it must provide a .decode method
                if not hasattr(self.model, "decode"):
                    raise RuntimeError("Model returned token IDs but has no .decode(...) for validation.")
                preds = [self.model.decode(g) for g in gen]

            predictions.extend(preds)

            # References: batch may provide ["references"] as List[List[str]] or List[str]
            refs = batch.get("references")
            if refs is None:
                refs = batch.get("caption")
            if refs is None:
                raise RuntimeError("Validation dataloader must yield 'references' per sample.")
            # Normalize refs to List[List[str]]
            if len(refs) > 0 and isinstance(refs[0], str):
                references_all.extend([[r] for r in refs])  # single-ref per item
            else:
                references_all.extend(refs)

        # Compute metrics
        scores = evaluator.evaluate(predictions, references_all)
        return scores
    

    def train(self, evaluator: 'CaptionEvaluator', save_dir: str = "checkpoints"):
        """
        Execute complete training procedure with periodic validation and checkpointing.

        Args:
            evaluator (CaptionEvaluator): Evaluation helper for validation metrics.
            save_dir (str): Directory path to save checkpoints.
        """
        # Responsibilities:
        # - Loop over epochs
        # - Call train_epoch() and validate() each epoch
        # - Log results and print progress
        # - Save best model checkpoint based on validation metric
        os.makedirs(save_dir, exist_ok=True)
        best_main = -float("inf")
        main_key = self.main_metric
        best_summary: Optional[Dict[str, Any]] = None

        for epoch in range(1, self.epochs + 1):
            train_loss = self.train_epoch(epoch)
            print(f"Epoch {epoch} finished. avg_train_loss={train_loss:.4f}")

            do_eval = (epoch % self.eval_every == 0)
            if not do_eval:
                continue

            val_scores = self.validate(evaluator)
            pretty = " | ".join([f"{k}: {v:.2f}" for k, v in val_scores.items()])
            print(f"[Val @ epoch {epoch}] {pretty}")

            main_value = val_scores.get(main_key)
            if main_value is None:
                if not val_scores:
                    raise RuntimeError("No validation scores returned.")
                main_key = next(iter(val_scores))
                main_value = val_scores[main_key]
                print(f"main_metric '{self.main_metric}' not found; falling back to '{main_key}'.")

            ckpt_path = self.save_checkpoint(epoch, val_scores, save_dir)
            self._maybe_register_topk(ckpt_path, main_value)

            if main_value > best_main:
                best_main = main_value
                print(f"New best {main_key}: {best_main:.2f} (epoch {epoch})")

            # track best trial using priority metrics
            compare_key = None
            for metric_name in self.metric_priority:
                if metric_name in val_scores:
                    compare_key = metric_name
                    break
            if compare_key is None and val_scores:
                compare_key = next(iter(val_scores))

            if compare_key is not None:
                candidate_value = val_scores[compare_key]
                if (
                    best_summary is None
                    or candidate_value > best_summary["value"]
                ):
                    best_summary = {
                        "epoch": epoch,
                        "scores": val_scores.copy(),
                        "key": compare_key,
                        "value": candidate_value,
                    }

        print("Training complete.")
        self.best_summary = best_summary
        if best_summary is not None:
            print("-----------------------")
            metrics_str = ", ".join(
                f"{k}={v:.2f}" for k, v in best_summary["scores"].items()
            )
            if not metrics_str:
                metrics_str = "n/a"
            data_cfg = self.config.get("data", {})
            model_cfg = self.config.get("model", {})
            samples = data_cfg.get("max_samples")
            samples = samples if samples is not None else "all"
            prefix_tokens = model_cfg.get("prefix_tokens", 1)
            encoder = model_cfg.get("encoder_type", "unknown")
            point_dim = data_cfg.get("point_cloud_size")
            if point_dim is None:
                point_dim = model_cfg.get("output_dim", "unknown")
            decoder_lora_used = bool(model_cfg.get("use_decoder_lora", False))
            decoder_lora_r = model_cfg.get("decoder_lora_r", "n/a")
            decoder_lora_alpha = model_cfg.get("decoder_lora_alpha", "n/a")

            if decoder_lora_used:
                lora_suffix = " | lora_r={r} | lora_alpha={alpha}".format(
                    r=decoder_lora_r,
                    alpha=decoder_lora_alpha,
                )
            else:
                lora_suffix = ""

            print(
                "best_epoch={epoch} | metrics=[{metrics}] | "
                "samples={samples} | prefix_tokens={prefix_tokens} | encoder={encoder} | "
                "lr={lr:.2e} | pointcloud_dim={point_dim} | decoder_lora={lora_flag}{lora_suffix}".format(
                    epoch=best_summary["epoch"],
                    key=best_summary["key"],
                    value=best_summary["value"],
                    metrics=metrics_str,
                    samples=samples,
                    prefix_tokens=prefix_tokens,
                    encoder=encoder,
                    lr=self.lr,
                    point_dim=point_dim,
                    lora_flag="yes" if decoder_lora_used else "no",
                    lora_suffix=lora_suffix,
                )
            )


    def save_checkpoint(self, epoch: int, scores: Dict[str, float], save_dir: str):
        """
        Save model checkpoint with training metadata and evaluation scores.
        Args:
            epoch (int): Current training epoch.
            scores (Dict[str, float]): Validation metrics for this checkpoint.
            save_dir (str): Directory path to save model and metadata.
        """
        # Responsibilities:
        # - Create save directory if not exists
        # - Serialize model state_dict, optimizer state, and current scores
        # - Optionally keep only best checkpoints
        os.makedirs(save_dir, exist_ok=True)
        ckpt = {
            "epoch": epoch,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler_state": self.scaler.state_dict() if self.scaler is not None else None,
            "scores": scores,
            "config": self.config,  # store raw YAML config
        }
        path = os.path.join(save_dir, f"epoch{epoch:03d}.pt")
        torch.save(ckpt, path)
        print(f"Saved checkpoint: {path}")
        return path
    
    
    def _maybe_register_topk(self, path: str, score: float):
        """
        Keep only top-k checkpoints. Never delete the newly saved checkpoint.
        """
        self.best_scores.append(score)
        self.saved_ckpts.append(path)

        pairs = list(zip(self.best_scores, self.saved_ckpts))
        newest = (score, path)

        ranked = sorted(pairs, key=lambda x: x[0], reverse=True)
        keep = ranked[: self.save_top_k]
        if newest not in keep:
            keep.append(newest)

        keep_paths = {p for _, p in keep}
        for _, p in pairs:
            if p not in keep_paths and os.path.exists(p):
                try:
                    os.remove(p)
                    print(f"Removed old checkpoint: {p}")
                except Exception as e:
                    print(f"Warning: failed to remove {p}: {e}")

        self.best_scores = [s for s, _ in keep]
        self.saved_ckpts = [p for _, p in keep]


__all__ = ["Trainer"]
