"""PyTorch / PyTorch-XLA training loop for Spec2FP.

On TPU the loop uses ``xm.optimizer_step`` and ``pl.ParallelLoader``. Input
batches are padded to a fixed ``batch_size`` so XLA never recompiles on the
partial last step.
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler, Subset

from src.config import Config, get_config
from src.data import SpectrumFingerprintDataset, assert_static_shapes, collate_fixed
from src.model import FocalBCELoss, Spec2FP, model_from_config


def try_import_xla():
    try:
        import torch_xla.core.xla_model as xm
        import torch_xla.distributed.parallel_loader as pl

        return xm, pl
    except Exception:
        return None, None


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(prefer_xla: bool = True):
    xm, _ = try_import_xla()
    if prefer_xla and xm is not None:
        return xm.xla_device(), True
    if torch.cuda.is_available():
        return torch.device("cuda"), False
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps"), False
    return torch.device("cpu"), False


def _pad_to_batch_size(batch: dict[str, torch.Tensor], target: int) -> dict[str, torch.Tensor]:
    n = int(batch["peak_mz"].shape[0])
    if n == target:
        return batch
    if n > target:
        return {k: v[:target] for k, v in batch.items()}
    pad = target - n
    out = {}
    for k, v in batch.items():
        if v.dim() == 0:
            out[k] = v
            continue
        pad_shape = (pad,) + tuple(v.shape[1:])
        z = torch.zeros(pad_shape, dtype=v.dtype, device=v.device)
        out[k] = torch.cat([v, z], dim=0)
    return out


def _move(batch: dict[str, torch.Tensor], device) -> dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


def _forward(model: Spec2FP, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return model(
        peak_mz=batch["peak_mz"],
        peak_intensity=batch["peak_intensity"],
        peak_nl=batch["peak_nl"],
        peak_mask=batch["peak_mask"],
        binned=batch["binned"],
        precursor_feat=batch["precursor_feat"],
        adduct_id=batch["adduct_id"],
    )


def _bit_metrics(logits: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    with torch.no_grad():
        pred = (torch.sigmoid(logits) >= 0.5).float()
        tgt = target.float()
        tp = (pred * tgt).sum()
        fp = (pred * (1 - tgt)).sum()
        fn = ((1 - pred) * tgt).sum()
        prec = tp / (tp + fp + 1e-6)
        rec = tp / (tp + fn + 1e-6)
        f1 = 2 * prec * rec / (prec + rec + 1e-6)
        acc = (pred == tgt).float().mean()
    return {
        "bit_acc": float(acc.item()),
        "bit_f1": float(f1.item()),
    }


def save_checkpoint(model: Spec2FP, path: Path, cfg: Config, extra: dict[str, Any] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "config": cfg.__dict__.copy(),
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_model(path: Path | str, cfg: Config | None = None, map_location: str | torch.device = "cpu") -> Spec2FP:
    try:
        blob = torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        blob = torch.load(str(path), map_location=map_location)
    cfg = cfg or get_config()
    model = model_from_config(cfg)
    model.load_state_dict(blob["model"] if "model" in blob else blob)
    model.eval()
    return model


def split_dataset(ds: Dataset, val_fraction: float, seed: int) -> tuple[Dataset, Dataset]:
    n = len(ds)
    n_val = max(1, int(round(n * val_fraction))) if n > 1 else 0
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    val_idx = perm[:n_val]
    train_idx = perm[n_val:] or perm
    return Subset(ds, train_idx), Subset(ds, val_idx) if n_val else Subset(ds, [])


@torch.no_grad()
def evaluate(
    model: Spec2FP,
    loader: DataLoader,
    device,
    cfg: Config,
    criterion: torch.nn.Module,
    use_xla: bool,
) -> dict[str, float]:
    model.eval()
    xm, pl = try_import_xla()
    if use_xla and pl is not None:
        iterator = pl.ParallelLoader(loader, [device]).per_device_loader(device)
    else:
        iterator = loader
    losses = []
    metrics = []
    for batch in iterator:
        if not use_xla:
            batch = _move(batch, device)
        n = int(batch["peak_mz"].shape[0])
        batch = _pad_to_batch_size(batch, cfg.batch_size)
        logits = _forward(model, batch)[:n]
        target = batch["fingerprint"][:n]
        loss = criterion(logits, target)
        losses.append(float(loss.detach().cpu().item()))
        metrics.append(_bit_metrics(logits, target))
        if use_xla and xm is not None:
            xm.mark_step()
    out = {"val_loss": float(np.mean(losses) if losses else 0.0)}
    if metrics:
        out["val_bit_acc"] = float(np.mean([m["bit_acc"] for m in metrics]))
        out["val_bit_f1"] = float(np.mean([m["bit_f1"] for m in metrics]))
    return out


def train_spec2fp(
    dataset: SpectrumFingerprintDataset,
    cfg: Config | None = None,
    *,
    device=None,
    time_limit_s: float | None = None,
    seed: int | None = None,
    ckpt_name: str | None = None,
) -> Spec2FP:
    cfg = cfg or get_config()
    if seed is not None:
        cfg.seed = int(seed)
    if ckpt_name is not None:
        cfg.ckpt_name = str(ckpt_name)
    set_seed(cfg.seed)
    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)
    xm, pl = try_import_xla()
    if device is None:
        device, use_xla = get_device(prefer_xla=True)
    else:
        use_xla = xm is not None and "xla" in str(device).lower()

    train_ds, val_ds = split_dataset(dataset, cfg.val_fraction, cfg.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=RandomSampler(train_ds),
        num_workers=cfg.num_workers,
        collate_fn=collate_fixed,
        drop_last=False,
        pin_memory=torch.cuda.is_available() and not use_xla,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            sampler=SequentialSampler(val_ds),
            num_workers=0,
            collate_fn=collate_fixed,
            drop_last=False,
        )
        if len(val_ds) > 0
        else None
    )

    model = model_from_config(cfg).to(device)
    criterion = FocalBCELoss(gamma=cfg.focal_gamma, pos_weight=cfg.bce_pos_weight).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    n_steps = max(1, math.ceil(len(train_ds) / cfg.batch_size) * cfg.num_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)

    # Probe static shapes on a dummy batch so XLA failures happen immediately.
    probe = collate_fixed([dataset[0] for _ in range(min(2, len(dataset)))])
    assert_static_shapes(probe, cfg)

    t0 = time.time()
    limit = cfg.train_time_limit_s if time_limit_s is None else time_limit_s
    global_step = 0
    best_val = float("inf")
    print(f"[train] device={device} xla={use_xla} n_train={len(train_ds)} n_val={len(val_ds)}")

    for epoch in range(cfg.num_epochs):
        model.train()
        if use_xla and pl is not None:
            iterator = pl.ParallelLoader(train_loader, [device]).per_device_loader(device)
        else:
            iterator = train_loader
        running = 0.0
        n_seen = 0
        for batch in iterator:
            if limit and (time.time() - t0) > limit:
                print("[train] time limit reached, stopping")
                save_checkpoint(model, cfg.checkpoint_path, cfg, extra={"epoch": epoch, "step": global_step})
                return model
            if not use_xla:
                batch = _move(batch, device)
            n = int(batch["peak_mz"].shape[0])
            batch = _pad_to_batch_size(batch, cfg.batch_size)
            optimizer.zero_grad(set_to_none=True)
            logits = _forward(model, batch)
            loss = criterion(logits[:n], batch["fingerprint"][:n])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            if use_xla and xm is not None:
                xm.optimizer_step(optimizer)
                xm.mark_step()
            else:
                optimizer.step()
            scheduler.step()
            running += float(loss.detach().cpu().item()) * n
            n_seen += n
            global_step += 1
            if global_step % cfg.log_every == 0:
                print(
                    f"[train] epoch={epoch+1} step={global_step} loss={running / max(n_seen, 1):.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e}"
                )
        msg = f"[train] epoch {epoch+1}/{cfg.num_epochs} train_loss={running / max(n_seen, 1):.4f}"
        if val_loader is not None and len(val_ds) > 0:
            stats = evaluate(model, val_loader, device, cfg, criterion, use_xla)
            msg += " " + " ".join(f"{k}={v:.4f}" for k, v in stats.items())
            if stats["val_loss"] < best_val:
                best_val = stats["val_loss"]
                save_checkpoint(
                    model,
                    cfg.checkpoint_path,
                    cfg,
                    extra={"epoch": epoch, "val_loss": best_val},
                )
        else:
            save_checkpoint(model, cfg.checkpoint_path, cfg, extra={"epoch": epoch})
        print(msg)

    save_checkpoint(model, cfg.checkpoint_path, cfg, extra={"epoch": cfg.num_epochs, "step": global_step})
    return model
