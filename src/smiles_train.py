"""TPU-static training for the prefix-conditioned SMILES decoder."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler

from src.chem import randomize_smiles
from src.config import NUM_ADDUCTS, Config, get_config
from src.models.smiles_decoder import LightweightSmilesDecoder, decoder_from_config
from src.smiles_tokenizer import SmilesTokenizer, build_tokenizer
from src.tpu_trainer import get_device, set_seed, split_dataset, try_import_xla


class SmilesLMDataset(Dataset):
    """Each item is already padded to ``cfg.smiles_max_len`` (XLA-static)."""

    def __init__(
        self,
        smiles: Sequence[str],
        masses: np.ndarray,
        fingerprints: np.ndarray,
        tokenizer: SmilesTokenizer,
        cfg: Config,
        *,
        adduct_ids: np.ndarray | None = None,
        randomize: bool | None = None,
    ) -> None:
        self.smiles = [str(s) for s in smiles]
        self.masses = np.ascontiguousarray(masses, dtype=np.float32).reshape(-1)
        self.fingerprints = np.ascontiguousarray(fingerprints, dtype=np.float32)
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.max_len = int(cfg.smiles_max_len)
        self.randomize = cfg.smiles_randomize if randomize is None else bool(randomize)
        if adduct_ids is None:
            self.adduct_ids = None
        else:
            self.adduct_ids = np.ascontiguousarray(adduct_ids, dtype=np.int64).reshape(-1)
        n = len(self.smiles)
        assert self.masses.shape[0] == n
        assert self.fingerprints.shape[0] == n

    def __len__(self) -> int:
        return len(self.smiles)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        smi = self.smiles[idx]
        if self.randomize:
            smi = randomize_smiles(smi) or smi
        ids = self.tokenizer.encode(smi, self.max_len)
        tokens = torch.tensor(ids, dtype=torch.long)
        pad = tokens == self.tokenizer.pad_id
        if self.adduct_ids is None:
            adduct = int(np.random.randint(0, NUM_ADDUCTS))
        else:
            if np.random.rand() < 0.3:
                adduct = int(np.random.randint(0, NUM_ADDUCTS))
            else:
                adduct = int(self.adduct_ids[idx] % NUM_ADDUCTS)
        return {
            "fp": torch.from_numpy(self.fingerprints[idx]),
            "mass": torch.tensor([float(self.masses[idx])], dtype=torch.float32),
            "adduct_idx": torch.tensor(adduct, dtype=torch.long),
            "tgt_tokens": tokens,
            "tgt_mask": pad,
        }


def collate_smiles(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}


def _pad_to_bs(batch: dict[str, torch.Tensor], target: int) -> dict[str, torch.Tensor]:
    n = int(batch["tgt_tokens"].shape[0])
    if n == target:
        return batch
    if n > target:
        return {k: v[:target] for k, v in batch.items()}
    pad = target - n
    out = {}
    for k, v in batch.items():
        pad_shape = (pad,) + tuple(v.shape[1:])
        z = torch.zeros(pad_shape, dtype=v.dtype, device=v.device)
        if k == "tgt_mask":
            z = torch.ones(pad_shape, dtype=v.dtype, device=v.device)
        out[k] = torch.cat([v, z], dim=0)
    return out


def _shifted_labels(tgt_tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
    labels = torch.full_like(tgt_tokens, pad_id)
    labels[:, :-1] = tgt_tokens[:, 1:]
    return labels


def dataset_from_structures(
    smiles: Sequence[str],
    masses: np.ndarray,
    fingerprints: np.ndarray,
    tokenizer: SmilesTokenizer,
    cfg: Config,
    adduct_ids: np.ndarray | None = None,
) -> SmilesLMDataset:
    return SmilesLMDataset(smiles, masses, fingerprints, tokenizer, cfg, adduct_ids=adduct_ids)


def save_decoder_checkpoint(
    model: LightweightSmilesDecoder,
    tokenizer: SmilesTokenizer,
    path: Path,
    cfg: Config,
    extra: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tokenizer.save(cfg.smiles_vocab_path)
    payload = {
        "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
        "vocab": tokenizer.id_to_token,
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_decoder(
    path: Path | str,
    cfg: Config | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[LightweightSmilesDecoder, SmilesTokenizer]:
    cfg = cfg or get_config()
    try:
        blob = torch.load(str(path), map_location=map_location, weights_only=False)
    except TypeError:
        blob = torch.load(str(path), map_location=map_location)
    if "vocab" in blob:
        tokenizer = SmilesTokenizer.from_tokens(blob["vocab"])
    elif cfg.smiles_vocab_path.exists():
        tokenizer = SmilesTokenizer.load(cfg.smiles_vocab_path)
    else:
        tokenizer = SmilesTokenizer.default()
    model = decoder_from_config(tokenizer, cfg)
    model.load_state_dict(blob["model"] if "model" in blob else blob)
    model.eval()
    return model, tokenizer


def train_smiles_decoder(
    dataset: SmilesLMDataset,
    tokenizer: SmilesTokenizer,
    cfg: Config | None = None,
    *,
    device=None,
    time_limit_s: float | None = None,
) -> LightweightSmilesDecoder:
    cfg = cfg or get_config()
    set_seed(cfg.seed)
    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)
    xm, pl = try_import_xla()
    if device is None:
        device, use_xla = get_device(prefer_xla=True)
    else:
        use_xla = xm is not None and "xla" in str(device).lower()

    bs = int(cfg.smiles_batch_size)
    n_epochs = int(cfg.smiles_num_epochs)
    train_ds, val_ds = split_dataset(dataset, cfg.val_fraction, cfg.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=bs,
        sampler=RandomSampler(train_ds),
        num_workers=cfg.num_workers,
        collate_fn=collate_smiles,
        drop_last=False,
        pin_memory=torch.cuda.is_available() and not use_xla,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=bs,
            sampler=SequentialSampler(val_ds),
            collate_fn=collate_smiles,
            drop_last=False,
        )
        if len(val_ds) > 0
        else None
    )

    model = decoder_from_config(tokenizer, cfg).to(device)
    pad_id = tokenizer.pad_id
    criterion = torch.nn.CrossEntropyLoss(
        ignore_index=pad_id, label_smoothing=float(cfg.smiles_label_smoothing)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.smiles_lr, weight_decay=cfg.weight_decay)
    n_steps = max(1, math.ceil(len(train_ds) / bs) * n_epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_steps)

    probe = collate_smiles([dataset[0], dataset[min(1, len(dataset) - 1)]])
    assert tuple(probe["tgt_tokens"].shape)[1] == cfg.smiles_max_len, probe["tgt_tokens"].shape
    assert tuple(probe["fp"].shape)[1] == cfg.fp_bits, probe["fp"].shape

    t0 = time.time()
    limit = cfg.smiles_train_time_limit_s if time_limit_s is None else time_limit_s
    global_step = 0
    best_val = float("inf")
    print(
        f"[smiles] device={device} xla={use_xla} n_train={len(train_ds)} "
        f"n_val={len(val_ds)} vocab={tokenizer.vocab_size} max_len={cfg.smiles_max_len}"
    )

    def _loss_on(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, int]:
        if not use_xla:
            batch = {k: v.to(device) for k, v in batch.items()}
        n = int(batch["tgt_tokens"].shape[0])
        batch = _pad_to_bs(batch, bs)
        logits = model(
            fp=batch["fp"],
            mass=batch["mass"],
            adduct_idx=batch["adduct_idx"],
            tgt_tokens=batch["tgt_tokens"],
            tgt_mask=batch["tgt_mask"],
        )
        labels = _shifted_labels(batch["tgt_tokens"], pad_id)
        loss = criterion(logits[:n].reshape(-1, tokenizer.vocab_size), labels[:n].reshape(-1))
        return loss, n

    for epoch in range(n_epochs):
        model.train()
        if use_xla and pl is not None:
            iterator = pl.ParallelLoader(train_loader, [device]).per_device_loader(device)
        else:
            iterator = train_loader
        running = 0.0
        n_seen = 0
        for batch in iterator:
            if limit and (time.time() - t0) > limit:
                print("[smiles] time limit reached, stopping")
                save_decoder_checkpoint(
                    model, tokenizer, cfg.smiles_checkpoint_path, cfg, extra={"epoch": epoch}
                )
                return model
            optimizer.zero_grad(set_to_none=True)
            loss, n = _loss_on(batch)
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
                    f"[smiles] epoch={epoch+1} step={global_step} "
                    f"loss={running / max(n_seen, 1):.4f} lr={scheduler.get_last_lr()[0]:.2e}"
                )

        msg = f"[smiles] epoch {epoch+1}/{n_epochs} train_loss={running / max(n_seen, 1):.4f}"
        if val_loader is not None and len(val_ds) > 0:
            model.eval()
            vloss = []
            with torch.no_grad():
                it = (
                    pl.ParallelLoader(val_loader, [device]).per_device_loader(device)
                    if use_xla and pl is not None
                    else val_loader
                )
                for batch in it:
                    loss, _n = _loss_on(batch)
                    vloss.append(float(loss.detach().cpu().item()))
                    if use_xla and xm is not None:
                        xm.mark_step()
            vl = float(np.mean(vloss) if vloss else 0.0)
            msg += f" val_loss={vl:.4f}"
            if vl < best_val:
                best_val = vl
                save_decoder_checkpoint(
                    model,
                    tokenizer,
                    cfg.smiles_checkpoint_path,
                    cfg,
                    extra={"epoch": epoch, "val_loss": best_val},
                )
        else:
            save_decoder_checkpoint(
                model, tokenizer, cfg.smiles_checkpoint_path, cfg, extra={"epoch": epoch}
            )
        print(msg)

    save_decoder_checkpoint(
        model, tokenizer, cfg.smiles_checkpoint_path, cfg, extra={"epoch": n_epochs}
    )
    return model


def fit_tokenizer_from_smiles(smiles: Sequence[str], cfg: Config) -> SmilesTokenizer:
    tok = build_tokenizer(smiles, min_count=3, max_size=96)
    tok.save(cfg.smiles_vocab_path)
    return tok
