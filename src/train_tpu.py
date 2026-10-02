"""CLI: train Spec2FP on a streamed / cached train sample (CPU, GPU, or TPU)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python src/train_tpu.py` from repo root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import get_config
from src.data import dataset_from_dataframe, iter_train_row_groups, load_train_slice
from src.tpu_trainer import train_spec2fp


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Spec2FP (PyTorch / PyTorch-XLA)")
    p.add_argument("--max-spectra", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--time-limit-s", type=float, default=None)
    p.add_argument("--slice-rows", type=int, default=None, help="Debug: only the first N parquet rows")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    overrides = {}
    if args.max_spectra is not None:
        overrides["max_train_spectra"] = args.max_spectra
    if args.epochs is not None:
        overrides["num_epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    cfg = get_config(**overrides)

    if args.slice_rows:
        df = load_train_slice(cfg, args.slice_rows)
        ds = dataset_from_dataframe(df, cfg)
    else:
        frames = []
        n = 0
        cap = cfg.max_train_spectra
        for g in iter_train_row_groups(cfg.train_path):
            take = min(len(g), cap - n)
            if take <= 0:
                break
            frames.append(g.iloc[:take])
            n += take
            print(f"[cache] collected {n} spectra")
            if n >= cap:
                break
        import pandas as pd

        df = pd.concat(frames, ignore_index=True)
        ds = dataset_from_dataframe(df, cfg)

    print(
        "static shapes:",
        {k: tuple(v.shape) if hasattr(v, "shape") else None for k, v in ds[0].items()},
    )
    train_spec2fp(ds, cfg, time_limit_s=args.time_limit_s)


if __name__ == "__main__":
    main()
