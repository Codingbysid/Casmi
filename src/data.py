"""Parquet loading, structure tables, and TPU-static PyTorch datasets."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from src.chem import formula_to_mass, morgan_fingerprint
from src.config import Config
from src.preprocessing import featurize_spectrum, stack_features


TRAIN_COLUMNS = [
    "ingest_lib",
    "normalized_smiles",
    "inchikey",
    "inchikey14",
    "molecular_formula",
    "ionization_mode",
    "instrument_type",
    "adduct",
    "adduct_orig",
    "precursor_mz",
    "precursor_error_ppm",
    "ms2_mzs",
    "ms2_normalized_intensities",
    "num_peaks",
    "base_peak_intensity",
    "collision_energy_ev",
    "collision_energy_orig",
    "collision_energy_orig_units",
]


def resolve_split_path(path: Path | str, split: str) -> Path:
    p = Path(path)
    if p.exists():
        return p
    raise FileNotFoundError(f"Missing {split} parquet: {p}")


def read_parquet(path: Path | str, columns: Sequence[str] | None = None) -> pd.DataFrame:
    return pq.read_table(str(path), columns=list(columns) if columns else None).to_pandas()


def iter_train_row_groups(
    path: Path | str,
    columns: Sequence[str] | None = None,
) -> Iterator[pd.DataFrame]:
    pf = pq.ParquetFile(str(path))
    cols = list(columns) if columns else None
    for i in range(pf.num_row_groups):
        yield pf.read_row_group(i, columns=cols).to_pandas()


def load_train_slice(cfg: Config, n_rows: int, columns: Sequence[str] | None = None) -> pd.DataFrame:
    """Read the first ``n_rows`` training spectra without loading the full 2.5M table."""
    pf = pq.ParquetFile(str(cfg.train_path))
    cols = list(columns) if columns else None
    chunks: list[pd.DataFrame] = []
    remaining = int(n_rows)
    for i in range(pf.num_row_groups):
        if remaining <= 0:
            break
        table = pf.read_row_group(i, columns=cols)
        take = min(remaining, table.num_rows)
        chunks.append(table.slice(0, take).to_pandas())
        remaining -= take
    if not chunks:
        return pd.DataFrame()
    return pd.concat(chunks, ignore_index=True)


def load_test(cfg: Config) -> pd.DataFrame:
    return read_parquet(cfg.test_path)


@dataclass
class StructureRecord:
    smiles: str
    inchikey14: str
    exact_mass: float
    formula: str


def build_structure_table(train_df: pd.DataFrame) -> pd.DataFrame:
    """Unique 2D skeletons from a train frame, keyed by InChIKey14."""
    cols = [c for c in ("normalized_smiles", "inchikey14", "molecular_formula") if c in train_df.columns]
    sub = train_df[cols].dropna(subset=["normalized_smiles", "inchikey14"])
    # Prefer the first SMILES per skeleton (already RDKit-normalized in the dump).
    sub = sub.drop_duplicates(subset=["inchikey14"], keep="first").reset_index(drop=True)
    masses = np.array(
        [formula_to_mass(f) if isinstance(f, str) else 0.0 for f in sub["molecular_formula"].tolist()],
        dtype=np.float64,
    )
    # If formula mass failed, leave 0; retrieval will skip zeros.
    sub = sub.copy()
    sub["exact_mass"] = masses
    return sub


def fingerprints_for_smiles(smiles: Sequence[str], cfg: Config) -> np.ndarray:
    fps = np.zeros((len(smiles), cfg.fp_bits), dtype=np.uint8)
    for i, smi in enumerate(smiles):
        fps[i] = morgan_fingerprint(smi, n_bits=cfg.fp_bits, radius=cfg.morgan_radius)
    return fps


class SpectrumFingerprintDataset(Dataset):
    """Fixed-shape tensors only: peak stacks, binned spectrum, precursor feats, fingerprints."""

    def __init__(
        self,
        peak_mz: np.ndarray,
        peak_intensity: np.ndarray,
        peak_nl: np.ndarray,
        peak_mask: np.ndarray,
        binned: np.ndarray,
        precursor_feat: np.ndarray,
        adduct_id: np.ndarray,
        fingerprints: np.ndarray,
    ) -> None:
        self.peak_mz = np.ascontiguousarray(peak_mz, dtype=np.float32)
        self.peak_intensity = np.ascontiguousarray(peak_intensity, dtype=np.float32)
        self.peak_nl = np.ascontiguousarray(peak_nl, dtype=np.float32)
        self.peak_mask = np.ascontiguousarray(peak_mask, dtype=np.float32)
        self.binned = np.ascontiguousarray(binned, dtype=np.float32)
        self.precursor_feat = np.ascontiguousarray(precursor_feat, dtype=np.float32)
        self.adduct_id = np.ascontiguousarray(adduct_id, dtype=np.int64)
        fp = np.ascontiguousarray(fingerprints)
        if fp.dtype != np.float32:
            fp = fp.astype(np.float32)
        self.fingerprints = fp
        n = self.peak_mz.shape[0]
        assert self.peak_intensity.shape[0] == n
        assert self.peak_nl.shape[0] == n
        assert self.peak_mask.shape[0] == n
        assert self.binned.shape[0] == n
        assert self.precursor_feat.shape[0] == n
        assert self.adduct_id.shape[0] == n
        assert self.fingerprints.shape[0] == n

    def __len__(self) -> int:
        return int(self.peak_mz.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "peak_mz": torch.from_numpy(self.peak_mz[idx]),
            "peak_intensity": torch.from_numpy(self.peak_intensity[idx]),
            "peak_nl": torch.from_numpy(self.peak_nl[idx]),
            "peak_mask": torch.from_numpy(self.peak_mask[idx]),
            "binned": torch.from_numpy(self.binned[idx]),
            "precursor_feat": torch.from_numpy(self.precursor_feat[idx]),
            "adduct_id": torch.tensor(self.adduct_id[idx], dtype=torch.long),
            "fingerprint": torch.from_numpy(self.fingerprints[idx]),
        }


def collate_fixed(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Default collate is already static-shape-safe; kept explicit for XLA loaders."""
    keys = batch[0].keys()
    out: dict[str, torch.Tensor] = {}
    for k in keys:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out


def featurize_dataframe(
    df: pd.DataFrame,
    cfg: Config,
    *,
    compute_fingerprints: bool = True,
    smiles_col: str = "normalized_smiles",
) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
    """Vector of static examples from a spectra dataframe."""
    feats: list[dict[str, Any]] = []
    mzs = df["ms2_mzs"].tolist()
    ints = df["ms2_normalized_intensities"].tolist()
    precs = df["precursor_mz"].to_numpy()
    adducts = df["adduct"].tolist() if "adduct" in df.columns else [""] * len(df)
    ces = df["collision_energy_ev"].tolist() if "collision_energy_ev" in df.columns else [None] * len(df)
    ions = df["ionization_mode"].tolist() if "ionization_mode" in df.columns else [None] * len(df)
    for i in range(len(df)):
        feats.append(
            featurize_spectrum(
                mzs[i],
                ints[i],
                float(precs[i] or 0.0),
                adducts[i],
                cfg=cfg,
                collision_energy=ces[i],
                ionization_mode=ions[i],
            )
        )
    stacked = stack_features(feats)
    fps: np.ndarray | None = None
    if compute_fingerprints and smiles_col in df.columns:
        smiles = df[smiles_col].astype(str).tolist()
        cache: dict[str, np.ndarray] = {}
        fps = np.zeros((len(smiles), cfg.fp_bits), dtype=np.uint8)
        if "inchikey14" in df.columns:
            keys = df["inchikey14"].astype(str).tolist()
        else:
            keys = smiles
        for i, (key, smi) in enumerate(zip(keys, smiles)):
            hit = cache.get(key)
            if hit is None:
                hit = morgan_fingerprint(smi, n_bits=cfg.fp_bits, radius=cfg.morgan_radius)
                cache[key] = hit
            fps[i] = hit
    return stacked, fps


def dataset_from_dataframe(df: pd.DataFrame, cfg: Config) -> SpectrumFingerprintDataset:
    stacked, fps = featurize_dataframe(df, cfg, compute_fingerprints=True)
    if fps is None:
        fps = np.zeros((len(df), cfg.fp_bits), dtype=np.float32)
    return SpectrumFingerprintDataset(
        peak_mz=stacked["peak_mz"],
        peak_intensity=stacked["peak_intensity"],
        peak_nl=stacked["peak_nl"],
        peak_mask=stacked["peak_mask"],
        binned=stacked["binned"],
        precursor_feat=stacked["precursor_feat"],
        adduct_id=stacked["adduct_id"],
        fingerprints=fps,
    )


def dataset_from_structure_index(
    index,
    cfg: Config,
    *,
    max_n: int | None = None,
    seed: int = 42,
) -> SpectrumFingerprintDataset:
    """Train Spec2FP from the retrieval library (no second 2.5M-row parquet scan)."""
    from src.preprocessing import bin_spectrum, precursor_feature_vector

    if getattr(index, "peak_mz", None) is None:
        raise ValueError("structure index has no representative peaks")
    n = int(index.smiles.shape[0])
    rng = np.random.default_rng(seed)
    take = np.arange(n)
    if max_n is not None and n > int(max_n):
        take = np.sort(rng.choice(n, size=int(max_n), replace=False))
    peak_mz = np.ascontiguousarray(index.peak_mz[take], dtype=np.float32)
    peak_int = np.ascontiguousarray(index.peak_intensity[take], dtype=np.float32)
    peak_mask = np.ascontiguousarray(index.peak_mask[take], dtype=np.float32)
    mass = np.ascontiguousarray(index.exact_mass[take], dtype=np.float32)
    fps = np.ascontiguousarray(index.fingerprints[take], dtype=np.float32)
    peak_nl = np.clip(mass[:, None] - peak_mz, 0.0, None) * peak_mask
    n_take = int(peak_mz.shape[0])
    binned = np.zeros((n_take, cfg.n_mz_bins), dtype=np.float32)
    feats = np.zeros((n_take, cfg.precursor_feat_dim), dtype=np.float32)
    for i in range(n_take):
        binned[i] = bin_spectrum(peak_mz[i], peak_int[i], peak_mask[i], cfg=cfg)
        n_peaks = int(peak_mask[i].sum())
        prec = float(mass[i]) + 1.007276
        feats[i] = precursor_feature_vector(
            precursor_mz=prec,
            neutral_mass=float(mass[i]),
            adduct="[M+H]+",
            collision_energy=None,
            ionization_mode="positive",
            n_peaks=n_peaks,
            cfg=cfg,
        )
        if (i + 1) % 20_000 == 0:
            print(f"[dataset] binned {i+1}/{n_take}")
    adduct_id = np.zeros((n_take,), dtype=np.int64)
    return SpectrumFingerprintDataset(
        peak_mz=peak_mz,
        peak_intensity=peak_int,
        peak_nl=peak_nl.astype(np.float32),
        peak_mask=peak_mask,
        binned=binned,
        precursor_feat=feats,
        adduct_id=adduct_id,
        fingerprints=fps,
    )


def assert_static_shapes(batch: dict[str, torch.Tensor | np.ndarray], cfg: Config) -> None:
    """Raise if any spectral tensor violates the XLA static-shape contract."""
    n = cfg.top_n_peaks
    b = cfg.n_mz_bins
    f = cfg.fp_bits
    p = cfg.precursor_feat_dim

    def _shape(x) -> tuple[int, ...]:
        if hasattr(x, "shape"):
            return tuple(int(s) for s in x.shape)
        raise TypeError(type(x))

    checks = {
        "peak_mz": (n,),
        "peak_intensity": (n,),
        "peak_nl": (n,),
        "peak_mask": (n,),
        "binned": (b,),
        "precursor_feat": (p,),
    }
    for key, tail in checks.items():
        if key not in batch:
            continue
        shape = _shape(batch[key])
        if shape[-len(tail) :] != tail:
            raise AssertionError(f"{key} has shape {shape}, expected trailing {tail}")
    if "fingerprint" in batch:
        shape = _shape(batch["fingerprint"])
        if shape[-1] != f:
            raise AssertionError(f"fingerprint last dim {shape[-1]} != {f}")
    if "adduct_id" in batch:
        shape = _shape(batch["adduct_id"])
        if len(shape) not in (0, 1):
            raise AssertionError(f"adduct_id rank {shape}")
