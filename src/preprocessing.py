"""Peak cleaning, neutral-loss features, and fixed-size TPU tensors."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np

from src.chem import precursor_to_neutral_mass
from src.config import Config, adduct_to_id


def _as_1d_float(x: Any) -> np.ndarray:
    """Coerce scalars, 1-D arrays, or ragged nested lists/arrays to float32."""
    if x is None:
        return np.zeros((0,), dtype=np.float32)
    if isinstance(x, np.ndarray) and x.dtype != object and x.dtype.kind in "fiu":
        return np.ascontiguousarray(x, dtype=np.float32).reshape(-1)
    if isinstance(x, (bool, np.bool_)):
        return np.zeros((0,), dtype=np.float32)
    if isinstance(x, (int, float, np.integer, np.floating)):
        return np.array([float(x)], dtype=np.float32)

    out: list[float] = []

    def walk(v: Any) -> None:
        if v is None:
            return
        if isinstance(v, (bool, np.bool_)):
            return
        if isinstance(v, (int, float, np.integer, np.floating)):
            out.append(float(v))
            return
        if isinstance(v, np.ndarray):
            if v.size == 0:
                return
            if v.dtype != object and v.dtype.kind in "fiu":
                out.extend(np.ascontiguousarray(v, dtype=np.float32).reshape(-1).tolist())
                return
            for item in v.reshape(-1):
                walk(item)
            return
        if isinstance(v, (bytes, str)):
            s = v.decode("utf-8", "ignore") if isinstance(v, bytes) else v
            s = s.strip()
            if not s or s.lower() in {"nan", "none", "null"}:
                return
            parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
            if len(parts) > 1:
                for p in parts:
                    walk(p)
                return
            try:
                out.append(float(s))
            except ValueError:
                return
            return
        if isinstance(v, dict):
            return
        if isinstance(v, (list, tuple, set)):
            for item in v:
                walk(item)
            return
        if hasattr(v, "tolist"):
            try:
                walk(v.tolist())
                return
            except Exception:
                pass
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            return

    walk(x)
    if not out:
        return np.zeros((0,), dtype=np.float32)
    return np.asarray(out, dtype=np.float32)


def mean_collision_energy(ce: Any) -> float:
    try:
        arr = _as_1d_float(ce)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return 0.0
        return float(np.mean(np.abs(arr)))
    except Exception:
        return 0.0


def clean_peaks(
    mz: np.ndarray,
    intensity: np.ndarray,
    precursor_mz: float,
    *,
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray]:
    """Drop precursor-region / tiny peaks and L2-renormalize intensities.

    Rules:
      * ``mz`` in ``(min_mz, precursor_mz + margin]``
      * intensity >= ``min_rel_intensity * max(intensity)``
    """
    mz = _as_1d_float(mz)
    intensity = _as_1d_float(intensity)
    n = min(mz.size, intensity.size)
    mz = mz[:n]
    intensity = intensity[:n]
    if n == 0:
        return mz, intensity

    finite = np.isfinite(mz) & np.isfinite(intensity) & (intensity > 0)
    mz = mz[finite]
    intensity = intensity[finite]
    if mz.size == 0:
        return mz, intensity

    prec = float(precursor_mz) if np.isfinite(precursor_mz) else 1e9
    keep = (mz >= cfg.min_mz) & (mz <= prec + cfg.precursor_peak_margin_da)
    mz = mz[keep]
    intensity = intensity[keep]
    if mz.size == 0:
        return mz, intensity

    base = float(np.max(intensity))
    if base > 0:
        intensity = intensity / base
        keep = intensity >= cfg.min_rel_intensity
        mz = mz[keep]
        intensity = intensity[keep]
    if mz.size == 0:
        return mz, intensity

    base = float(np.max(intensity))
    if base > 0:
        intensity = intensity / base
    return mz.astype(np.float32, copy=False), intensity.astype(np.float32, copy=False)


def top_n_pad(
    mz: np.ndarray,
    intensity: np.ndarray,
    precursor_mz: float,
    *,
    cfg: Config,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return fixed-length ``(top_n,)`` arrays: mz, intensity, neutral loss, mask."""
    n = cfg.top_n_peaks
    out_mz = np.zeros(n, dtype=np.float32)
    out_int = np.zeros(n, dtype=np.float32)
    out_nl = np.zeros(n, dtype=np.float32)
    out_mask = np.zeros(n, dtype=np.float32)
    if mz.size == 0:
        return out_mz, out_int, out_nl, out_mask

    k = min(n, int(mz.size))
    if mz.size > n:
        idx = np.argpartition(intensity, -k)[-k:]
        idx = idx[np.argsort(mz[idx])]
    else:
        idx = np.argsort(mz)
        k = int(idx.size)
    sel_mz = mz[idx][:k]
    sel_int = intensity[idx][:k]
    # Re-normalize after top-N so cosine is well scaled.
    s = float(np.max(sel_int)) if k else 0.0
    if s > 0:
        sel_int = sel_int / s
    prec = float(precursor_mz) if np.isfinite(precursor_mz) else 0.0
    nl = np.clip(prec - sel_mz, 0.0, None).astype(np.float32)
    out_mz[:k] = sel_mz
    out_int[:k] = sel_int
    out_nl[:k] = nl
    out_mask[:k] = 1.0
    return out_mz, out_int, out_nl, out_mask


def bin_spectrum(
    mz: np.ndarray,
    intensity: np.ndarray,
    mask: np.ndarray | None = None,
    *,
    cfg: Config,
) -> np.ndarray:
    """Max-pool intensities into a fixed ``(n_mz_bins,)`` histogram (XLA-static)."""
    out = np.zeros(cfg.n_mz_bins, dtype=np.float32)
    mz = _as_1d_float(mz)
    intensity = _as_1d_float(intensity)
    n = min(mz.size, intensity.size)
    if n == 0:
        return out
    mz = mz[:n]
    intensity = intensity[:n]
    if mask is not None:
        m = _as_1d_float(mask)[:n] > 0
        mz = mz[m]
        intensity = intensity[m]
    if mz.size == 0:
        return out
    width = cfg.mz_bin_width
    if width <= 0:
        return out
    bins = np.floor((mz - cfg.mz_bin_min) / width).astype(np.int32)
    valid = (bins >= 0) & (bins < cfg.n_mz_bins) & np.isfinite(intensity)
    if not np.any(valid):
        return out
    np.maximum.at(out, bins[valid], intensity[valid])
    peak = float(out.max())
    if peak > 0:
        out /= peak
    return out


def merge_peak_lists(
    peak_lists: Sequence[tuple[np.ndarray, np.ndarray]],
    precursor_mz: float,
    *,
    cfg: Config,
    bin_da: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge several cleaned spectra by max-pooling into ``bin_da`` buckets."""
    if not peak_lists:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    all_mz = []
    all_int = []
    for mz, inten in peak_lists:
        mz = _as_1d_float(mz)
        inten = _as_1d_float(inten)
        if mz.size:
            all_mz.append(mz)
            all_int.append(inten)
    if not all_mz:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)
    mz = np.concatenate(all_mz)
    inten = np.concatenate(all_int)
    mz, inten = clean_peaks(mz, inten, precursor_mz, cfg=cfg)
    if mz.size == 0:
        return mz, inten
    keys = np.round(mz / bin_da).astype(np.int64)
    order = np.argsort(keys)
    keys = keys[order]
    mz = mz[order]
    inten = inten[order]
    uniq, start = np.unique(keys, return_index=True)
    # max intensity per bin, mz = intensity-weighted mean
    merged_mz = np.empty(uniq.size, dtype=np.float32)
    merged_int = np.empty(uniq.size, dtype=np.float32)
    start = np.append(start, keys.size)
    for i in range(uniq.size):
        sl = slice(int(start[i]), int(start[i + 1]))
        w = inten[sl]
        merged_int[i] = float(w.max())
        sw = float(w.sum())
        merged_mz[i] = float((mz[sl] * w).sum() / sw) if sw > 0 else float(mz[sl].mean())
    peak = float(merged_int.max()) if merged_int.size else 0.0
    if peak > 0:
        merged_int = merged_int / peak
    return merged_mz, merged_int


def precursor_feature_vector(
    *,
    precursor_mz: float,
    neutral_mass: float,
    adduct: str | None,
    collision_energy: Any,
    ionization_mode: str | None,
    n_peaks: int,
    cfg: Config,
) -> np.ndarray:
    """Fixed-length precursor / metadata features (length ``cfg.precursor_feat_dim``)."""
    from src.chem import parse_adduct

    info = parse_adduct(adduct)
    ion = 1.0 if str(ionization_mode or "").lower().startswith("pos") else 0.0
    ce = mean_collision_energy(collision_energy)
    feat = np.array(
        [
            float(neutral_mass) / 1000.0,
            float(precursor_mz) / 1000.0,
            float(np.log1p(max(neutral_mass, 0.0))),
            float(info.abs_charge),
            float(info.n_molecules),
            float(ce) / 200.0,
            ion,
            float(n_peaks) / float(cfg.top_n_peaks),
        ],
        dtype=np.float32,
    )
    if feat.size < cfg.precursor_feat_dim:
        feat = np.pad(feat, (0, cfg.precursor_feat_dim - feat.size))
    return feat[: cfg.precursor_feat_dim]


def featurize_spectrum(
    mz: Any,
    intensity: Any,
    precursor_mz: float,
    adduct: str | None,
    *,
    cfg: Config,
    collision_energy: Any = None,
    ionization_mode: str | None = None,
) -> dict[str, np.ndarray | int | float]:
    """Clean one MS2 spectrum into the static tensors consumed by the model."""
    mz_arr = _as_1d_float(mz)
    int_arr = _as_1d_float(intensity)
    prec = float(precursor_mz) if precursor_mz is not None else 0.0
    mz_c, int_c = clean_peaks(mz_arr, int_arr, prec, cfg=cfg)
    if mz_c.size == 0:
        # Fall back to unfiltered peaks so we never emit an empty example.
        mz_c, int_c = mz_arr, int_arr
        finite = np.isfinite(mz_c) & np.isfinite(int_c)
        mz_c, int_c = mz_c[finite], int_c[finite]
    peak_mz, peak_int, peak_nl, peak_mask = top_n_pad(mz_c, int_c, prec, cfg=cfg)
    binned = bin_spectrum(peak_mz, peak_int, peak_mask, cfg=cfg)
    neutral = precursor_to_neutral_mass(prec, adduct)
    n_peaks = int(peak_mask.sum())
    feats = precursor_feature_vector(
        precursor_mz=prec,
        neutral_mass=neutral,
        adduct=adduct,
        collision_energy=collision_energy,
        ionization_mode=ionization_mode,
        n_peaks=n_peaks,
        cfg=cfg,
    )
    return {
        "peak_mz": peak_mz,
        "peak_intensity": peak_int,
        "peak_nl": peak_nl,
        "peak_mask": peak_mask,
        "binned": binned,
        "precursor_feat": feats,
        "adduct_id": int(adduct_to_id(adduct)),
        "adduct": str(adduct or ""),
        "ionization_mode": str(ionization_mode or ""),
        "neutral_mass": float(neutral),
        "precursor_mz": float(prec),
    }


def featurize_aggregated(
    spectra: Sequence[dict[str, Any]],
    *,
    cfg: Config,
    precursor_key: str = "precursor_mz",
    mz_key: str = "ms2_mzs",
    int_key: str = "ms2_normalized_intensities",
    adduct_key: str = "adduct",
    ce_key: str = "collision_energy_ev",
    ion_key: str = "ionization_mode",
) -> dict[str, np.ndarray | int | float]:
    """Merge a molecule's multi-energy spectra (same adduct) by max-intensity peak fusion."""
    if not spectra:
        raise ValueError("featurize_aggregated requires at least one spectrum")
    # Group by rounded precursor so mixed adducts are not blindly pooled.
    best: dict[str, np.ndarray | int | float] | None = None
    best_peaks = -1
    from collections import defaultdict

    groups: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in spectra:
        adduct = str(row.get(adduct_key) or "")
        prec = float(row.get(precursor_key) or 0.0)
        groups[(adduct, int(round(prec * 1000.0)))].append(row)

    merged_features: list[dict[str, np.ndarray | int | float]] = []
    for (adduct, _prec_key), rows in groups.items():
        peak_lists = []
        prec_vals = []
        ces = []
        ion = rows[0].get(ion_key)
        for row in rows:
            prec = float(row.get(precursor_key) or 0.0)
            prec_vals.append(prec)
            ces.append(row.get(ce_key))
            mz_c, int_c = clean_peaks(
                _as_1d_float(row.get(mz_key)),
                _as_1d_float(row.get(int_key)),
                prec,
                cfg=cfg,
            )
            peak_lists.append((mz_c, int_c))
        prec_mean = float(np.mean(prec_vals)) if prec_vals else 0.0
        mz_m, int_m = merge_peak_lists(peak_lists, prec_mean, cfg=cfg)
        feat = featurize_spectrum(
            mz_m,
            int_m,
            prec_mean,
            adduct,
            cfg=cfg,
            collision_energy=ces,
            ionization_mode=ion,
        )
        merged_features.append(feat)
        n = int(feat["peak_mask"].sum()) if isinstance(feat["peak_mask"], np.ndarray) else 0
        if n > best_peaks:
            best_peaks = n
            best = feat

    assert best is not None
    # Average binned spectra across adduct groups for a molecule-level view.
    if len(merged_features) > 1:
        stacked = np.stack([f["binned"] for f in merged_features], axis=0)
        best = dict(best)
        best["binned"] = stacked.max(axis=0).astype(np.float32)
        best["group_features"] = merged_features  # type: ignore[assignment]
    else:
        best = dict(best)
        best["group_features"] = merged_features  # type: ignore[assignment]
    return best


def stack_features(rows: Iterable[dict[str, Any]]) -> dict[str, np.ndarray]:
    """Stack a list of ``featurize_spectrum`` dicts into batched numpy arrays."""
    rows = list(rows)
    if not rows:
        raise ValueError("no features to stack")
    keys = [
        "peak_mz",
        "peak_intensity",
        "peak_nl",
        "peak_mask",
        "binned",
        "precursor_feat",
    ]
    out: dict[str, np.ndarray] = {k: np.stack([r[k] for r in rows], axis=0) for k in keys}
    out["adduct_id"] = np.asarray([int(r["adduct_id"]) for r in rows], dtype=np.int64)
    out["neutral_mass"] = np.asarray([float(r["neutral_mass"]) for r in rows], dtype=np.float32)
    out["precursor_mz"] = np.asarray([float(r["precursor_mz"]) for r in rows], dtype=np.float32)
    return out
