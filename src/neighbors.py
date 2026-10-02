"""MS2Query-style spectral-neighbor fingerprints (binned cosine kNN)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.config import Config
from src.preprocessing import bin_spectrum


def _l2_normalize(x: np.ndarray, dtype=np.float32) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        n = float(np.linalg.norm(arr))
        if n < 1e-8:
            return np.zeros_like(arr, dtype=dtype)
        return (arr / n).astype(dtype, copy=False)
    n = np.linalg.norm(arr, axis=-1, keepdims=True)
    return (arr / np.clip(n, 1e-8, None)).astype(dtype, copy=False)


def _softmax(x: np.ndarray, temperature: float = 0.15) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64) / max(float(temperature), 1e-6)
    z = z - float(np.max(z))
    e = np.exp(z)
    s = float(e.sum())
    if s <= 0:
        return np.full(x.shape, 1.0 / max(x.size, 1), dtype=np.float32)
    return (e / s).astype(np.float32)


def _ion_key(value: Any) -> str:
    s = str(value or "").strip().lower()
    if s.startswith("neg") or s in {"-", "n"}:
        return "negative"
    if s.startswith("pos") or s in {"+", "p"}:
        return "positive"
    return s


def blend_fingerprints(
    spec2fp: np.ndarray,
    neighbor_fp: np.ndarray,
    alpha: float = 0.5,
) -> np.ndarray:
    """``fp = a * spec2fp + (1-a) * neighbor_fp``, clipped to [0, 1]."""
    a = float(np.clip(alpha, 0.0, 1.0))
    out = a * np.asarray(spec2fp, dtype=np.float32) + (1.0 - a) * np.asarray(neighbor_fp, dtype=np.float32)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


@dataclass
class SpectralNeighborIndex:
    """L2-normalized binned library spectra for chunked cosine kNN."""

    binned: np.ndarray  # (N, n_bins) float16/float32, L2-normalized
    fingerprints: np.ndarray  # (N, bits) uint8 / float
    ion_code: np.ndarray | None = None  # 0 unknown, 1 pos, 2 neg
    chunk: int = 4096

    @classmethod
    def from_structure_index(cls, index, cfg: Config, *, chunk: int = 4096) -> "SpectralNeighborIndex":
        if getattr(index, "peak_mz", None) is None:
            raise ValueError("structure index has no representative peaks")
        n = int(index.peak_mz.shape[0])
        binned = np.zeros((n, cfg.n_mz_bins), dtype=np.float32)
        for i in range(n):
            binned[i] = bin_spectrum(
                index.peak_mz[i],
                index.peak_intensity[i],
                index.peak_mask[i] if index.peak_mask is not None else None,
                cfg=cfg,
            )
            if (i + 1) % 50_000 == 0:
                print(f"[neighbor] binned {i+1}/{n}")
        binned = _l2_normalize(binned, dtype=np.float16)
        fps = np.asarray(index.fingerprints)
        ion = getattr(index, "ionization_mode", None)
        ion_code = np.zeros(n, dtype=np.int8)
        if ion is not None:
            for i, value in enumerate(ion):
                key = _ion_key(value)
                if key == "positive":
                    ion_code[i] = 1
                elif key == "negative":
                    ion_code[i] = 2
        print(f"[neighbor] index n={n} bins={binned.shape[1]} kNN cosine vs train spectra")
        return cls(binned=binned, fingerprints=fps, ion_code=ion_code, chunk=int(chunk))

    def neighbor_fp(
        self,
        query_binned: np.ndarray,
        query_ion: Any = None,
        *,
        k: int = 20,
        temperature: float = 0.15,
    ) -> np.ndarray:
        q = _l2_normalize(np.asarray(query_binned, dtype=np.float32).ravel(), dtype=np.float32)
        n = int(self.binned.shape[0])
        bits = int(self.fingerprints.shape[1]) if self.fingerprints.ndim == 2 else 0
        if n == 0 or bits < 8:
            return np.zeros((max(bits, 1),), dtype=np.float32)[:bits] if bits else np.zeros((0,), dtype=np.float32)
        scores = np.empty(n, dtype=np.float32)
        q32 = q.astype(np.float32, copy=False)
        for start in range(0, n, int(self.chunk)):
            sl = slice(start, min(start + int(self.chunk), n))
            block = np.asarray(self.binned[sl], dtype=np.float32)
            scores[sl] = block @ q32
        q_ion = _ion_key(query_ion) if query_ion is not None else ""
        if self.ion_code is not None and q_ion in {"positive", "negative"}:
            ban = 2 if q_ion == "positive" else 1
            scores[self.ion_code == ban] = -1.0
        k = min(int(k), n)
        pick = np.argpartition(-scores, k - 1)[:k]
        pick = pick[np.argsort(-scores[pick])]
        w = _softmax(scores[pick], temperature=temperature)
        neigh = np.asarray(self.fingerprints[pick], dtype=np.float32)
        return (w[:, None] * neigh).sum(axis=0).astype(np.float32)

    def blend_one(
        self,
        spec2fp: np.ndarray,
        feat: dict[str, Any],
        *,
        k: int = 20,
        alpha: float = 0.5,
    ) -> np.ndarray:
        nfp = self.neighbor_fp(
            feat.get("binned"),
            feat.get("ionization_mode"),
            k=k,
        )
        if nfp.size != np.asarray(spec2fp).size:
            return np.asarray(spec2fp, dtype=np.float32)
        return blend_fingerprints(spec2fp, nfp, alpha=alpha)


def blend_predicted_fingerprints(
    spec2fp: np.ndarray,
    features: list[dict[str, Any]],
    neighbor_index: SpectralNeighborIndex | None,
    *,
    k: int = 20,
    alpha: float = 0.5,
) -> np.ndarray:
    fps = np.asarray(spec2fp, dtype=np.float32)
    if neighbor_index is None or fps.ndim != 2:
        return fps
    out = np.empty_like(fps)
    for i, feat in enumerate(features):
        out[i] = neighbor_index.blend_one(fps[i], feat, k=k, alpha=alpha)
        if (i + 1) % 50 == 0:
            print(f"[neighbor] blended {i+1}/{len(features)} k={k} a={alpha}")
    print(f"[neighbor] neighbor fp k={k} blend_a={alpha} n={len(features)}")
    return out
