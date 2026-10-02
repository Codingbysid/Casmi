"""Binned P(true | modified cosine) and P(true | Tanimoto) for the V6 merge."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Conservative prior: cosine >= 0.8 maps to p ~= 0.92+ so V5 lock behavior is
# preserved if the empirical table is missing. Tanimoto is weaker than cosine.
DEFAULT_COSINE_X = (0.0, 0.30, 0.50, 0.65, 0.75, 0.80, 0.90, 1.00)
DEFAULT_COSINE_Y = (0.02, 0.06, 0.15, 0.35, 0.62, 0.92, 0.97, 0.995)
DEFAULT_TANI_X = (0.0, 0.15, 0.25, 0.35, 0.50, 0.70, 1.00)
DEFAULT_TANI_Y = (0.02, 0.06, 0.12, 0.22, 0.40, 0.62, 0.90)


def _interp(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.clip(np.interp(x, xp, fp), 0.0, 1.0).astype(np.float32)


@dataclass
class ScoreCalibration:
    cosine_x: np.ndarray
    cosine_y: np.ndarray
    tanimoto_x: np.ndarray
    tanimoto_y: np.ndarray
    adopt_soft_merge: bool = True
    keep_lock_above: float = 0.8
    neighbor_blend: float = 0.5
    class1_mrr_soft: float | None = None
    class1_mrr_lock: float | None = None

    @classmethod
    def default(cls) -> "ScoreCalibration":
        return cls(
            cosine_x=np.asarray(DEFAULT_COSINE_X, dtype=np.float64),
            cosine_y=np.asarray(DEFAULT_COSINE_Y, dtype=np.float64),
            tanimoto_x=np.asarray(DEFAULT_TANI_X, dtype=np.float64),
            tanimoto_y=np.asarray(DEFAULT_TANI_Y, dtype=np.float64),
        )

    def p_cosine(self, spec: np.ndarray | float) -> np.ndarray:
        return _interp(np.asarray(spec, dtype=np.float64), self.cosine_x, self.cosine_y)

    def p_tanimoto(self, tani: np.ndarray | float) -> np.ndarray:
        return _interp(np.asarray(tani, dtype=np.float64), self.tanimoto_x, self.tanimoto_y)

    def or_score(
        self,
        spec: np.ndarray | float,
        tani: np.ndarray | float,
        mass: np.ndarray | float | None = None,
    ) -> np.ndarray:
        """1 - (1-p_spec)*(1-p_fp), optionally multiplied by a mass Gaussian."""
        p_s = self.p_cosine(spec)
        p_f = self.p_tanimoto(tani)
        merged = 1.0 - (1.0 - p_s) * (1.0 - p_f)
        if mass is None:
            return merged.astype(np.float32)
        mass_arr = np.asarray(mass, dtype=np.float32)
        return (merged * mass_arr).astype(np.float32)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cosine": {"x": self.cosine_x.tolist(), "y": self.cosine_y.tolist()},
            "tanimoto": {"x": self.tanimoto_x.tolist(), "y": self.tanimoto_y.tolist()},
            "adopt_soft_merge": bool(self.adopt_soft_merge),
            "keep_lock_above": float(self.keep_lock_above),
            "neighbor_blend": float(self.neighbor_blend),
            "class1_mrr_soft": self.class1_mrr_soft,
            "class1_mrr_lock": self.class1_mrr_lock,
        }


def _from_payload(payload: dict[str, Any]) -> ScoreCalibration:
    cos = payload.get("cosine") or {}
    tani = payload.get("tanimoto") or {}
    cal = ScoreCalibration(
        cosine_x=np.asarray(cos.get("x") or DEFAULT_COSINE_X, dtype=np.float64),
        cosine_y=np.asarray(cos.get("y") or DEFAULT_COSINE_Y, dtype=np.float64),
        tanimoto_x=np.asarray(tani.get("x") or DEFAULT_TANI_X, dtype=np.float64),
        tanimoto_y=np.asarray(tani.get("y") or DEFAULT_TANI_Y, dtype=np.float64),
        adopt_soft_merge=bool(payload.get("adopt_soft_merge", True)),
        keep_lock_above=float(payload.get("keep_lock_above", 0.8)),
        neighbor_blend=float(payload.get("neighbor_blend", 0.5)),
        class1_mrr_soft=payload.get("class1_mrr_soft"),
        class1_mrr_lock=payload.get("class1_mrr_lock"),
    )
    if cal.class1_mrr_soft is not None and cal.class1_mrr_lock is not None:
        cal.adopt_soft_merge = float(cal.class1_mrr_soft) + 1e-9 >= float(cal.class1_mrr_lock)
    return cal


def calibration_search_paths(extra: Path | None = None) -> list[Path]:
    here = Path(__file__).resolve().parent
    root = here.parent
    paths = [
        extra,
        root / "data" / "calibration.json",
        here / "calibration_data.json",
        Path("/kaggle/working/src/calibration_data.json"),
        Path("/kaggle/working/data/calibration.json"),
    ]
    return [p for p in paths if p is not None]


def load_calibration(path: Path | str | None = None) -> ScoreCalibration:
    extra = Path(path) if path else None
    for cand in calibration_search_paths(extra):
        if cand.exists():
            try:
                payload = json.loads(cand.read_text())
                return _from_payload(payload)
            except Exception:
                continue
    return ScoreCalibration.default()


def save_calibration(cal: ScoreCalibration, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cal.to_dict(), indent=2))
