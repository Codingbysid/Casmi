"""Learned reranker for the unlocked candidate pool.

One fixed, deterministic ``HistGradientBoostingClassifier`` scores the top
``cfg.rerank_window`` unlocked candidates of a query. Locked Class 1 hits are
never touched. Every feature is available at test inference: gated spectral
similarity, fingerprint similarity against the *predicted* fingerprint,
observed-precursor mass error, candidate source, and within-query context.

Nothing here may see the answer's fingerprint, a formula-derived query mass,
candidate identity, or labels. ``FEATURE_NAMES`` is the contract that the
tests check.
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

FEATURE_NAMES: tuple[str, ...] = (
    "spec_cosine",  # gated modified cosine (0 when unsupported)
    "n_match",  # matched peaks
    "intensity_fraction",  # matched query intensity fraction
    "spec_evaluated",  # 1 if a spectrum comparison was actually run
    "tanimoto",  # vs predicted (ensemble + neighbor) fingerprint
    "fp_cosine",
    "mass_gaussian",
    "abs_ppm",  # to the nearest observed-precursor-derived query mass
    "is_class1",
    "is_class2",
    "is_analog",
    "linear_score",  # the 0.150 compete score
    "log_linear_rank",  # log1p(rank by linear score within the unlocked pool)
    "tanimoto_margin",  # tanimoto - pool max
    "spec_margin",  # spec - pool max
    "fp_cosine_margin",
    "log_pool_size",
    "log_n_class2",
    "n_locked",
    "log_query_mass",
    "query_n_peaks",
    "n_adduct_groups",
)

# +1 monotone increasing, -1 decreasing, 0 unconstrained.
MONOTONIC: dict[str, int] = {
    "spec_cosine": 1,
    "n_match": 1,
    "intensity_fraction": 1,
    "tanimoto": 1,
    "fp_cosine": 1,
    "mass_gaussian": 1,
    "abs_ppm": -1,
    "linear_score": 1,
    "log_linear_rank": -1,
    "tanimoto_margin": 1,
    "spec_margin": 1,
    "fp_cosine_margin": 1,
}

# One configuration. No search, no fallback estimator family.
RERANKER_PARAMS: dict[str, Any] = {
    "loss": "log_loss",
    "max_iter": 150,
    "learning_rate": 0.05,
    "max_leaf_nodes": 15,
    "max_depth": 4,
    "min_samples_leaf": 40,
    "l2_regularization": 1.0,
    "early_stopping": False,
    "random_state": 0,
}

MIN_TRAIN_QUERIES = 100
SOURCE_CLASS1 = 1
SOURCE_CLASS2 = 2
SOURCE_ANALOG = 3


def monotonic_constraints() -> list[int]:
    return [int(MONOTONIC.get(name, 0)) for name in FEATURE_NAMES]


def window_feature_matrix(
    *,
    spec: np.ndarray,
    n_match: np.ndarray,
    frac: np.ndarray,
    spec_evaluated: np.ndarray,
    tani: np.ndarray,
    fcos: np.ndarray,
    mass_sc: np.ndarray,
    ppm: np.ndarray,
    source: np.ndarray,
    lin_score: np.ndarray,
    lin_rank: np.ndarray,
    pool_max_tani: float,
    pool_max_spec: float,
    pool_max_fcos: float,
    pool_size: int,
    n_class2: int,
    n_locked: int,
    query_mass: float,
    query_n_peaks: int,
    n_groups: int,
) -> np.ndarray:
    """Assemble the (n, len(FEATURE_NAMES)) float32 feature matrix."""
    spec = np.asarray(spec, dtype=np.float32).ravel()
    n = spec.size
    if n == 0:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
    n_match = np.asarray(n_match, dtype=np.float32).ravel()
    frac = np.asarray(frac, dtype=np.float32).ravel()
    spec_evaluated = np.asarray(spec_evaluated, dtype=np.float32).ravel()
    tani = np.asarray(tani, dtype=np.float32).ravel()
    fcos = np.asarray(fcos, dtype=np.float32).ravel()
    mass_sc = np.asarray(mass_sc, dtype=np.float32).ravel()
    ppm = np.clip(np.nan_to_num(np.asarray(ppm, dtype=np.float64).ravel(), nan=100.0), 0.0, 100.0)
    source = np.asarray(source, dtype=np.int64).ravel()
    lin_score = np.asarray(lin_score, dtype=np.float32).ravel()
    lin_rank = np.asarray(lin_rank, dtype=np.float32).ravel()
    qmass = float(query_mass) if np.isfinite(query_mass) and query_mass > 0 else 0.0
    cols = [
        spec,
        n_match,
        frac,
        spec_evaluated,
        tani,
        fcos,
        mass_sc,
        ppm.astype(np.float32),
        (source == SOURCE_CLASS1).astype(np.float32),
        (source == SOURCE_CLASS2).astype(np.float32),
        (source == SOURCE_ANALOG).astype(np.float32),
        lin_score,
        np.log1p(np.clip(lin_rank, 0.0, None)).astype(np.float32),
        (tani - np.float32(pool_max_tani)).astype(np.float32),
        (spec - np.float32(pool_max_spec)).astype(np.float32),
        (fcos - np.float32(pool_max_fcos)).astype(np.float32),
        np.full(n, np.log1p(max(int(pool_size), 0)), dtype=np.float32),
        np.full(n, np.log1p(max(int(n_class2), 0)), dtype=np.float32),
        np.full(n, float(min(max(int(n_locked), 0), 3)), dtype=np.float32),
        np.full(n, np.log(qmass) if qmass > 0 else 0.0, dtype=np.float32),
        np.full(n, float(max(int(query_n_peaks), 0)), dtype=np.float32),
        np.full(n, float(max(int(n_groups), 1)), dtype=np.float32),
    ]
    X = np.stack(cols, axis=1).astype(np.float32)
    assert X.shape[1] == len(FEATURE_NAMES), X.shape
    return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)


@dataclass
class LearnedReranker:
    """Fitted booster plus the metadata needed to audit it."""

    model: Any
    feature_names: tuple[str, ...] = FEATURE_NAMES
    meta: dict[str, Any] = field(default_factory=dict)

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != len(self.feature_names):
            raise ValueError(f"expected (n, {len(self.feature_names)}) features, got {X.shape}")
        if X.shape[0] == 0:
            return np.zeros((0,), dtype=np.float32)
        proba = self.model.predict_proba(X)
        return np.asarray(proba[:, 1], dtype=np.float32)


def rerank_scores(reranker: Any, X: np.ndarray) -> np.ndarray:
    """Higher is better. Any object with ``predict(X) -> (n,)`` works (tests use stubs)."""
    scores = np.asarray(reranker.predict(np.asarray(X, dtype=np.float32)), dtype=np.float64).ravel()
    if scores.shape[0] != X.shape[0]:
        raise ValueError(f"reranker returned {scores.shape[0]} scores for {X.shape[0]} rows")
    return np.nan_to_num(scores, nan=-1.0, posinf=1.0, neginf=-1.0)


def reranked_order(scores: np.ndarray) -> np.ndarray:
    """Stable descending order so equal scores keep the baseline order."""
    return np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")


def booster_available() -> tuple[bool, str]:
    try:
        import sklearn
        from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: F401

        return True, str(sklearn.__version__)
    except Exception as exc:  # pragma: no cover - depends on the environment
        return False, str(exc)


def _new_booster():
    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(monotonic_cst=monotonic_constraints(), **RERANKER_PARAMS)


def query_balanced_weights(y: np.ndarray, query_id: np.ndarray) -> np.ndarray:
    """Each query contributes unit mass to its positives and unit mass to its negatives."""
    y = np.asarray(y, dtype=np.int64).ravel()
    qid = np.asarray(query_id).ravel()
    w = np.zeros(y.shape[0], dtype=np.float32)
    for q in np.unique(qid):
        rows = np.where(qid == q)[0]
        pos = rows[y[rows] > 0]
        neg = rows[y[rows] <= 0]
        if pos.size:
            w[pos] = 1.0 / float(pos.size)
        if neg.size:
            w[neg] = 1.0 / float(neg.size)
    return w


def fit_reranker(
    X: np.ndarray,
    y: np.ndarray,
    query_id: np.ndarray,
    *,
    min_queries: int = MIN_TRAIN_QUERIES,
    check_determinism: bool = True,
) -> tuple[LearnedReranker | None, dict[str, Any]]:
    """Fit the single configured booster. Returns ``(None, report)`` when ineligible."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64).ravel()
    qid = np.asarray(query_id).ravel()
    report: dict[str, Any] = {
        "n_rows": int(X.shape[0]),
        "n_pos": int((y > 0).sum()),
        "n_queries": int(np.unique(qid).size) if qid.size else 0,
        "params": dict(RERANKER_PARAMS),
        "feature_names": list(FEATURE_NAMES),
        "monotonic": monotonic_constraints(),
    }
    if X.ndim != 2 or X.shape[1] != len(FEATURE_NAMES):
        report["status"] = f"bad feature shape {X.shape}"
        return None, report
    if not np.isfinite(X).all():
        report["status"] = "non-finite features"
        return None, report
    pos_queries = np.unique(qid[y > 0]).size if qid.size else 0
    report["n_queries_with_positive"] = int(pos_queries)
    if pos_queries < int(min_queries):
        report["status"] = f"only {pos_queries} queries with a positive (< {min_queries})"
        return None, report
    ok, version = booster_available()
    report["sklearn"] = version
    if not ok:
        report["status"] = f"sklearn unavailable: {version}"
        return None, report
    w = query_balanced_weights(y, qid)
    t0 = time.time()
    model = _new_booster()
    model.fit(X, y, sample_weight=w)
    report["fit_seconds"] = round(time.time() - t0, 2)
    if check_determinism:
        probe = X[: min(5000, X.shape[0])]
        again = _new_booster()
        again.fit(X, y, sample_weight=w)
        a = model.predict_proba(probe)[:, 1]
        b = again.predict_proba(probe)[:, 1]
        report["deterministic"] = bool(np.allclose(a, b, atol=1e-6))
        report["max_refit_abs_diff"] = float(np.max(np.abs(a - b))) if a.size else 0.0
        if not report["deterministic"]:
            report["status"] = "refit mismatch"
            return None, report
    report["status"] = "fitted"
    report["n_iter"] = int(getattr(model, "n_iter_", RERANKER_PARAMS["max_iter"]))
    return LearnedReranker(model=model, meta=dict(report)), report


def save_reranker(reranker: LearnedReranker, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(reranker, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load_reranker(path: Path | str) -> LearnedReranker:
    with Path(path).open("rb") as fh:
        obj = pickle.load(fh)
    if not isinstance(obj, LearnedReranker):
        raise TypeError(f"{path} does not contain a LearnedReranker")
    if tuple(obj.feature_names) != FEATURE_NAMES:
        raise ValueError("reranker feature contract does not match FEATURE_NAMES")
    return obj
