"""Learned candidate re-ranker (MS2Query-style).

Trains a small gradient-boosted tree on pairwise features:
modified cosine, Tanimoto, fingerprint cosine, mass error, matched peaks.
Falls back to the linear combine_scores weights if boosting is unavailable.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.config import Config
from src.ranker import combine_scores


FEATURE_NAMES = (
    "modified_cosine",
    "tanimoto",
    "fp_cosine",
    "mass_gaussian",
    "abs_ppm",
    "n_match_frac",
    "is_class2",
)


def candidate_feature_matrix(
    spec: np.ndarray,
    tani: np.ndarray,
    fcos: np.ndarray,
    mass_sc: np.ndarray,
    dm: np.ndarray,
    mass: np.ndarray,
    n_match: np.ndarray,
    is_class2: bool,
) -> np.ndarray:
    spec = np.asarray(spec, dtype=np.float32).ravel()
    tani = np.asarray(tani, dtype=np.float32).ravel()
    fcos = np.asarray(fcos, dtype=np.float32).ravel()
    mass_sc = np.asarray(mass_sc, dtype=np.float32).ravel()
    dm = np.asarray(dm, dtype=np.float64).ravel()
    mass = np.asarray(mass, dtype=np.float64).ravel()
    n_match = np.asarray(n_match, dtype=np.float32).ravel()
    ppm = np.abs(dm) / np.clip(mass, 1e-6, None) * 1e6
    flag = np.full(spec.shape, 1.0 if is_class2 else 0.0, dtype=np.float32)
    return np.stack(
        [
            spec,
            tani,
            fcos,
            mass_sc,
            np.clip(ppm, 0.0, 200.0).astype(np.float32),
            np.clip(n_match / 64.0, 0.0, 2.0),
            flag,
        ],
        axis=1,
    ).astype(np.float32)


def _new_booster():
    try:
        import xgboost as xgb

        return xgb.XGBClassifier(
            n_estimators=120,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.85,
            colsample_bytree=0.85,
            n_jobs=4,
            eval_metric="logloss",
            verbosity=0,
        )
    except Exception:
        pass
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_depth=4,
            max_iter=80,
            learning_rate=0.08,
            l2_regularization=0.1,
        )
    except Exception:
        return None


def train_reranker(
    X: np.ndarray,
    y: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
) -> Any | None:
    """Fit a classifier that predicts P(same InChIKey14). ``None`` if boosting is missing."""
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)
    if X.size == 0 or y.size < 20 or int(y.sum()) < 5:
        print("[rerank] not enough positive pairs, keeping linear weights")
        return None
    model = _new_booster()
    if model is None:
        print("[rerank] xgboost/sklearn missing, keeping linear weights")
        return None
    try:
        if sample_weight is not None:
            model.fit(X, y, sample_weight=np.asarray(sample_weight, dtype=np.float32))
        else:
            # Balance positives (true skeleton) vs mass-window decoys.
            n_pos = max(int(y.sum()), 1)
            n_neg = max(int((1 - y).sum()), 1)
            w = np.where(y > 0, n_neg / n_pos, 1.0).astype(np.float32)
            model.fit(X, y, sample_weight=w)
        print(f"[rerank] fitted {type(model).__name__} n={len(y)} pos={int(y.sum())}")
        return model
    except TypeError:
        model.fit(X, y)
        print(f"[rerank] fitted {type(model).__name__} n={len(y)} pos={int(y.sum())}")
        return model
    except Exception as exc:
        print(f"[rerank] fit failed ({exc}); keeping linear weights")
        return None


def predict_rerank_scores(model: Any | None, X: np.ndarray, cfg: Config, **linear_parts) -> np.ndarray:
    """Probability of a true match, or linear combine_scores if ``model`` is None."""
    if model is None:
        return combine_scores(
            linear_parts["spec"],
            linear_parts["tani"],
            linear_parts["mass_sc"],
            cfg,
            fp_cosine=linear_parts.get("fcos"),
        )
    X = np.asarray(X, dtype=np.float32)
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1].astype(np.float32)
        return proba.ravel().astype(np.float32)
    if hasattr(model, "decision_function"):
        z = np.asarray(model.decision_function(X), dtype=np.float64).ravel()
        return (1.0 / (1.0 + np.exp(-z))).astype(np.float32)
    pred = np.asarray(model.predict(X), dtype=np.float32).ravel()
    return pred


def train_reranker_from_index(index, cfg: Config, *, n_queries: int = 2500, max_cand: int = 48):
    """Self-retrieval pairs from the spectral library (true skeleton vs mass isobars)."""
    from src.ranker import fingerprint_scores, mass_gaussian_score
    from src.retrieval import modified_cosine

    n = int(index.smiles.shape[0])
    if n < 30 or index.peak_mz is None:
        return None
    rng = np.random.default_rng(int(cfg.seed))
    queries = rng.choice(n, size=min(int(n_queries), n), replace=False)
    Xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for i in queries:
        mass = float(index.exact_mass[i])
        cand = index.query_masses([mass], cfg, use_fallback=False)
        if cand.size < 2:
            continue
        if cand.size > max_cand:
            true = np.where(index.inchikey14[cand] == index.inchikey14[i])[0]
            rest = np.setdiff1d(np.arange(cand.size), true)
            take = rng.choice(rest, size=min(int(max_cand) - 1, rest.size), replace=False)
            keep = np.concatenate([true[:1], take]) if true.size else take
            cand = cand[keep]
        spec, n_match = modified_cosine(
            index.peak_mz[i],
            index.peak_intensity[i],
            index.peak_mask[i],
            mass,
            index.peak_mz[cand],
            index.peak_intensity[cand],
            index.peak_mask[cand],
            index.exact_mass[cand].astype(np.float32),
            cfg.modified_cosine_mz_tol,
            return_n_match=True,
        )
        qfp = index.fingerprints[i].astype(np.float32)
        tani, fcos = fingerprint_scores(qfp, index.fingerprints[cand], cfg.fp_threshold)
        dm = np.abs(index.exact_mass[cand] - mass)
        mass_sc = mass_gaussian_score(dm, index.exact_mass[cand], cfg.mass_score_ppm_scale)
        X = candidate_feature_matrix(
            spec, tani, fcos, mass_sc, dm, index.exact_mass[cand], n_match, is_class2=False
        )
        y = (index.inchikey14[cand] == index.inchikey14[i]).astype(np.int32)
        Xs.append(X)
        ys.append(y)
    if not Xs:
        return None
    return train_reranker(np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0))
