"""End-to-end retrieval + fingerprint inference + ranked SMILES export."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.chem import (
    exact_mass_from_smiles,
    formula_to_mass,
    inchikey14_from_smiles,
    morgan_fingerprint,
    plausible_neutral_masses,
    unpack_fingerprints,
)
from src.config import Config, get_config
from src.data import iter_train_row_groups, load_test
from src.model import Spec2FP
from src.preprocessing import featurize_aggregated, featurize_spectrum, stack_features
from src.ranker import (
    FALLBACK_SMILES,
    combine_scores,
    dedup_inchikey14,
    fingerprint_scores,
    mass_gaussian_score,
    merge_unique_smiles,
    official_molecule_ids,
    build_submission_frame,
    sanitize_guesses,
    write_submission_csv,
)
from src.retrieval import StructureIndex, fingerprint_knn, load_external_structure_index, modified_cosine
from src.tpu_trainer import _forward, _move, _pad_to_batch_size, get_device, load_model


def build_library_from_train(
    cfg: Config,
    *,
    max_row_groups: int | None = None,
    progress: bool = True,
) -> StructureIndex:
    """Stream train.parquet and keep one representative spectrum per InChIKey14.

    Two passes: (1) record the max peak count per skeleton, (2) featurize only
    those winning rows so we do not pay RDKit/numpy work 2.5M times.
    """
    cols = [
        "normalized_smiles",
        "inchikey14",
        "molecular_formula",
        "adduct",
        "precursor_mz",
        "ms2_mzs",
        "ms2_normalized_intensities",
        "num_peaks",
        "collision_energy_ev",
        "ionization_mode",
    ]
    smiles_map: dict[str, str] = {}
    formula_map: dict[str, str] = {}
    best_n: dict[str, int] = {}

    n_groups_done = 0
    for df in iter_train_row_groups(cfg.train_path, columns=["normalized_smiles", "inchikey14", "molecular_formula", "num_peaks"]):
        n_groups_done += 1
        ikeys = df["inchikey14"].astype(str).to_numpy()
        smiles = df["normalized_smiles"].astype(str).to_numpy()
        formulas = df["molecular_formula"].astype(str).to_numpy()
        npeaks = df["num_peaks"].fillna(0).astype(np.int32).to_numpy()
        for ikey, smi, formula, n_peaks in zip(ikeys, smiles, formulas, npeaks):
            if not ikey or ikey == "nan" or not smi or smi == "nan":
                continue
            prev = best_n.get(ikey, -1)
            if n_peaks > prev:
                best_n[ikey] = int(n_peaks)
                smiles_map[ikey] = smi
                formula_map[ikey] = formula
            elif ikey not in smiles_map:
                smiles_map[ikey] = smi
                formula_map[ikey] = formula
                best_n[ikey] = int(n_peaks)
        if progress:
            print(f"[library] pass1 row_group={n_groups_done} unique_skeletons={len(smiles_map)}")
        if max_row_groups is not None and n_groups_done >= max_row_groups:
            break

    filled: set[str] = set()
    best_feat: dict[str, dict[str, Any]] = {}
    n_groups_done = 0
    for df in iter_train_row_groups(cfg.train_path, columns=cols):
        n_groups_done += 1
        npeaks = df["num_peaks"].fillna(0).astype(np.int32).to_numpy()
        ikeys = df["inchikey14"].astype(str).to_numpy()
        take = np.zeros(len(df), dtype=bool)
        for i, (ikey, n_peaks) in enumerate(zip(ikeys, npeaks)):
            if ikey in filled:
                continue
            if best_n.get(ikey, -1) == int(n_peaks):
                take[i] = True
        if not take.any():
            continue
        sub = df.loc[take]
        for rec in sub.itertuples(index=False):
            ikey = str(getattr(rec, "inchikey14") or "")
            if ikey in filled:
                continue
            feat = featurize_spectrum(
                getattr(rec, "ms2_mzs"),
                getattr(rec, "ms2_normalized_intensities"),
                float(getattr(rec, "precursor_mz") or 0.0),
                getattr(rec, "adduct"),
                cfg=cfg,
                collision_energy=getattr(rec, "collision_energy_ev"),
                ionization_mode=getattr(rec, "ionization_mode"),
            )
            best_feat[ikey] = feat
            filled.add(ikey)
        if progress:
            print(f"[library] pass2 row_group={n_groups_done} featurized={len(filled)}/{len(smiles_map)}")
        if max_row_groups is not None and n_groups_done >= max_row_groups:
            break
        if len(filled) >= len(smiles_map):
            break

    keys = list(smiles_map.keys())
    smiles = [smiles_map[k] for k in keys]
    formulas = [formula_map[k] for k in keys]
    masses = np.array([formula_to_mass(f) for f in formulas], dtype=np.float64)
    # Fall back to representative precursor-derived mass if formula failed.
    for i, k in enumerate(keys):
        if masses[i] <= 0 and k in best_feat:
            masses[i] = float(best_feat[k]["neutral_mass"])

    fps = np.zeros((len(keys), cfg.fp_bits), dtype=np.uint8)
    for i, smi in enumerate(smiles):
        fps[i] = morgan_fingerprint(smi, n_bits=cfg.fp_bits, radius=cfg.morgan_radius)
        if progress and (i + 1) % 25000 == 0:
            print(f"[library] fingerprints {i+1}/{len(keys)}")

    peak_mz = np.zeros((len(keys), cfg.top_n_peaks), dtype=np.float32)
    peak_int = np.zeros((len(keys), cfg.top_n_peaks), dtype=np.float32)
    peak_mask = np.zeros((len(keys), cfg.top_n_peaks), dtype=np.float32)
    peak_nl = np.zeros((len(keys), cfg.top_n_peaks), dtype=np.float32)
    precursor_feat = np.full((len(keys), cfg.precursor_feat_dim), np.nan, dtype=np.float32)
    adduct_id = np.full(len(keys), -1, dtype=np.int64)
    for i, k in enumerate(keys):
        feat = best_feat.get(k)
        if feat is None:
            continue
        peak_mz[i] = feat["peak_mz"]
        peak_int[i] = feat["peak_intensity"]
        peak_mask[i] = feat["peak_mask"]
        peak_nl[i] = feat["peak_nl"]
        precursor_feat[i] = feat["precursor_feat"]
        adduct_id[i] = feat["adduct_id"]

    ion_mode = np.array(
        [str((best_feat.get(k) or {}).get("ionization_mode") or "") for k in keys],
        dtype=object,
    )
    return StructureIndex.build(
        smiles=smiles,
        inchikey14=keys,
        exact_mass=masses,
        fingerprints=fps,
        peak_mz=peak_mz,
        peak_intensity=peak_int,
        peak_mask=peak_mask,
        formulas=formulas,
        ionization_mode=ion_mode,
        peak_nl=peak_nl,
        precursor_feat=precursor_feat,
        adduct_id=adduct_id,
    )


def build_library_from_frame(df: pd.DataFrame, cfg: Config) -> StructureIndex:
    """Small-data path used by tests: unique InChIKey14 from an in-memory frame."""
    smiles_map: dict[str, str] = {}
    formula_map: dict[str, str] = {}
    best_n: dict[str, int] = {}
    best_feat: dict[str, dict[str, Any]] = {}
    for rec in df.itertuples(index=False):
        ikey = str(getattr(rec, "inchikey14") or "")
        smi = str(getattr(rec, "normalized_smiles") or "")
        if not ikey or not smi:
            continue
        n_peaks = int(getattr(rec, "num_peaks") or 0)
        if ikey not in smiles_map:
            smiles_map[ikey] = smi
            formula_map[ikey] = str(getattr(rec, "molecular_formula") or "")
            best_n[ikey] = -1
        if n_peaks < best_n[ikey]:
            continue
        feat = featurize_spectrum(
            getattr(rec, "ms2_mzs"),
            getattr(rec, "ms2_normalized_intensities"),
            float(getattr(rec, "precursor_mz") or 0.0),
            getattr(rec, "adduct"),
            cfg=cfg,
            collision_energy=getattr(rec, "collision_energy_ev", None),
            ionization_mode=getattr(rec, "ionization_mode", None),
        )
        best_n[ikey] = n_peaks
        best_feat[ikey] = feat
    keys = list(smiles_map)
    smiles = [smiles_map[k] for k in keys]
    masses = np.array([formula_to_mass(formula_map[k]) for k in keys], dtype=np.float64)
    for i, k in enumerate(keys):
        if masses[i] <= 0:
            masses[i] = float(best_feat[k]["neutral_mass"])
    fps = np.stack(
        [morgan_fingerprint(s, n_bits=cfg.fp_bits, radius=cfg.morgan_radius) for s in smiles]
    )
    peak_mz = np.stack([best_feat[k]["peak_mz"] for k in keys])
    peak_int = np.stack([best_feat[k]["peak_intensity"] for k in keys])
    peak_mask = np.stack([best_feat[k]["peak_mask"] for k in keys])
    peak_nl = np.stack([best_feat[k]["peak_nl"] for k in keys])
    precursor_feat = np.stack([best_feat[k]["precursor_feat"] for k in keys])
    adduct_id = np.asarray([best_feat[k]["adduct_id"] for k in keys], dtype=np.int64)
    ion_mode = np.array(
        [str(best_feat[k].get("ionization_mode") or "") for k in keys],
        dtype=object,
    )
    return StructureIndex.build(
        smiles=smiles,
        inchikey14=keys,
        exact_mass=masses,
        fingerprints=fps,
        peak_mz=peak_mz,
        peak_intensity=peak_int,
        peak_mask=peak_mask,
        formulas=[formula_map[k] for k in keys],
        ionization_mode=ion_mode,
        peak_nl=peak_nl,
        precursor_feat=precursor_feat,
        adduct_id=adduct_id,
    )


@torch.no_grad()
def predict_fingerprints(
    model: Spec2FP | None,
    features: list[dict[str, Any]],
    cfg: Config,
    device=None,
    *,
    return_logits: bool = False,
) -> np.ndarray:
    """Sigmoid fingerprint probabilities, shape ``(N, fp_bits)``. Uniform if no model.

    ``return_logits=True`` skips the sigmoid so callers can average an ensemble.
    """
    n = len(features)
    if n == 0:
        return np.zeros((0, cfg.fp_bits), dtype=np.float32)
    if model is None:
        if return_logits:
            return np.zeros((n, cfg.fp_bits), dtype=np.float32)
        return np.full((n, cfg.fp_bits), 0.5, dtype=np.float32)
    if device is None:
        device, _ = get_device(prefer_xla=False)
    model = model.to(device)
    model.eval()
    stacked = stack_features(features)
    out = np.zeros((n, cfg.fp_bits), dtype=np.float32)
    bs = cfg.batch_size
    for start in range(0, n, bs):
        sl = slice(start, min(start + bs, n))
        batch = {
            "peak_mz": torch.from_numpy(stacked["peak_mz"][sl]),
            "peak_intensity": torch.from_numpy(stacked["peak_intensity"][sl]),
            "peak_nl": torch.from_numpy(stacked["peak_nl"][sl]),
            "peak_mask": torch.from_numpy(stacked["peak_mask"][sl]),
            "binned": torch.from_numpy(stacked["binned"][sl]),
            "precursor_feat": torch.from_numpy(stacked["precursor_feat"][sl]),
            "adduct_id": torch.from_numpy(stacked["adduct_id"][sl]),
        }
        real_n = int(batch["peak_mz"].shape[0])
        batch = _pad_to_batch_size(_move(batch, device), bs)
        logits = _forward(model, batch)[:real_n]
        arr = logits.detach().cpu().numpy().astype(np.float32)
        if return_logits:
            out[sl] = arr
        else:
            out[sl] = 1.0 / (1.0 + np.exp(-np.clip(arr, -20.0, 20.0)))
    return out


def predict_fingerprint_ensemble(
    models: list[Spec2FP | None] | Spec2FP | None,
    features: list[dict[str, Any]],
    cfg: Config,
    device=None,
) -> np.ndarray:
    """Average pre-sigmoid logits across trained seeds, then sigmoid."""
    if models is None:
        seq: list[Spec2FP | None] = [None]
    elif isinstance(models, list):
        seq = models
    else:
        seq = [models]
    acc = None
    n_ok = 0
    for model in seq:
        if model is None:
            continue
        logits = predict_fingerprints(model, features, cfg, device=device, return_logits=True)
        acc = logits if acc is None else acc + logits
        n_ok += 1
    if acc is None or n_ok == 0:
        return predict_fingerprints(None, features, cfg)
    mean = acc / float(n_ok)
    print(f"[ensemble] models={n_ok} averaged pre-sigmoid logits")
    return (1.0 / (1.0 + np.exp(-np.clip(mean, -20.0, 20.0)))).astype(np.float32)


@dataclass
class _Hit:
    smiles: str
    inchikey14: str
    score: float
    spec: float = 0.0
    tani: float = 0.0
    mass: float = 1.0
    n_match: float = 0.0


def _collect_query_masses(
    feat: dict[str, Any],
    extra_groups: list[dict[str, Any]],
    *,
    exhaustive: bool = False,
) -> list[float]:
    groups = extra_groups if extra_groups else [feat]
    masses: list[float] = []
    for g in groups:
        prec = float(g.get("precursor_mz") or 0.0)
        adduct = g.get("adduct") or feat.get("adduct")
        ion = g.get("ionization_mode") or feat.get("ionization_mode")
        masses.extend(
            plausible_neutral_masses(
                prec,
                adduct if isinstance(adduct, str) else None,
                ionization_mode=str(ion) if ion is not None else None,
                exhaustive=exhaustive,
            )
        )
        masses.append(float(g.get("neutral_mass") or 0.0))
    masses.append(float(feat.get("neutral_mass") or 0.0))
    out: list[float] = []
    seen: set[float] = set()
    for m in masses:
        if m <= 0 or not np.isfinite(m):
            continue
        key = round(float(m), 4)
        if key in seen:
            continue
        seen.add(key)
        out.append(float(m))
    return out or [0.0]


def _fingerprints_for(index: StructureIndex, cand_idx: np.ndarray, cfg: Config) -> np.ndarray:
    fps = getattr(index, "fingerprints", None)
    if fps is not None and fps.ndim == 2 and fps.shape[1] >= 8:
        return np.ascontiguousarray(fps[cand_idx])
    packed = getattr(index, "fp_packed", None)
    if packed is not None and getattr(packed, "ndim", 0) == 2 and packed.shape[0] == len(index.smiles):
        return unpack_fingerprints(np.ascontiguousarray(packed[cand_idx]), cfg.fp_bits).astype(np.uint8)
    return np.stack(
        [morgan_fingerprint(str(index.smiles[int(i)]), n_bits=cfg.fp_bits) for i in cand_idx]
    )


_CALIB = None


def _score_calibration():
    global _CALIB
    if _CALIB is None:
        from src.calibration import load_calibration

        _CALIB = load_calibration()
    return _CALIB


def _score_candidates(
    feat: dict[str, Any],
    extra_groups: list[dict[str, Any]],
    pred_fp: np.ndarray,
    index: StructureIndex,
    cand_idx: np.ndarray,
    cfg: Config,
    *,
    use_spectral: bool,
    is_class2: bool = False,
    reranker=None,
    formula_counts: np.ndarray | None = None,
) -> list[_Hit]:
    if cand_idx is None or cand_idx.size == 0:
        return []
    groups = extra_groups if extra_groups else [feat]
    masses = np.asarray(_collect_query_masses(feat, groups), dtype=np.float64)
    cand_mass = index.exact_mass[cand_idx]
    dm = np.min(np.abs(cand_mass[:, None] - masses[None, :]), axis=1)
    mass_sc = mass_gaussian_score(dm, cand_mass, cfg.mass_score_ppm_scale)

    spec = np.zeros(cand_idx.size, dtype=np.float32)
    n_match = np.zeros(cand_idx.size, dtype=np.float32)
    spec_cap = 512
    if use_spectral and index.peak_mz is not None and cand_idx.size:
        if cand_idx.size > spec_cap:
            pick = np.argpartition(-mass_sc, spec_cap - 1)[:spec_cap]
        else:
            pick = np.arange(cand_idx.size)
        spec_pick = np.zeros(pick.size, dtype=np.float32)
        match_pick = np.zeros(pick.size, dtype=np.float32)
        for g in groups:
            s, m = modified_cosine(
                g["peak_mz"],
                g["peak_intensity"],
                g["peak_mask"],
                float(g["precursor_mz"]),
                index.peak_mz[cand_idx[pick]],
                index.peak_intensity[cand_idx[pick]],
                index.peak_mask[cand_idx[pick]],
                cand_mass[pick].astype(np.float32),
                cfg.modified_cosine_mz_tol,
                return_n_match=True,
            )
            better = s > spec_pick
            match_pick = np.where(better, m, match_pick)
            spec_pick = np.maximum(spec_pick, s)
        spec[pick] = spec_pick
        n_match[pick] = match_pick

    fps = _fingerprints_for(index, cand_idx, cfg)
    tani, fcos = fingerprint_scores(
        pred_fp, fps, cfg.fp_threshold, soft_mix=float(getattr(cfg, "fp_soft_mix", 0.5))
    )
    spec = np.nan_to_num(spec, nan=0.0).astype(np.float32)
    tani = np.nan_to_num(tani, nan=0.0).astype(np.float32)
    fcos = np.nan_to_num(fcos, nan=0.0).astype(np.float32)
    mass_sc = np.nan_to_num(mass_sc, nan=0.0).astype(np.float32)
    if bool(getattr(cfg, "use_soft_merge", True)):
        spec_for_p = spec if use_spectral else np.zeros_like(spec)
        scores = _score_calibration().or_score(spec_for_p, tani, mass_sc)
    else:
        scores = combine_scores(spec, tani, mass_sc, cfg, fp_cosine=fcos)
    scores = np.nan_to_num(scores, nan=0.0)
    smiles = index.smiles[cand_idx]
    keys = index.inchikey14[cand_idx]
    hits: list[_Hit] = []
    for i in range(cand_idx.size):
        smi = str(smiles[i])
        key = str(keys[i] or "") or inchikey14_from_smiles(smi) or smi
        hits.append(
            _Hit(
                smiles=smi,
                inchikey14=key,
                score=float(scores[i]),
                spec=float(spec[i]),
                tani=float(tani[i]),
                mass=float(mass_sc[i]),
                n_match=float(n_match[i]),
            )
        )
    hits.sort(key=lambda h: (-h.spec, -h.score) if use_spectral else (-h.score,))
    return hits


def _merge_hits(*groups: list[_Hit], top_k: int) -> list[_Hit]:
    best: dict[str, _Hit] = {}
    for group in groups:
        for hit in group:
            if not hit.smiles:
                continue
            key = hit.inchikey14 or inchikey14_from_smiles(hit.smiles) or hit.smiles
            prev = best.get(key)
            if prev is None or hit.score > prev.score:
                best[key] = _Hit(
                    hit.smiles,
                    key,
                    hit.score,
                    spec=hit.spec,
                    tani=hit.tani,
                    mass=hit.mass,
                    n_match=hit.n_match,
                )
    return sorted(best.values(), key=lambda h: -h.score)[: int(top_k)]


def _hits_to_smiles(hits: list[_Hit]) -> list[str]:
    return [h.smiles for h in hits]


def _rank_one(
    feat: dict[str, Any],
    extra_groups: list[dict[str, Any]],
    pred_fp: np.ndarray,
    index: StructureIndex,
    cfg: Config,
    *,
    reranker=None,
    formula_counts: np.ndarray | None = None,
) -> list[_Hit]:
    masses = _collect_query_masses(feat, extra_groups, exhaustive=False)
    tight = index.query_masses(masses, cfg, use_fallback=False)
    return _score_candidates(
        feat,
        extra_groups,
        pred_fp,
        index,
        tight,
        cfg,
        use_spectral=True,
        is_class2=False,
        reranker=reranker,
        formula_counts=formula_counts,
    )


def _rank_fingerprint_only(
    feat: dict[str, Any],
    extra_groups: list[dict[str, Any]],
    pred_fp: np.ndarray,
    index: StructureIndex,
    cfg: Config,
    *,
    reranker=None,
    formula_counts: np.ndarray | None = None,
) -> list[_Hit]:
    """Class 2: mass window + fingerprint Tanimoto (no spectral library)."""
    masses = _collect_query_masses(feat, extra_groups, exhaustive=True)
    cand_idx = index.query_masses(
        masses,
        cfg,
        use_fallback=False,
        ppm=float(cfg.mass_ppm),
        abs_da=float(cfg.mass_abs_da),
        max_candidates=max(int(cfg.max_mass_candidates), 8192),
    )
    if cand_idx.size < cfg.top_k:
        extra = index.query_masses(
            masses,
            cfg,
            use_fallback=False,
            ppm=25.0,
            abs_da=0.03,
            max_candidates=8192,
        )
        if extra.size:
            cand_idx = np.unique(np.concatenate([cand_idx, extra])) if cand_idx.size else extra
    return _score_candidates(
        feat,
        extra_groups,
        pred_fp,
        index,
        cand_idx,
        cfg,
        use_spectral=False,
        is_class2=True,
        reranker=reranker,
        formula_counts=formula_counts,
    )


def _rank_mass_shifted_analogs(
    feat: dict[str, Any],
    extra_groups: list[dict[str, Any]],
    pred_fp: np.ndarray,
    index: StructureIndex,
    cfg: Config,
) -> list[_Hit]:
    """Retrieve mass-shifted parents, apply NP transforms, keep in-window products."""
    from src.analogs import expand_smiles, shifted_parent_masses

    masses = _collect_query_masses(feat, extra_groups, exhaustive=True)
    parent_masses = shifted_parent_masses(masses)
    if not parent_masses:
        return []
    cand_idx = index.query_masses(
        parent_masses,
        cfg,
        use_fallback=False,
        ppm=float(cfg.mass_ppm),
        abs_da=float(cfg.mass_abs_da),
        max_candidates=max(int(cfg.max_mass_candidates), 4096),
    )
    if cand_idx.size == 0:
        return []
    fps = _fingerprints_for(index, cand_idx, cfg)
    tani, _ = fingerprint_scores(
        pred_fp, fps, cfg.fp_threshold, soft_mix=float(getattr(cfg, "fp_soft_mix", 0.5))
    )
    k = min(40, int(cand_idx.size))
    pick = np.argpartition(-tani, k - 1)[:k]
    pick = pick[np.argsort(-tani[pick])]
    parents = [str(index.smiles[int(cand_idx[int(i)])]) for i in pick]
    qmass = float(feat.get("neutral_mass") or 0.0)
    products = expand_smiles(
        parents,
        query_mass=qmass,
        mass_ppm=float(cfg.mass_ppm),
        max_per_parent=8,
        max_total=40,
        in_window_only=True,
    )
    if not products:
        return []
    calib = _score_calibration() if bool(getattr(cfg, "use_soft_merge", True)) else None
    hits: list[_Hit] = []
    for smi in products:
        key = inchikey14_from_smiles(smi) or smi
        mass = exact_mass_from_smiles(smi)
        fp = morgan_fingerprint(smi, n_bits=cfg.fp_bits)
        t, fcos = fingerprint_scores(
            pred_fp, fp.reshape(1, -1), cfg.fp_threshold, soft_mix=float(getattr(cfg, "fp_soft_mix", 0.5))
        )
        dm = np.array([abs(mass - qmass) if mass > 0 and qmass > 0 else 50.0], dtype=np.float64)
        mass_sc = mass_gaussian_score(dm, np.array([max(mass, qmass, 1.0)]), cfg.mass_score_ppm_scale)
        if calib is not None:
            score = float(calib.or_score(0.0, t, mass_sc)[0])
        else:
            spec = np.array([0.0], dtype=np.float32)
            score = float(combine_scores(spec, t, mass_sc, cfg, fp_cosine=fcos)[0])
        hits.append(_Hit(smi, key, score, spec=0.0, tani=float(t[0]), mass=float(mass_sc[0])))
    hits.sort(key=lambda h: -h.score)
    return hits


def _class3_denovo(
    feat: dict[str, Any],
    pred_fp: np.ndarray,
    cfg: Config,
    decoder,
    tokenizer,
) -> list[str]:
    """Sample the SMILES decoder on CPU to fill remaining InChIKey14 slots."""
    if decoder is None or tokenizer is None:
        return []
    from src.models.smiles_decoder import generate_candidates_for_molecule

    cpu_decoder = decoder.to("cpu").eval()
    fp_t = torch.from_numpy(np.asarray(pred_fp, dtype=np.float32))
    adduct_idx = int(feat.get("adduct_id") or 0)
    return generate_candidates_for_molecule(
        cpu_decoder,
        fp_t,
        float(feat["neutral_mass"]),
        adduct_idx,
        tokenizer,
        num_samples=int(cfg.smiles_num_samples),
        mass_tolerance_ppm=float(cfg.smiles_mass_ppm),
        target_k=int(cfg.top_k),
        temp=float(cfg.smiles_temperature),
        top_p=float(cfg.smiles_top_p),
        max_len=int(cfg.smiles_gen_max_len),
        keep_mass_misses=True,
    )


def _fingerprint_pad(pred_fp: np.ndarray, index: StructureIndex, cfg: Config, already: list[str]) -> list[str]:
    have = {inchikey14_from_smiles(s) or s for s in already}
    need = max(int(cfg.top_k) - len(already), 0)
    if need <= 0:
        return []
    pick, _ = fingerprint_knn(
        pred_fp,
        index.fingerprints,
        k=min(need + 16, 64),
        threshold=cfg.fp_threshold,
    )
    out: list[str] = []
    for i in pick:
        key = str(index.inchikey14[int(i)])
        if key in have:
            continue
        out.append(str(index.smiles[int(i)]))
        if len(out) >= need:
            break
    return out


def _hits_from_smiles(
    smiles_list: list[str],
    pred_fp: np.ndarray,
    query_mass: float,
    cfg: Config,
    *,
    base_score: float = 0.15,
) -> list[_Hit]:
    hits: list[_Hit] = []
    for smi in smiles_list:
        key = inchikey14_from_smiles(smi) or smi
        mass = 0.0
        try:
            from src.chem import exact_mass_from_smiles

            mass = exact_mass_from_smiles(smi)
        except Exception:
            mass = 0.0
        fp = morgan_fingerprint(smi, n_bits=cfg.fp_bits)
        tani, fcos = fingerprint_scores(pred_fp, fp.reshape(1, -1), cfg.fp_threshold)
        dm = np.array([abs(mass - query_mass) if mass > 0 and query_mass > 0 else 50.0])
        mass_sc = mass_gaussian_score(dm, np.array([max(mass, query_mass, 1.0)]), cfg.mass_score_ppm_scale)
        spec = np.array([0.0], dtype=np.float32)
        score = float(combine_scores(spec, tani, mass_sc, cfg, fp_cosine=fcos)[0])
        hits.append(
            _Hit(
                smi,
                key,
                max(score, base_score * 0.01),
                spec=0.0,
                tani=float(tani[0]),
                mass=float(mass_sc[0]),
            )
        )
    return hits


CLASS1_LOCK_COSINE = 0.75


def _linear_score(hit: _Hit, cfg: Config) -> float:
    """Equal-footing merge score: 0.42 spec + 0.46 Tanimoto + 0.12 mass."""
    w_s = float(cfg.w_spectral)
    w_f = float(cfg.w_fingerprint)
    w_m = float(cfg.w_mass)
    w = w_s + w_f + w_m
    if w <= 0:
        w = 1.0
    return float((w_s * hit.spec + w_f * hit.tani + w_m * hit.mass) / w)


def _with_linear_score(hit: _Hit, cfg: Config) -> _Hit:
    return _Hit(
        hit.smiles,
        hit.inchikey14,
        _linear_score(hit, cfg),
        spec=hit.spec,
        tani=hit.tani,
        mass=hit.mass,
        n_match=hit.n_match,
    )


def _merge_tiers(
    class1: list[_Hit],
    class2: list[_Hit],
    analog_hits: list[_Hit],
    cfg: Config,
    *,
    lock_cut: float | None = None,
) -> list[_Hit]:
    """Lock cosine>=0.75 Class 1; compete weak Class 1, Class 2, and analogs by score."""
    cut = float(lock_cut if lock_cut is not None else getattr(cfg, "soft_merge_lock_cosine", CLASS1_LOCK_COSINE) or CLASS1_LOCK_COSINE)
    top_k = int(cfg.top_k)
    locked = [h for h in class1 if h.spec >= cut]
    locked.sort(key=lambda h: (-h.spec, -h.score))
    locked = locked[:top_k]
    locked_keys = {h.inchikey14 for h in locked}
    remain = max(top_k - len(locked), 0)
    weak_c1 = [_with_linear_score(h, cfg) for h in class1 if h.inchikey14 not in locked_keys]
    valid_c2 = [_with_linear_score(h, cfg) for h in class2 if h.inchikey14 not in locked_keys]
    analog_scored = [_with_linear_score(h, cfg) for h in analog_hits if h.inchikey14 not in locked_keys]
    competed = _merge_hits(weak_c1, valid_c2, analog_scored, top_k=remain)
    return (locked + competed)[:top_k]


def rank_molecules(
    molecule_features: dict[str, dict[str, Any]],
    pred_fps: dict[str, np.ndarray],
    index: StructureIndex,
    cfg: Config,
    *,
    external_index: StructureIndex | None = None,
    decoder=None,
    tokenizer=None,
    reranker=None,
    formula_model=None,
) -> dict[str, list[str]]:
    """Lock cosine>=0.75 Class 1; compete weak train, Class 2, and mass-shifted analogs."""
    from src.analogs import expand_smiles

    lock_cut = float(getattr(cfg, "soft_merge_lock_cosine", CLASS1_LOCK_COSINE) or CLASS1_LOCK_COSINE)
    out: dict[str, list[str]] = {}
    n = len(molecule_features)
    n_c2 = 0
    n_c2_hits = 0
    n_c2_empty = 0
    n_mol_locked = 0
    n_shift = 0
    n_c2_placed = 0
    n_analog_placed = 0
    c2_logged = False
    analog_logged = False
    for i, (mid, feat) in enumerate(molecule_features.items()):
        groups = feat.get("group_features") or [feat]
        qmass = float(feat.get("neutral_mass") or 0.0)
        class1 = _rank_one(feat, groups, pred_fps[mid], index, cfg)
        locked = [h for h in class1 if h.spec >= lock_cut]
        locked.sort(key=lambda h: (-h.spec, -h.score))
        locked = locked[: int(cfg.top_k)]
        if locked:
            n_mol_locked += 1
        locked_keys = {h.inchikey14 for h in locked}
        class2: list[_Hit] = []
        analog_hits: list[_Hit] = []
        if external_index is not None:
            if not c2_logged:
                print(
                    "[class2] dynamic mass window lookup via np.searchsorted on sorted exact_mass "
                    f"(n_lib={len(external_index.smiles)} "
                    f"mass={float(np.min(external_index.exact_mass)):.1f}-"
                    f"{float(np.max(external_index.exact_mass)):.1f} Da; "
                    "not prefiltered on test.parquet)"
                )
                n_lib = len(external_index.smiles)
                if 20_000 <= n_lib < 250_000:
                    print(
                        f"[class2] WARNING: n_lib={n_lib} looks like LOTUS-only (~133k). "
                        "V6 expects COCONUT ∪ LOTUS ≈ 400k."
                    )
                c2_logged = True
            raw_c2 = _rank_fingerprint_only(feat, groups, pred_fps[mid], external_index, cfg)
            if not raw_c2:
                n_c2_empty += 1
            class2 = [h for h in raw_c2 if h.inchikey14 not in locked_keys]
            if class2:
                n_c2 += 1
                n_c2_hits += len(class2)
            remain_after_lock = max(int(cfg.top_k) - len(locked), 0)
            if remain_after_lock > 0 and bool(getattr(cfg, "use_mass_shift", True)):
                if not analog_logged:
                    print(
                        "[analog] mass-shifted NP products compete with weak Class 1 and Class 2"
                    )
                    analog_logged = True
                analog_hits = [
                    h
                    for h in _rank_mass_shifted_analogs(
                        feat, groups, pred_fps[mid], external_index, cfg
                    )
                    if h.inchikey14 not in locked_keys
                ]
                if analog_hits:
                    n_shift += 1
        analog_keys = {h.inchikey14 for h in analog_hits}
        merged = _merge_tiers(class1, class2, analog_hits, cfg, lock_cut=lock_cut)
        shifted = [h for h in merged if h.inchikey14 in analog_keys]
        n_analog_placed += len(shifted)
        n_c2_placed += sum(1 for h in merged if h.inchikey14 in {x.inchikey14 for x in class2})
        smiles = _hits_to_smiles(merged)[: cfg.top_k]
        if len(smiles) < cfg.top_k:
            analog_smi = expand_smiles(
                smiles[:5] or _hits_to_smiles(class1[:5]),
                query_mass=qmass,
                mass_ppm=max(float(cfg.mass_ppm), 20.0),
            )
            smiles = merge_unique_smiles(smiles, analog_smi, top_k=cfg.top_k)
        if len(smiles) < cfg.top_k and decoder is not None:
            denovo = _class3_denovo(feat, pred_fps[mid], cfg, decoder, tokenizer)
            smiles = merge_unique_smiles(smiles, denovo, top_k=cfg.top_k)
        if len(smiles) < cfg.top_k:
            smiles = merge_unique_smiles(
                smiles,
                _fingerprint_pad(pred_fps[mid], index, cfg, smiles),
                top_k=cfg.top_k,
            )
        if len(smiles) < cfg.top_k:
            smiles = merge_unique_smiles(
                smiles,
                _nearest_library_smiles(index, qmass, cfg.top_k),
                top_k=cfg.top_k,
            )
        out[mid] = smiles[: cfg.top_k]
        if (i + 1) % 50 == 0:
            print(
                f"[rank] {i+1}/{n} last_n={len(out[mid])} locked={len(locked)} "
                f"class2_hits={len(class2)} shifted={len(shifted)}"
            )
    print(f"[rank] molecules with locked Class 1 (cosine>={lock_cut}): {n_mol_locked}/{n}")
    if external_index is not None:
        print(
            f"[rank] molecules with Class 2 mass hits: {n_c2}/{n} "
            f"empty_windows={n_c2_empty} "
            f"mean_hits={0 if n_c2 == 0 else n_c2_hits / n_c2:.1f} "
            f"mass_shifted={n_shift} class2_slots={n_c2_placed} analog_slots={n_analog_placed}"
        )
        if n_c2 == 0:
            print(
                "[rank] WARNING: 0 Class 2 mass hits from np.searchsorted. "
                "Check that class2_candidates.parquet is attached and spans 50-2000 Da."
            )
    return out


def featurize_test_molecules(test_df: pd.DataFrame, cfg: Config) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in test_df.to_dict(orient="records"):
        grouped[str(rec["molecule_id"])].append(rec)
    feats = {}
    n_fail = 0
    for mid, rows in grouped.items():
        try:
            feats[mid] = featurize_aggregated(rows, cfg=cfg)
        except Exception as exc:
            n_fail += 1
            row = rows[0]
            feats[mid] = featurize_spectrum(
                row.get("ms2_mzs"),
                row.get("ms2_normalized_intensities"),
                float(row.get("precursor_mz") or 0.0),
                row.get("adduct"),
                cfg=cfg,
                collision_energy=None,
                ionization_mode=row.get("ionization_mode"),
            )
            print(f"[warn] featurize_aggregated failed for {mid}: {exc}; used first spectrum")
    if n_fail:
        print(f"[warn] featurize fallback on {n_fail}/{len(grouped)} molecules")
    return feats


def _nearest_library_smiles(index: StructureIndex, mass: float, top_k: int) -> list[str]:
    n = int(index.smiles.shape[0])
    if n == 0:
        return []
    k = min(int(top_k), n)
    if not np.isfinite(mass) or mass <= 0:
        return [str(s) for s in index.smiles[:k].tolist()]
    err = np.abs(index.exact_mass - float(mass))
    pick = np.argpartition(err, k - 1)[:k]
    pick = pick[np.argsort(err[pick])]
    return [str(index.smiles[int(i)]) for i in pick]


def predict_test(
    cfg: Config | None = None,
    *,
    index: StructureIndex | None = None,
    model: Spec2FP | None = None,
    models: list[Spec2FP | None] | None = None,
    test_df: pd.DataFrame | None = None,
    output_path: Path | str | None = None,
    decoder=None,
    tokenizer=None,
    external_index: StructureIndex | None = None,
    reranker=None,
    formula_model=None,
) -> pd.DataFrame:
    cfg = cfg or get_config()
    if test_df is None:
        test_df = load_test(cfg)
    if index is None:
        index = build_library_from_train(cfg)
    ensemble = [m for m in (models or []) if m is not None]
    if not ensemble and model is None and cfg.checkpoint_path.exists():
        model = load_model(cfg.checkpoint_path, cfg)
    if not ensemble and model is not None:
        ensemble = [model]
    if decoder is None and cfg.smiles_checkpoint_path.exists():
        from src.smiles_train import load_decoder

        decoder, tokenizer = load_decoder(cfg.smiles_checkpoint_path, cfg, map_location="cpu")
    if external_index is None:
        external_index = load_external_structure_index(None, cfg)
    if external_index is None:
        print("[class2] no LOTUS/COCONUT table attached — ranking Class 1 only")
    # V4 reranker / formula head are disabled until Class 1 is recovered.
    reranker = None
    formula_model = None

    mol_feats = featurize_test_molecules(test_df, cfg)
    mids = list(mol_feats.keys())
    feat_list = [mol_feats[m] for m in mids]
    fps = predict_fingerprint_ensemble(ensemble, feat_list, cfg)
    if bool(getattr(cfg, "use_neighbor_fp", True)) and getattr(index, "peak_mz", None) is not None:
        try:
            from src.neighbors import SpectralNeighborIndex, blend_predicted_fingerprints

            nbr = SpectralNeighborIndex.from_structure_index(index, cfg)
            fps = blend_predicted_fingerprints(
                fps,
                feat_list,
                nbr,
                k=int(getattr(cfg, "neighbor_k", 20)),
                alpha=float(getattr(cfg, "neighbor_blend", 0.5)),
            )
        except Exception as exc:
            print(f"[neighbor] skipped: {exc}")
    pred_map = {m: fps[i] for i, m in enumerate(mids)}
    ranked = rank_molecules(
        mol_feats,
        pred_map,
        index,
        cfg,
        external_index=external_index,
        decoder=decoder,
        tokenizer=tokenizer,
        reranker=reranker,
        formula_model=formula_model,
    )
    ids = official_molecule_ids(cfg, test_df)
    n_empty = 0
    filled: dict[str, list[str]] = {}
    n_short = 0
    for mid in ids:
        guesses = sanitize_guesses(ranked.get(mid, []), top_k=cfg.top_k)
        if len(guesses) < cfg.top_k:
            n_short += 1
            feat = mol_feats.get(mid) or {}
            mass = float(feat.get("neutral_mass") or 0.0)
            guesses = merge_unique_smiles(
                guesses,
                sanitize_guesses(_nearest_library_smiles(index, mass, cfg.top_k), top_k=cfg.top_k),
                top_k=cfg.top_k,
            )
        if not guesses:
            n_empty += 1
            guesses = [FALLBACK_SMILES]
        filled[mid] = guesses
    if n_empty:
        print(f"[submit] filled {n_empty} empty rankings with fallback SMILES")
    if n_short:
        print(f"[submit] padded {n_short} rows to {cfg.top_k} guesses")
    sub = build_submission_frame(ids, filled, top_k=cfg.top_k)
    if output_path is not None:
        write_submission_csv(sub, output_path)
        from src.ranker import validate_submission

        validate_submission(output_path, cfg.sample_submission_path)
    return sub
