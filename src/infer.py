"""End-to-end retrieval + fingerprint inference + ranked SMILES export."""

from __future__ import annotations

import zlib
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
    heavy_atom_graph_key,
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
from src.retrieval import (
    MIN_MATCHED_PEAKS,
    MIN_QUERY_INTENSITY_FRACTION,
    StructureIndex,
    fingerprint_knn,
    load_external_structure_index,
    modified_cosine,
)
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
    # Reranker features / provenance. Defaults keep older constructors valid.
    frac: float = 0.0  # matched query-intensity fraction
    fcos: float = 0.0  # fingerprint cosine vs predicted fp
    ppm: float = 100.0  # |mass error| to the nearest plausible query mass
    exact_mass: float = 0.0  # candidate monoisotopic mass
    source: int = 0  # 1 train library, 2 external pool, 3 mass-shift analog
    spec_evaluated: bool = False  # a spectrum comparison actually ran


def _copy_hit(hit: "_Hit", *, key: str | None = None, score: float | None = None) -> "_Hit":
    return _Hit(
        hit.smiles,
        hit.inchikey14 if key is None else key,
        hit.score if score is None else float(score),
        spec=hit.spec,
        tani=hit.tani,
        mass=hit.mass,
        n_match=hit.n_match,
        frac=hit.frac,
        fcos=hit.fcos,
        ppm=hit.ppm,
        exact_mass=hit.exact_mass,
        source=hit.source,
        spec_evaluated=hit.spec_evaluated,
    )


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
    frac_all = np.zeros(cand_idx.size, dtype=np.float32)
    evaluated = np.zeros(cand_idx.size, dtype=bool)
    spec_cap = 512
    if use_spectral and index.peak_mz is not None and cand_idx.size:
        if cand_idx.size > spec_cap:
            pick = np.argpartition(-mass_sc, spec_cap - 1)[:spec_cap]
        else:
            pick = np.arange(cand_idx.size)
        evaluated[pick] = True
        spec_pick = np.zeros(pick.size, dtype=np.float32)
        match_pick = np.zeros(pick.size, dtype=np.float32)
        frac_pick = np.zeros(pick.size, dtype=np.float32)
        for g in groups:
            s, m, frac = modified_cosine(
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
                return_intensity_fraction=True,
            )
            better = s > spec_pick
            # Keep support stats when the gated score ties at zero, so the
            # intensity and peak-count gates below still see the raw match.
            take = better | ((s == spec_pick) & (m > match_pick))
            match_pick = np.where(take, m, match_pick)
            frac_pick = np.where(take, frac, frac_pick)
            spec_pick = np.maximum(spec_pick, s)
        # modified_cosine already zeros weak support. Repeat the gates here so
        # a one- or two-peak cosine cannot survive into the Class 1 lock.
        few_peaks = match_pick < float(MIN_MATCHED_PEAKS)
        weak_intensity = (match_pick > 0) & (frac_pick < float(MIN_QUERY_INTENSITY_FRACTION))
        spec_pick = np.where(few_peaks | weak_intensity, np.float32(0.0), spec_pick)
        spec[pick] = spec_pick
        n_match[pick] = match_pick
        frac_all[pick] = frac_pick

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
    ppm = np.nan_to_num(dm / np.clip(cand_mass, 1e-6, None) * 1e6, nan=100.0)
    smiles = index.smiles[cand_idx]
    keys = index.inchikey14[cand_idx]
    source = 2 if is_class2 else 1
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
                frac=float(frac_all[i]),
                fcos=float(fcos[i]),
                ppm=float(ppm[i]),
                exact_mass=float(cand_mass[i]),
                source=source,
                spec_evaluated=bool(evaluated[i]),
            )
        )
    hits.sort(key=lambda h: (-h.spec, -h.score) if use_spectral else (-h.score,))
    return hits


def _skeleton_key(hit: _Hit) -> str:
    """Tautomer-canonical InChIKey14, so keto/enol forms share one slot."""
    return inchikey14_from_smiles(hit.smiles) or hit.inchikey14 or hit.smiles


def _graph_key(hit: _Hit) -> str:
    """Heavy-atom graph key; equal canonical keys imply equal graph keys."""
    return heavy_atom_graph_key(hit.smiles) or hit.inchikey14 or hit.smiles


def _merge_hits(*groups: list[_Hit], top_k: int) -> list[_Hit]:
    """Reference dedup: canonicalize every hit, keep the best score per key."""
    best: dict[str, _Hit] = {}
    for group in groups:
        for hit in group:
            if not hit.smiles:
                continue
            key = _skeleton_key(hit)
            prev = best.get(key)
            if prev is None or hit.score > prev.score:
                best[key] = _copy_hit(hit, key=key)
    return sorted(best.values(), key=lambda h: -h.score)[: int(top_k)]


def _emit_unique(
    ordered: list[_Hit],
    locked: list[_Hit],
    top_k: int,
    *,
    locked_keys: list[str] | None = None,
) -> list[_Hit]:
    """Walk ``ordered`` (best first) and emit the first hit of each skeleton.

    Produces the same sequence as ``_merge_hits`` on the pool minus aliases of
    locked hits, but canonicalizes a SMILES only when another emitted or locked
    hit shares its heavy-atom graph key. Everything else keeps its raw key.
    """
    if top_k <= 0:
        return []
    # graph key -> [[hit, canonical key or None], ...] over locked + emitted hits
    seen_by_graph: dict[str, list[list[Any]]] = {}
    for i, hit in enumerate(locked):
        key = locked_keys[i] if locked_keys is not None and i < len(locked_keys) else _skeleton_key(hit)
        seen_by_graph.setdefault(_graph_key(hit), []).append([hit, key])
    out: list[_Hit] = []
    for hit in ordered:
        if not hit.smiles:
            continue
        g = _graph_key(hit)
        mates = seen_by_graph.get(g)
        if mates:
            key = _skeleton_key(hit)
            duplicate = False
            for entry in mates:
                if entry[1] is None:
                    entry[1] = _skeleton_key(entry[0])
                if entry[1] == key:
                    duplicate = True
                    break
            if duplicate:
                continue
            mates.append([hit, key])
            out.append(_copy_hit(hit, key=key))
        else:
            seen_by_graph[g] = [[hit, None]]
            out.append(_copy_hit(hit, key=hit.inchikey14 or hit.smiles))
        if len(out) >= int(top_k):
            break
    return out


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
        hits.append(
            _Hit(
                smi,
                key,
                score,
                spec=0.0,
                tani=float(t[0]),
                mass=float(mass_sc[0]),
                fcos=float(fcos[0]),
                ppm=float(dm[0] / max(mass, qmass, 1.0) * 1e6),
                exact_mass=float(mass),
                source=3,
                spec_evaluated=False,
            )
        )
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
CLASS1_LOCK_MIN_PEAKS = 5
# Database-only hits have no MS2. Renormalize onto fingerprint + mass, then
# apply a small prior so an equally strong spectral match still ranks first.
CLASS2_FP_WEIGHT = 0.78
CLASS2_MASS_WEIGHT = 0.22
CLASS2_PRIOR = 0.92


def _is_class1_lock(hit: _Hit, cut: float) -> bool:
    """Lock only a high cosine that is also supported by at least 5 peaks."""
    return float(hit.spec) >= float(cut) and float(hit.n_match) >= float(CLASS1_LOCK_MIN_PEAKS)


def _linear_score(hit: _Hit, cfg: Config) -> float:
    """Compete score for unlocked candidates.

    Verified spectral matches (spec > 0 and at least 4 peaks) keep
    ``0.42 * spec + 0.46 * tanimoto + 0.12 * mass``. Class 2 and analogs have
    no spectrum, so they are scored on fingerprint and mass only. A 0.92 prior
    keeps a real spectral match ahead of an equal database hit, while a
    high-Tanimoto natural product still beats a noisy few-peak train decoy.
    """
    verified = float(hit.spec) > 0.0 and float(hit.n_match) >= float(MIN_MATCHED_PEAKS)
    if verified:
        w_s = float(cfg.w_spectral)
        w_f = float(cfg.w_fingerprint)
        w_m = float(cfg.w_mass)
        w = w_s + w_f + w_m
        if w <= 0:
            w = 1.0
        return float((w_s * hit.spec + w_f * hit.tani + w_m * hit.mass) / w)
    w = CLASS2_FP_WEIGHT + CLASS2_MASS_WEIGHT
    score = (CLASS2_FP_WEIGHT * float(hit.tani) + CLASS2_MASS_WEIGHT * float(hit.mass)) / w
    return float(score * CLASS2_PRIOR)


def _with_linear_score(hit: _Hit, cfg: Config) -> _Hit:
    return _copy_hit(hit, score=_linear_score(hit, cfg))


def _locked_hits(class1: list[_Hit], cut: float, top_k: int) -> list[_Hit]:
    locked = [h for h in class1 if _is_class1_lock(h, cut)]
    locked.sort(key=lambda h: (-h.spec, -h.score))
    return locked[:top_k]


def _unlocked_pool(
    class1: list[_Hit],
    class2: list[_Hit],
    analog_hits: list[_Hit],
    locked: list[_Hit],
    cfg: Config,
) -> list[_Hit]:
    """Compete pool with the 0.150 linear score, best first (stable on ties)."""
    locked_ids = {id(h) for h in locked}
    pool = [_with_linear_score(h, cfg) for h in class1 if id(h) not in locked_ids]
    pool += [_with_linear_score(h, cfg) for h in class2]
    pool += [_with_linear_score(h, cfg) for h in analog_hits]
    pool.sort(key=lambda h: -h.score)
    return pool


def _window_features(
    window: list[_Hit],
    pool: list[_Hit],
    n_class2: int,
    n_locked: int,
    context: dict[str, Any] | None,
) -> np.ndarray:
    from src.rerank import window_feature_matrix

    ctx = context or {}
    if not window:
        return window_feature_matrix(
            spec=np.zeros(0),
            n_match=np.zeros(0),
            frac=np.zeros(0),
            spec_evaluated=np.zeros(0),
            tani=np.zeros(0),
            fcos=np.zeros(0),
            mass_sc=np.zeros(0),
            ppm=np.zeros(0),
            source=np.zeros(0),
            lin_score=np.zeros(0),
            lin_rank=np.zeros(0),
            pool_max_tani=0.0,
            pool_max_spec=0.0,
            pool_max_fcos=0.0,
            pool_size=len(pool),
            n_class2=n_class2,
            n_locked=n_locked,
            query_mass=float(ctx.get("query_mass") or 0.0),
            query_n_peaks=int(ctx.get("query_n_peaks") or 0),
            n_groups=int(ctx.get("n_groups") or 1),
        )
    return window_feature_matrix(
        spec=np.array([h.spec for h in window]),
        n_match=np.array([h.n_match for h in window]),
        frac=np.array([h.frac for h in window]),
        spec_evaluated=np.array([1.0 if h.spec_evaluated else 0.0 for h in window]),
        tani=np.array([h.tani for h in window]),
        fcos=np.array([h.fcos for h in window]),
        mass_sc=np.array([h.mass for h in window]),
        ppm=np.array([h.ppm for h in window]),
        source=np.array([h.source for h in window]),
        lin_score=np.array([h.score for h in window]),
        lin_rank=np.arange(len(window), dtype=np.float32),
        pool_max_tani=max((h.tani for h in pool), default=0.0),
        pool_max_spec=max((h.spec for h in pool), default=0.0),
        pool_max_fcos=max((h.fcos for h in pool), default=0.0),
        pool_size=len(pool),
        n_class2=n_class2,
        n_locked=n_locked,
        query_mass=float(ctx.get("query_mass") or 0.0),
        query_n_peaks=int(ctx.get("query_n_peaks") or 0),
        n_groups=int(ctx.get("n_groups") or 1),
    )


TRACE_TAIL = 30


def _window_trace(window: list[_Hit], tail: list[_Hit], X: np.ndarray) -> dict[str, Any]:
    return {
        "window": {
            "smiles": [h.smiles for h in window],
            "raw_key": [h.inchikey14 for h in window],
            "exact_mass": np.array([h.exact_mass for h in window], dtype=np.float64),
            "source": np.array([h.source for h in window], dtype=np.int8),
            "lin_score": np.array([h.score for h in window], dtype=np.float32),
            "spec": np.array([h.spec for h in window], dtype=np.float32),
            "n_match": np.array([h.n_match for h in window], dtype=np.float32),
            "tani": np.array([h.tani for h in window], dtype=np.float32),
            "features": np.asarray(X, dtype=np.float32),
        },
        "tail": {
            "smiles": [h.smiles for h in tail[:TRACE_TAIL]],
            "raw_key": [h.inchikey14 for h in tail[:TRACE_TAIL]],
            "exact_mass": np.array([h.exact_mass for h in tail[:TRACE_TAIL]], dtype=np.float64),
            "source": np.array([h.source for h in tail[:TRACE_TAIL]], dtype=np.int8),
            "lin_score": np.array([h.score for h in tail[:TRACE_TAIL]], dtype=np.float32),
        },
    }


def _merge_tiers(
    class1: list[_Hit],
    class2: list[_Hit],
    analog_hits: list[_Hit],
    cfg: Config,
    *,
    lock_cut: float | None = None,
    reranker=None,
    context: dict[str, Any] | None = None,
    trace: dict[str, Any] | None = None,
) -> list[_Hit]:
    """Lock cosine>=0.75 and >=5 peaks; compete the rest, including Class 2.

    Without a reranker this reproduces the 0.150 ordering: locked hits by
    cosine, then every unlocked candidate by the linear compete score, one
    slot per tautomer-canonical skeleton. With a reranker, only the top
    ``cfg.rerank_window`` unlocked candidates are reordered by the learned
    score; the tail keeps the baseline order and the locks never move.
    """
    cut = float(lock_cut if lock_cut is not None else getattr(cfg, "soft_merge_lock_cosine", CLASS1_LOCK_COSINE) or CLASS1_LOCK_COSINE)
    top_k = int(cfg.top_k)
    locked = _locked_hits(class1, cut, top_k)
    locked_keys = [_skeleton_key(h) for h in locked]
    remain = max(top_k - len(locked), 0)
    pool = _unlocked_pool(class1, class2, analog_hits, locked, cfg)
    ordered = pool
    if reranker is not None or trace is not None:
        window_n = max(int(getattr(cfg, "rerank_window", 200) or 200), 1)
        window = pool[:window_n]
        tail = pool[window_n:]
        n_class2 = sum(1 for h in pool if h.source == 2)
        X = _window_features(window, pool, n_class2, len(locked), context)
        if trace is not None:
            trace.update(_window_trace(window, tail, X))
            trace["locked"] = {
                "smiles": [h.smiles for h in locked],
                "key": list(locked_keys),
                "spec": [float(h.spec) for h in locked],
                "n_match": [float(h.n_match) for h in locked],
                "tani": [float(h.tani) for h in locked],
            }
            trace["pool_size"] = len(pool)
            trace["pool_key_crc"] = np.array(
                [zlib.crc32(str(h.inchikey14).encode()) for h in pool], dtype=np.uint32
            )
            trace["n_class2_pool"] = int(n_class2)
            trace["n_analog_pool"] = int(sum(1 for h in pool if h.source == 3))
            trace["n_weak_c1_pool"] = int(sum(1 for h in pool if h.source == 1))
            trace["window_n"] = int(window_n)
        if reranker is not None and window:
            from src.rerank import rerank_scores, reranked_order

            scores = rerank_scores(reranker, X)
            order = reranked_order(scores)
            window = [window[int(i)] for i in order]
            if trace is not None:
                trace["rerank_scores"] = np.asarray(scores, dtype=np.float32)
                trace["rerank_order"] = np.asarray(order, dtype=np.int32)
        ordered = window + tail
    competed = _emit_unique(ordered, locked, remain, locked_keys=locked_keys)
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
    trace_out: dict[str, dict[str, Any]] | None = None,
) -> dict[str, list[str]]:
    """Lock cosine>=0.75 with >=5 peaks; compete weak train, Class 2, and analogs.

    ``reranker`` (optional) reorders only the unlocked window; ``trace_out``
    (optional dict) receives the per-molecule candidate trace used by the
    validation harness so both arms can be scored on identical pools.
    """
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
    if external_index is not None:
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
        if bool(getattr(cfg, "use_mass_shift", True)):
            print("[analog] mass-shifted NP products compete with weak Class 1 and Class 2")
    if reranker is not None:
        print(
            f"[rank] reranker=active window={int(getattr(cfg, 'rerank_window', 200))} "
            "(locked Class 1 order preserved; unlocked window reordered by learned score)"
        )
    else:
        print("[rank] reranker=none (0.150 linear compete score)")
    for i, (mid, feat) in enumerate(molecule_features.items()):
        smiles, stats, trace = _rank_query(
            feat,
            pred_fps[mid],
            index,
            cfg,
            external_index=external_index,
            reranker=reranker,
            lock_cut=lock_cut,
            want_trace=trace_out is not None,
            decoder=decoder,
            tokenizer=tokenizer,
        )
        out[mid] = smiles
        if trace_out is not None and trace is not None:
            trace_out[mid] = trace
        if stats["n_locked"]:
            n_mol_locked += 1
        if external_index is not None:
            if stats["n_class2_hits"] == 0:
                n_c2_empty += 1
            else:
                n_c2 += 1
                n_c2_hits += stats["n_class2_hits"]
            if stats["n_analog_hits"]:
                n_shift += 1
        n_c2_placed += stats["n_class2_placed"]
        n_analog_placed += stats["n_analog_placed"]
        if (i + 1) % 50 == 0:
            print(
                f"[rank] {i+1}/{n} last_n={len(out[mid])} locked={stats['n_locked']} "
                f"class2_hits={stats['n_class2_hits']} shifted={stats['n_analog_placed']}"
            )
    print(
        f"[rank] molecules with locked Class 1 "
        f"(cosine>={lock_cut}, n_match>={CLASS1_LOCK_MIN_PEAKS}): {n_mol_locked}/{n}"
    )
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


def _rank_query(
    feat: dict[str, Any],
    pred_fp: np.ndarray,
    index: StructureIndex,
    cfg: Config,
    *,
    external_index: StructureIndex | None = None,
    reranker=None,
    lock_cut: float | None = None,
    want_trace: bool = False,
    decoder=None,
    tokenizer=None,
) -> tuple[list[str], dict[str, int], dict[str, Any] | None]:
    """Rank one molecule. Shared by production inference and the validation harness."""
    from src.analogs import expand_smiles

    cut = float(lock_cut if lock_cut is not None else getattr(cfg, "soft_merge_lock_cosine", CLASS1_LOCK_COSINE) or CLASS1_LOCK_COSINE)
    top_k = int(cfg.top_k)
    groups = feat.get("group_features") or [feat]
    qmass = float(feat.get("neutral_mass") or 0.0)
    class1 = _rank_one(feat, groups, pred_fp, index, cfg)
    n_locked = len(_locked_hits(class1, cut, top_k))
    class2: list[_Hit] = []
    analog_hits: list[_Hit] = []
    if external_index is not None:
        class2 = _rank_fingerprint_only(feat, groups, pred_fp, external_index, cfg)
        remain_after_lock = max(top_k - n_locked, 0)
        if remain_after_lock > 0 and bool(getattr(cfg, "use_mass_shift", True)):
            analog_hits = _rank_mass_shifted_analogs(feat, groups, pred_fp, external_index, cfg)
    peak_mask = feat.get("peak_mask")
    context = {
        "n_locked": int(n_locked),
        "query_mass": qmass,
        "query_n_peaks": int(np.sum(peak_mask)) if isinstance(peak_mask, np.ndarray) else 0,
        "n_groups": len(groups),
    }
    trace: dict[str, Any] | None = {"context": context} if want_trace else None
    merged = _merge_tiers(
        class1,
        class2,
        analog_hits,
        cfg,
        lock_cut=cut,
        reranker=reranker,
        context=context,
        trace=trace,
    )
    stats = {
        "n_locked": int(n_locked),
        "n_class1_hits": len(class1),
        "n_class2_hits": len(class2),
        "n_analog_hits": len(analog_hits),
        "n_class2_placed": sum(1 for h in merged if h.source == 2),
        "n_analog_placed": sum(1 for h in merged if h.source == 3),
        "n_merged": len(merged),
    }
    smiles = _hits_to_smiles(merged)[:top_k]
    if len(smiles) < top_k:
        analog_smi = expand_smiles(
            smiles[:5] or _hits_to_smiles(class1[:5]),
            query_mass=qmass,
            mass_ppm=max(float(cfg.mass_ppm), 20.0),
        )
        smiles = merge_unique_smiles(smiles, analog_smi, top_k=top_k)
    if len(smiles) < top_k and decoder is not None:
        denovo = _class3_denovo(feat, pred_fp, cfg, decoder, tokenizer)
        smiles = merge_unique_smiles(smiles, denovo, top_k=top_k)
    if len(smiles) < top_k:
        smiles = merge_unique_smiles(smiles, _fingerprint_pad(pred_fp, index, cfg, smiles), top_k=top_k)
    if len(smiles) < top_k:
        smiles = merge_unique_smiles(smiles, _nearest_library_smiles(index, qmass, top_k), top_k=top_k)
    smiles = smiles[:top_k]
    if trace is not None:
        trace["merged_smiles"] = _hits_to_smiles(merged)[:top_k]
        trace["merged_source"] = [int(h.source) for h in merged[:top_k]]
        trace["final_smiles"] = list(smiles)
        trace["stats"] = dict(stats)
    return smiles, stats, trace


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
    trace_out: dict[str, dict[str, Any]] | None = None,
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
    # The formula head stays off. The learned reranker is used when the caller
    # passes one (the notebook does so only after the promotion rule passes).
    formula_model = None
    print(f"[predict] reranker={'active' if reranker is not None else 'none'}")

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
        trace_out=trace_out,
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
