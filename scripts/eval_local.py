#!/usr/bin/env python3
"""Local Class 1 / Class 2 holdout so V6 ranking can be tuned without Kaggle submits.

Emits ``data/calibration.json`` with binned P(true | modified cosine) and
P(true | Tanimoto). Does not read test.parquet.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.calibration import (  # noqa: E402
    DEFAULT_COSINE_X,
    DEFAULT_COSINE_Y,
    DEFAULT_TANI_X,
    DEFAULT_TANI_Y,
    ScoreCalibration,
    save_calibration,
)
from src.chem import formula_to_mass, inchikey14_from_smiles, morgan_fingerprint  # noqa: E402
from src.config import get_config  # noqa: E402
from src.data import iter_train_row_groups  # noqa: E402
from src.infer import (  # noqa: E402
    _rank_fingerprint_only,
    _rank_one,
    build_library_from_frame,
    rank_molecules,
)
from src.metrics import mrr_at_k  # noqa: E402
from src.preprocessing import featurize_spectrum  # noqa: E402
from src.retrieval import StructureIndex  # noqa: E402

QUERY_LIBS = ("massbank", "gnps", "enveda-np-examples")
LIB_LIBS = ("enveda-180", "pluskal_ms2", "riken", "gnps", "massbank")
SANITY_LIB = "enveda-np-examples"
WEIGHTS = (0.16, 0.45, 0.39)  # Class 1 / 2 / 3 CASMI mix


def _ikey(value: object) -> str:
    s = str(value or "").strip()
    if not s or s.lower() in {"nan", "none"}:
        return ""
    return s.split("-")[0][:14]


def _pick_lib(available: dict[str, object], preferred: tuple[str, ...]) -> str | None:
    low = {str(k).lower(): k for k in available}
    for name in preferred:
        if name in low:
            return low[name]
    return next(iter(available), None)


def _row_to_feat(rec, cfg) -> dict:
    return featurize_spectrum(
        getattr(rec, "ms2_mzs"),
        getattr(rec, "ms2_normalized_intensities"),
        float(getattr(rec, "precursor_mz") or 0.0),
        getattr(rec, "adduct"),
        cfg=cfg,
        collision_energy=getattr(rec, "collision_energy_ev", None),
        ionization_mode=getattr(rec, "ionization_mode", None),
    )


def _empirical_curve(scores: list[float], labels: list[int], edges: np.ndarray) -> tuple[list[float], list[float]]:
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64)
    xs: list[float] = []
    ps: list[float] = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (s >= lo) & (s < hi) if hi < edges[-1] else (s >= lo) & (s <= hi)
        if not np.any(mask):
            continue
        xs.append(float(0.5 * (lo + hi)))
        ps.append(float(y[mask].mean()))
    if not xs:
        return [0.0, 1.0], [0.02, 0.9]
    if xs[0] > 0:
        xs.insert(0, 0.0)
        ps.insert(0, max(0.01, ps[0] * 0.5))
    if xs[-1] < 1:
        xs.append(1.0)
        ps.append(min(0.995, max(ps[-1] * 1.15, 0.9)))
    return xs, ps


def _blend_curve(
    emp_x: list[float],
    emp_y: list[float],
    prior_x,
    prior_y,
    *,
    use_empirical: bool,
    lock_floor: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    px = np.asarray(prior_x, dtype=np.float64)
    py = np.asarray(prior_y, dtype=np.float64)
    grid = np.unique(np.concatenate([px, np.asarray(emp_x, dtype=np.float64), [0.0, 0.8, 1.0]]))
    prior = np.interp(grid, px, py)
    if not use_empirical:
        y = prior
    else:
        emp = np.interp(grid, np.asarray(emp_x, dtype=np.float64), np.asarray(emp_y, dtype=np.float64))
        y = np.maximum(emp, prior)
    if lock_floor is not None:
        y = np.where(grid >= 0.8, np.maximum(y, lock_floor), y)
    return grid, np.clip(y, 0.0, 0.995)


def _load_meta(train_path: Path) -> pd.DataFrame:
    cols = ["ingest_lib", "inchikey14", "normalized_smiles", "molecular_formula"]
    print(f"[eval] reading {cols} from {train_path}")
    df = pd.read_parquet(train_path, columns=cols)
    df["inchikey14"] = df["inchikey14"].map(_ikey)
    df["ingest_lib"] = df["ingest_lib"].astype(str)
    df = df[df["inchikey14"] != ""]
    return df


def _sample_keys(meta: pd.DataFrame, n: int, rng: np.random.Generator, exclude: set[str] | None = None) -> list[str]:
    counts = meta.groupby("inchikey14")["ingest_lib"].nunique()
    multi = counts[counts >= 2].index.tolist()
    if exclude:
        multi = [k for k in multi if k not in exclude]
    rng.shuffle(multi)
    return multi[: int(n)]


def _collect_by_key_lib(cfg, keys: set[str], sanity_lib: str) -> tuple[dict, dict, list]:
    """Return holdout[key][lib]=row-feat, sanity feats, and raw rows for library frames."""
    hold: dict[str, dict[str, dict]] = defaultdict(dict)
    sanity_rows: list[dict] = []
    wanted = set(keys)
    cols = [
        "ingest_lib",
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
    n_seen = 0
    for df in iter_train_row_groups(cfg.train_path, columns=cols):
        ikeys = df["inchikey14"].astype(str).map(_ikey)
        libs = df["ingest_lib"].astype(str)
        take = ikeys.isin(wanted) | (libs == sanity_lib)
        if not take.any():
            continue
        sub = df.loc[take]
        for rec in sub.itertuples(index=False):
            key = _ikey(getattr(rec, "inchikey14"))
            lib = str(getattr(rec, "ingest_lib") or "")
            n_seen += 1
            if lib == sanity_lib:
                sanity_rows.append(rec)
            if key in wanted and lib not in hold[key]:
                try:
                    hold[key][lib] = {
                        "feat": _row_to_feat(rec, cfg),
                        "smiles": str(getattr(rec, "normalized_smiles") or ""),
                        "formula": str(getattr(rec, "molecular_formula") or ""),
                        "rec": rec,
                    }
                except Exception:
                    continue
    print(f"[eval] collected holdout keys={len(hold)} sanity_rows={len(sanity_rows)} scanned_hits={n_seen}")
    return hold, {}, sanity_rows


def _frame_from_records(records: list) -> pd.DataFrame:
    rows = []
    for rec in records:
        rows.append(
            {
                "normalized_smiles": str(getattr(rec, "normalized_smiles") or ""),
                "inchikey14": _ikey(getattr(rec, "inchikey14")),
                "molecular_formula": str(getattr(rec, "molecular_formula") or ""),
                "adduct": getattr(rec, "adduct", "[M+H]+"),
                "precursor_mz": float(getattr(rec, "precursor_mz") or 0.0),
                "ms2_mzs": getattr(rec, "ms2_mzs"),
                "ms2_normalized_intensities": getattr(rec, "ms2_normalized_intensities"),
                "num_peaks": int(getattr(rec, "num_peaks") or 0),
                "collision_energy_ev": getattr(rec, "collision_energy_ev", None),
                "ionization_mode": getattr(rec, "ionization_mode", None),
            }
        )
    return pd.DataFrame(rows)


def _keys_in_mass_windows(meta: pd.DataFrame, query_masses: list[float], exclude: set[str], cap: int) -> set[str]:
    uniq = meta.drop_duplicates("inchikey14")
    formulas = uniq["molecular_formula"].astype(str).tolist()
    keys = uniq["inchikey14"].astype(str).tolist()
    masses = np.array([formula_to_mass(f) for f in formulas], dtype=np.float64)
    wanted = np.zeros(len(masses), dtype=bool)
    for m in query_masses:
        if m <= 0:
            continue
        tol = max(0.02, m * 15e-6)
        wanted |= np.abs(masses - m) <= tol
    hit = [k for k, keep in zip(keys, wanted) if keep and k not in exclude]
    return set(hit[: int(cap)])


def _collect_rows_for_keys(cfg, keys: set[str]) -> pd.DataFrame:
    """One representative train row per InChIKey14."""
    if not keys:
        return pd.DataFrame()
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
    kept: dict[str, object] = {}
    for df in iter_train_row_groups(cfg.train_path, columns=cols):
        ikeys = df["inchikey14"].astype(str).map(_ikey)
        take = ikeys.isin(keys) & ~ikeys.isin(set(kept))
        if not take.any():
            continue
        for rec in df.loc[take].itertuples(index=False):
            key = _ikey(getattr(rec, "inchikey14"))
            if key and key not in kept:
                kept[key] = rec
        if len(kept) >= len(keys):
            break
    print(f"[eval] collected library rows={len(kept)} of {len(keys)} keys")
    return _frame_from_records(list(kept.values()))


def _np_index_from_smiles(smiles: list[str], cfg) -> StructureIndex:
    keys = []
    masses = []
    keep_smi = []
    for smi in smiles:
        key = inchikey14_from_smiles(smi)
        if not key:
            continue
        from src.chem import exact_mass_from_smiles

        mass = exact_mass_from_smiles(smi)
        if mass <= 0:
            continue
        keep_smi.append(smi)
        keys.append(key)
        masses.append(mass)
    fps = np.stack([morgan_fingerprint(s, n_bits=cfg.fp_bits) for s in keep_smi]) if keep_smi else np.zeros((0, cfg.fp_bits), dtype=np.uint8)
    return StructureIndex.build(keep_smi, keys, np.asarray(masses, dtype=np.float64), fps)


def evaluate(args: argparse.Namespace) -> ScoreCalibration:
    cfg = get_config()
    rng = np.random.default_rng(int(args.seed))
    meta = _load_meta(cfg.train_path)
    c1_keys = _sample_keys(meta, args.n_class1, rng)
    c2_keys = _sample_keys(meta, args.n_class2, rng, exclude=set(c1_keys))
    print(f"[eval] class1 keys={len(c1_keys)} class2 keys={len(c2_keys)}")

    hold, _, sanity_recs = _collect_by_key_lib(cfg, set(c1_keys) | set(c2_keys), SANITY_LIB)

    # Class 1 queries + other-lib library rows
    c1_queries: list[dict] = []
    c1_true: list[str] = []
    c1_lib_recs = []
    c1_cos: list[float] = []
    c1_tani: list[float] = []
    c1_lab: list[int] = []
    for key in c1_keys:
        by_lib = hold.get(key) or {}
        if len(by_lib) < 2:
            continue
        qlib = _pick_lib(by_lib, QUERY_LIBS)
        llib = _pick_lib({k: v for k, v in by_lib.items() if k != qlib}, LIB_LIBS)
        if qlib is None or llib is None:
            continue
        c1_queries.append(by_lib[qlib]["feat"])
        c1_true.append(key)
        c1_lib_recs.append(by_lib[llib]["rec"])

    query_masses = [float(f.get("neutral_mass") or 0.0) for f in c1_queries]
    distract_keys = _keys_in_mass_windows(meta, query_masses, exclude=set(c1_true), cap=int(args.max_distractors))
    distract = _collect_rows_for_keys(cfg, distract_keys)
    lib_frame = pd.concat([_frame_from_records(c1_lib_recs), distract], ignore_index=True)
    print(f"[eval] class1 library rows={len(lib_frame)}")
    c1_index = build_library_from_frame(lib_frame, cfg)

    key_to_fp = {str(k): c1_index.fingerprints[i].astype(np.float32) for i, k in enumerate(c1_index.inchikey14)}
    mol_feats = {}
    pred_map = {}
    for i, (feat, key) in enumerate(zip(c1_queries, c1_true)):
        mid = f"c1_{i}"
        mol_feats[mid] = feat
        pred_map[mid] = key_to_fp.get(key, np.full(cfg.fp_bits, 0.5, dtype=np.float32))
        hits = _rank_one(feat, [feat], pred_map[mid], c1_index, cfg)
        for h in hits:
            c1_cos.append(h.spec)
            c1_tani.append(h.tani)
            c1_lab.append(1 if h.inchikey14 == key else 0)

    ranked_lock = rank_molecules(mol_feats, pred_map, c1_index, cfg)
    pred_lists = [ranked_lock[f"c1_{i}"] for i in range(len(c1_queries))]
    mrr_lock = mrr_at_k(pred_lists, true_keys=c1_true, k=25)
    print(f"[eval] Class 1 cross-lib MRR@25 (lock+soft)={mrr_lock:.4f} n={len(c1_true)}")

    # Soft vs lock comparison: rerank with use_soft_merge False
    cfg_lock = get_config()
    cfg_lock.use_soft_merge = False
    ranked_hard = rank_molecules(mol_feats, pred_map, c1_index, cfg_lock)
    pred_hard = [ranked_hard[f"c1_{i}"] for i in range(len(c1_queries))]
    mrr_hard = mrr_at_k(pred_hard, true_keys=c1_true, k=25)
    print(f"[eval] Class 1 cross-lib MRR@25 (V5 hard-lock scoring)={mrr_hard:.4f}")

    # Class 2 structure holdout
    c2_queries = []
    c2_true = []
    c2_smi = []
    c2_tani_s: list[float] = []
    c2_lab: list[int] = []
    for key in c2_keys:
        by_lib = hold.get(key) or {}
        if not by_lib:
            continue
        lib = _pick_lib(by_lib, QUERY_LIBS + LIB_LIBS)
        if lib is None:
            continue
        c2_queries.append(by_lib[lib]["feat"])
        c2_true.append(key)
        c2_smi.append(by_lib[lib]["smiles"])

    # Train library without the holdout skeletons.
    c2_query_masses = [float(f.get("neutral_mass") or 0.0) for f in c2_queries]
    extra_keys = _keys_in_mass_windows(
        meta, c2_query_masses, exclude=set(c2_true) | set(c1_true), cap=int(args.max_distractors)
    )
    extra = _collect_rows_for_keys(cfg, extra_keys)
    c2_lib_frame = pd.concat([distract, extra], ignore_index=True) if len(extra) else distract.copy()
    if "inchikey14" in c2_lib_frame.columns:
        c2_lib_frame = c2_lib_frame[~c2_lib_frame["inchikey14"].isin(set(c2_true))]
    c2_train_index = build_library_from_frame(c2_lib_frame, cfg) if len(c2_lib_frame) else c1_index
    lotus_smi: list[str] = []
    lotus_path = ROOT / "data" / "class2_candidates.parquet"
    if lotus_path.exists():
        lotus = pd.read_parquet(lotus_path, columns=["smiles"])
        n_take = min(8000, len(lotus))
        lotus_smi = lotus["smiles"].astype(str).sample(n=n_take, random_state=int(args.seed)).tolist()
    np_index = _np_index_from_smiles(c2_smi + lotus_smi, cfg)
    print(f"[eval] class2 NP index n={len(np_index.smiles)}")

    mol2 = {}
    pred2 = {}
    for i, (feat, key, smi) in enumerate(zip(c2_queries, c2_true, c2_smi)):
        mid = f"c2_{i}"
        mol2[mid] = feat
        fp = morgan_fingerprint(smi, n_bits=cfg.fp_bits).astype(np.float32)
        pred2[mid] = fp
        hits = _rank_fingerprint_only(feat, [feat], fp, np_index, cfg)
        for h in hits:
            c2_tani_s.append(h.tani)
            c2_lab.append(1 if h.inchikey14 == key else 0)
    ranked_c2 = rank_molecules(mol2, pred2, c2_train_index, cfg, external_index=np_index)
    pred_c2_lists = [ranked_c2[f"c2_{i}"] for i in range(len(c2_queries))]
    mrr_c2 = mrr_at_k(pred_c2_lists, true_keys=c2_true, k=25) if c2_true else 0.0
    print(f"[eval] Class 2 structure-holdout MRR@25={mrr_c2:.4f} n={len(c2_true)}")

    # Sanity: enveda-np-examples
    sanity_mrr = 0.0
    if sanity_recs:
        sanity_df = _frame_from_records(sanity_recs[: int(args.n_sanity)])
        sanity_index = c1_index
        s_feats = {}
        s_pred = {}
        s_true = []
        for i, rec in enumerate(sanity_df.itertuples(index=False)):
            feat = featurize_spectrum(
                rec.ms2_mzs,
                rec.ms2_normalized_intensities,
                float(rec.precursor_mz or 0.0),
                rec.adduct,
                cfg=cfg,
                collision_energy=rec.collision_energy_ev,
                ionization_mode=rec.ionization_mode,
            )
            mid = f"s_{i}"
            s_feats[mid] = feat
            key = _ikey(rec.inchikey14)
            s_true.append(key)
            s_pred[mid] = morgan_fingerprint(str(rec.normalized_smiles), n_bits=cfg.fp_bits).astype(np.float32)
        ranked_s = rank_molecules(s_feats, s_pred, sanity_index, cfg)
        sanity_mrr = mrr_at_k([ranked_s[f"s_{i}"] for i in range(len(s_true))], true_keys=s_true, k=25)
        print(f"[eval] enveda-np-examples sanity MRR@25={sanity_mrr:.4f} n={len(s_true)}")
    else:
        print("[eval] no enveda-np-examples rows collected")

    cos_edges = np.array([0.0, 0.3, 0.5, 0.65, 0.75, 0.8, 0.9, 1.01])
    tani_edges = np.array([0.0, 0.15, 0.25, 0.35, 0.5, 0.7, 1.01])
    cx, cy = _empirical_curve(c1_cos, c1_lab, cos_edges)
    tx, ty = _empirical_curve(c1_tani + c2_tani_s, c1_lab + c2_lab, tani_edges)
    # Oracle fingerprints make Tanimoto a step at ~1.0; keep the conservative prior.
    tani_mid = [p for x, p in zip(tx, ty) if x < 0.7]
    tani_usable = bool(tani_mid) and max(tani_mid) >= 0.05
    cx_b, cy_b = _blend_curve(cx, cy, DEFAULT_COSINE_X, DEFAULT_COSINE_Y, use_empirical=True, lock_floor=0.92)
    tx_b, ty_b = _blend_curve(tx, ty, DEFAULT_TANI_X, DEFAULT_TANI_Y, use_empirical=tani_usable)
    adopt = float(mrr_lock) + 1e-9 >= float(mrr_hard)
    cal = ScoreCalibration(
        cosine_x=np.asarray(cx_b, dtype=np.float64),
        cosine_y=np.asarray(cy_b, dtype=np.float64),
        tanimoto_x=np.asarray(tx_b, dtype=np.float64),
        tanimoto_y=np.asarray(ty_b, dtype=np.float64),
        adopt_soft_merge=adopt,
        keep_lock_above=0.8,
        neighbor_blend=0.5,
        class1_mrr_soft=float(mrr_lock),
        class1_mrr_lock=float(mrr_hard),
    )
    mixed = WEIGHTS[0] * mrr_lock + WEIGHTS[1] * mrr_c2 + WEIGHTS[2] * 0.0
    print(
        f"[eval] weighted estimate 0.16*C1 + 0.45*C2 + 0.39*C3 = {mixed:.4f} "
        f"(C3 treated as 0; sanity={sanity_mrr:.4f})"
    )
    print(f"[eval] adopt_soft_merge={adopt} (soft {mrr_lock:.4f} vs hard {mrr_hard:.4f})")
    out = ROOT / "data" / "calibration.json"
    save_calibration(cal, out)
    src_copy = ROOT / "src" / "calibration_data.json"
    save_calibration(cal, src_copy)
    summary = {
        "class1_mrr_soft": mrr_lock,
        "class1_mrr_lock": mrr_hard,
        "class2_mrr": mrr_c2,
        "sanity_mrr": sanity_mrr,
        "weighted_c1_c2": mixed,
        "n_class1": len(c1_true),
        "n_class2": len(c2_true),
        "adopt_soft_merge": adopt,
    }
    (ROOT / "data" / "eval_local_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[eval] wrote {out} and {src_copy}")
    return cal


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n-class1", type=int, default=800)
    p.add_argument("--n-class2", type=int, default=1000)
    p.add_argument("--n-sanity", type=int, default=1184)
    p.add_argument("--max-distractors", type=int, default=12_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args()
    if args.quick:
        args.n_class1 = min(args.n_class1, 40)
        args.n_class2 = min(args.n_class2, 40)
        args.n_sanity = min(args.n_sanity, 40)
        args.max_distractors = min(args.max_distractors, 1500)
    evaluate(args)


if __name__ == "__main__":
    main()
