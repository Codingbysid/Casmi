"""Leak-free validation harness for the unlocked-pool reranker experiment.

Everything the Kaggle notebook needs to (1) split train skeletons that Spec2FP
never saw into disjoint tautomer-aware partitions, (2) build realistic query
spectra for two regimes, (3) generate candidate pools once with the production
ranking code, (4) fit the reranker on one partition, and (5) audit both arms on
identical pools with a promotion rule declared before the audit runs.

Regimes
-------
U (unseen structure): the query is the library representative spectrum; the
    skeleton and every tautomer alias are removed from the spectral library and
    the neighbor index; the truth is inserted into a validation copy of the
    external pool (answer-in-pool). Mirrors a hidden-test Class 2 molecule.
K (known structure): the query is a *different* acquisition of the skeleton;
    the library keeps its representative. Mirrors hidden-test Class 1.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import time
import zipfile
import zlib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from src.chem import (
    exact_mass_from_smiles,
    heavy_atom_graph_key,
    inchikey14_from_smiles,
    morgan_fingerprint,
    pack_fingerprints,
)
from src.config import Config
from src.data import fitting_subset_indices, iter_train_row_groups  # noqa: F401  (re-exported: one definition of the fit/holdout split)
from src.preprocessing import featurize_aggregated, featurize_spectrum
from src.retrieval import StructureIndex

PANEL_LIB = "enveda-np-examples"
PARTITIONS = ("RR_TRAIN", "DEV", "AUDIT", "PANEL")
DEFAULT_SIZES = {"RR_TRAIN": 1200, "DEV": 400, "AUDIT": 800, "PANEL": 400}
ALIAS_MASS_TOL_DA = 2e-3

# Declared before any audit number exists. The notebook prints this dict
# before the audit cell runs and the decision function below applies it.
PROMOTION_RULE: dict[str, Any] = {
    "audit_partition": "AUDIT",
    "min_audit_queries": 300,
    "min_delta_mrr": 0.010,  # pooled U+K paired mean(RR_rerank - RR_baseline)
    "ci_lower_must_exceed": 0.0,  # 95% bootstrap CI lower bound
    "min_known_regime_delta": -0.005,  # regime K must not lose Class 1
    "min_panel_delta": -0.02,  # enveda-np-examples panel, gating only if large enough
    "panel_min_n": 50,
    "bootstrap_resamples": 4096,
    "bootstrap_seed": 0,
    "dev_used_for_selection": False,
}


# --------------------------------------------------------------------------
# hashing / provenance
# --------------------------------------------------------------------------
def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path | str, *, full: bool = True, chunk: int = 1 << 22) -> str:
    """Full SHA-256, or head+tail+size for very large files when ``full=False``."""
    path = Path(path)
    h = hashlib.sha256()
    size = path.stat().st_size
    with path.open("rb") as fh:
        if full or size <= 2 * chunk:
            while True:
                block = fh.read(chunk)
                if not block:
                    break
                h.update(block)
        else:
            h.update(fh.read(chunk))
            fh.seek(max(size - chunk, 0))
            h.update(fh.read(chunk))
            h.update(str(size).encode())
    return h.hexdigest()


def file_fingerprint(path: Path | str, *, full_hash_limit: int = 512 * 1024 * 1024) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {"path": str(path), "exists": False}
    size = path.stat().st_size
    full = size <= int(full_hash_limit)
    return {
        "path": str(path),
        "exists": True,
        "bytes": int(size),
        "sha256" if full else "sha256_head_tail": sha256_file(path, full=full),
    }


MODEL_CONFIG_FIELDS = (
    "d_model",
    "n_heads",
    "n_transformer_layers",
    "n_conv_channels",
    "adduct_embed_dim",
    "dropout",
    "precursor_feat_dim",
    "top_n_peaks",
    "n_mz_bins",
    "fp_bits",
    "morgan_radius",
    "batch_size",
    "num_epochs",
    "learning_rate",
    "weight_decay",
    "max_grad_norm",
    "focal_gamma",
    "bce_pos_weight",
    "val_fraction",
    "use_representative_metadata",
    "restore_best_checkpoint",
    "max_train_spectra",
)


def model_config_hash(cfg: Config) -> str:
    payload = {k: getattr(cfg, k, None) for k in MODEL_CONFIG_FIELDS}
    return sha256_text(json.dumps(payload, sort_keys=True, default=str))[:16]


def fit_subset_hash(index: StructureIndex, take: np.ndarray) -> str:
    keys = [str(k) for k in index.inchikey14[np.asarray(take, dtype=np.int64)]]
    return sha256_text("\n".join(keys))[:16]


def config_to_jsonable(cfg: Config) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in cfg.__dict__.items():
        if isinstance(v, Path):
            out[k] = str(v)
        elif isinstance(v, (tuple, list)):
            out[k] = [str(x) if isinstance(x, Path) else x for x in v]
        elif isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        else:
            out[k] = str(v)
    return out


def checkpoint_extra(path: Path | str) -> dict[str, Any]:
    import torch

    try:
        blob = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        blob = torch.load(str(path), map_location="cpu")
    return dict(blob.get("extra") or {})


def checkpoint_is_complete(
    extra: dict[str, Any],
    cfg: Config,
    *,
    fit_hash: str,
    config_hash: str,
    seed: int | None = None,
) -> tuple[bool, str]:
    """Only a checkpoint from a finished run on the same fit subset may be reused."""
    if not extra:
        return False, "no extra metadata"
    if str(extra.get("stop_reason")) != "epochs_complete":
        return False, f"stop_reason={extra.get('stop_reason')}"
    if int(extra.get("completed_epochs", -1)) != int(cfg.num_epochs):
        return False, f"completed_epochs={extra.get('completed_epochs')} != {cfg.num_epochs}"
    if str(extra.get("fit_subset_hash", "")) != str(fit_hash):
        return False, "fit subset hash mismatch"
    if str(extra.get("model_config_hash", "")) != str(config_hash):
        return False, "model config hash mismatch"
    if seed is not None and int(extra.get("seed", -1)) != int(seed):
        return False, f"seed={extra.get('seed')} != {seed}"
    return True, "complete"


# --------------------------------------------------------------------------
# partitions
# --------------------------------------------------------------------------
@dataclass
class HoldoutPlan:
    partitions: dict[str, list[int]]
    canonical: dict[int, str]
    graph: dict[int, str]
    alias_rows: dict[int, list[int]]
    excluded: dict[str, int]
    panel_lib: str
    fit_hash: str
    n_fit: int
    n_library: int

    def masked_rows(self) -> set[int]:
        rows: set[int] = set()
        for part_rows in self.partitions.values():
            for r in part_rows:
                rows.update(self.alias_rows.get(r, [r]))
        return rows

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "partitions": {k: [int(x) for x in v] for k, v in self.partitions.items()},
            "canonical": {str(k): v for k, v in self.canonical.items()},
            "alias_rows": {str(k): [int(x) for x in v] for k, v in self.alias_rows.items() if len(v) > 1},
            "excluded": dict(self.excluded),
            "panel_lib": self.panel_lib,
            "fit_hash": self.fit_hash,
            "n_fit": int(self.n_fit),
            "n_library": int(self.n_library),
        }


def _rows_near_mass(index: StructureIndex, mass: float, tol: float = ALIAS_MASS_TOL_DA) -> np.ndarray:
    lo = int(np.searchsorted(index.sorted_mass, mass - tol, side="left"))
    hi = int(np.searchsorted(index.sorted_mass, mass + tol, side="right"))
    if hi <= lo:
        return np.zeros((0,), dtype=np.int64)
    return index.order[lo:hi].astype(np.int64, copy=False)


def alias_rows_for(index: StructureIndex, row: int, canonical: str, graph: str) -> list[int]:
    """Library rows sharing ``row``'s tautomer-canonical key (same formula, same graph)."""
    out = [int(row)]
    if not canonical:
        return out
    for other in _rows_near_mass(index, float(index.exact_mass[row])):
        other = int(other)
        if other == int(row):
            continue
        if heavy_atom_graph_key(str(index.smiles[other])) != graph:
            continue
        if inchikey14_from_smiles(str(index.smiles[other])) == canonical:
            out.append(other)
    return out


def plan_holdouts(
    index: StructureIndex,
    fit_take: np.ndarray,
    *,
    sizes: dict[str, int] | None = None,
    panel_rows: Iterable[int] = (),
    seed: int = 0,
    panel_lib: str = PANEL_LIB,
    progress: bool = True,
) -> HoldoutPlan:
    """Disjoint holdout partitions drawn outside the Spec2FP fitting subset.

    A row is eligible when it has representative peaks, a canonical key, and no
    tautomer alias inside the fitting subset. Partitions never share a
    canonical key. Panel rows (skeletons with ``panel_lib`` acquisitions) go to
    PANEL only.
    """
    sizes = dict(DEFAULT_SIZES, **(sizes or {}))
    n = int(index.smiles.shape[0])
    fit_take = np.asarray(fit_take, dtype=np.int64)
    in_fit = np.zeros(n, dtype=bool)
    in_fit[fit_take] = True
    complement = np.where(~in_fit)[0]
    rng = np.random.default_rng(int(seed))
    rng.shuffle(complement)
    panel_set = {int(r) for r in panel_rows}
    has_peaks = (
        index.peak_mask.sum(axis=1) > 0 if index.peak_mask is not None else np.ones(n, dtype=bool)
    )

    partitions: dict[str, list[int]] = {p: [] for p in PARTITIONS}
    canonical: dict[int, str] = {}
    graph: dict[int, str] = {}
    alias_rows: dict[int, list[int]] = {}
    excluded = {"no_peaks": 0, "no_canonical": 0, "alias_in_fit": 0, "duplicate_canonical": 0}
    used_canonical: set[str] = set()
    quota_non_panel = sizes["RR_TRAIN"] + sizes["DEV"] + sizes["AUDIT"]
    non_panel: list[int] = []
    panel: list[int] = []
    t0 = time.time()
    for i, row in enumerate(complement.tolist()):
        if len(non_panel) >= quota_non_panel and len(panel) >= sizes["PANEL"]:
            break
        is_panel = row in panel_set
        if is_panel and len(panel) >= sizes["PANEL"]:
            continue
        if not is_panel and len(non_panel) >= quota_non_panel:
            continue
        if not has_peaks[row]:
            excluded["no_peaks"] += 1
            continue
        smi = str(index.smiles[row])
        canon = inchikey14_from_smiles(smi)
        if not canon:
            excluded["no_canonical"] += 1
            continue
        if canon in used_canonical:
            excluded["duplicate_canonical"] += 1
            continue
        g = heavy_atom_graph_key(smi)
        aliases = alias_rows_for(index, row, canon, g)
        if any(in_fit[a] for a in aliases):
            excluded["alias_in_fit"] += 1
            continue
        used_canonical.add(canon)
        canonical[row] = canon
        graph[row] = g
        alias_rows[row] = aliases
        (panel if is_panel else non_panel).append(row)
        if progress and (i + 1) % 1000 == 0:
            print(
                f"[holdout] scanned {i+1} rows: non_panel={len(non_panel)}/{quota_non_panel} "
                f"panel={len(panel)}/{sizes['PANEL']} excluded={excluded} {time.time()-t0:.0f}s"
            )
    partitions["RR_TRAIN"] = non_panel[: sizes["RR_TRAIN"]]
    partitions["DEV"] = non_panel[sizes["RR_TRAIN"] : sizes["RR_TRAIN"] + sizes["DEV"]]
    partitions["AUDIT"] = non_panel[sizes["RR_TRAIN"] + sizes["DEV"] : quota_non_panel]
    partitions["PANEL"] = panel[: sizes["PANEL"]]
    plan = HoldoutPlan(
        partitions=partitions,
        canonical=canonical,
        graph=graph,
        alias_rows=alias_rows,
        excluded=excluded,
        panel_lib=panel_lib,
        fit_hash=fit_subset_hash(index, fit_take),
        n_fit=int(fit_take.size),
        n_library=n,
    )
    n_alias = sum(1 for v in alias_rows.values() if len(v) > 1)
    print(
        f"[holdout] partitions={ {k: len(v) for k, v in partitions.items()} } "
        f"rows_with_aliases={n_alias} excluded={excluded} fit_hash={plan.fit_hash}"
    )
    return plan


def check_plan_disjoint(plan: HoldoutPlan, fit_take: np.ndarray) -> None:
    fit = set(int(x) for x in np.asarray(fit_take).tolist())
    seen_rows: set[int] = set()
    seen_canon: set[str] = set()
    for name, rows in plan.partitions.items():
        for r in rows:
            if r in fit:
                raise AssertionError(f"{name} row {r} is inside the fitting subset")
            if r in seen_rows:
                raise AssertionError(f"row {r} appears in two partitions")
            seen_rows.add(r)
            canon = plan.canonical[r]
            if canon in seen_canon:
                raise AssertionError(f"canonical key {canon} appears in two partitions")
            seen_canon.add(canon)
            for a in plan.alias_rows.get(r, [r]):
                if a in fit:
                    raise AssertionError(f"alias row {a} of {name} row {r} is inside the fitting subset")


# --------------------------------------------------------------------------
# acquisitions (one streaming pass over train.parquet)
# --------------------------------------------------------------------------
ACQ_COLUMNS = [
    "ingest_lib",
    "instrument_type",
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


def panel_metadata(cfg: Config, panel_lib: str = PANEL_LIB) -> tuple[set[str], dict[str, int], int]:
    """Raw InChIKey14 set with ``panel_lib`` acquisitions, instrument counts, row count."""
    keys: set[str] = set()
    instruments: dict[str, int] = {}
    n_rows = 0
    for df in iter_train_row_groups(cfg.train_path, columns=["ingest_lib", "inchikey14", "instrument_type"]):
        mask = df["ingest_lib"].astype(str) == panel_lib
        if not mask.any():
            continue
        sub = df.loc[mask]
        n_rows += int(len(sub))
        keys.update(k for k in sub["inchikey14"].astype(str).tolist() if k and k != "nan")
        for inst, cnt in sub["instrument_type"].astype(str).value_counts().items():
            instruments[str(inst)] = instruments.get(str(inst), 0) + int(cnt)
    print(f"[panel] lib={panel_lib} rows={n_rows} skeletons={len(keys)} instruments={instruments}")
    return keys, instruments, n_rows


def _row_dict(rec) -> dict[str, Any]:
    return {
        "ingest_lib": str(getattr(rec, "ingest_lib", "") or ""),
        "instrument_type": str(getattr(rec, "instrument_type", "") or ""),
        "normalized_smiles": str(getattr(rec, "normalized_smiles", "") or ""),
        "inchikey14": str(getattr(rec, "inchikey14", "") or ""),
        "molecular_formula": str(getattr(rec, "molecular_formula", "") or ""),
        "adduct": getattr(rec, "adduct", None),
        "precursor_mz": float(getattr(rec, "precursor_mz", 0.0) or 0.0),
        "ms2_mzs": np.asarray(getattr(rec, "ms2_mzs"), dtype=np.float64),
        "ms2_normalized_intensities": np.asarray(getattr(rec, "ms2_normalized_intensities"), dtype=np.float64),
        "num_peaks": int(getattr(rec, "num_peaks", 0) or 0),
        "collision_energy_ev": getattr(rec, "collision_energy_ev", None),
        "ionization_mode": getattr(rec, "ionization_mode", None),
    }


def collect_acquisitions(
    cfg: Config,
    wanted_raw_keys: set[str],
    *,
    panel_lib: str | None = PANEL_LIB,
    panel_raw_keys: set[str] | None = None,
    max_rows_per_lib: int = 2,
    max_libs: int = 6,
    max_panel_rows: int = 12,
    max_row_groups: int | None = None,
    progress: bool = True,
) -> dict[str, dict[str, Any]]:
    """Representative (first row at max num_peaks, the library rule) plus alternates."""
    wanted = set(wanted_raw_keys)
    panel_keys = set(panel_raw_keys or ())
    acq: dict[str, dict[str, Any]] = {
        k: {"rep": None, "rep_n": -1, "rows": [], "panel_rows": []} for k in wanted
    }
    n_groups = 0
    t0 = time.time()
    for df in iter_train_row_groups(cfg.train_path, columns=ACQ_COLUMNS):
        n_groups += 1
        if max_row_groups is not None and n_groups > int(max_row_groups):
            break
        ikeys = df["inchikey14"].astype(str)
        libs = df["ingest_lib"].astype(str)
        take = ikeys.isin(wanted)
        if panel_lib and panel_keys:
            take |= ikeys.isin(panel_keys) & (libs == panel_lib)
        if not take.any():
            continue
        for rec in df.loc[take].itertuples(index=False):
            key = str(getattr(rec, "inchikey14") or "")
            entry = acq.get(key)
            if entry is None:
                continue
            row = _row_dict(rec)
            n_peaks = row["num_peaks"]
            if n_peaks > entry["rep_n"]:
                entry["rep"] = row
                entry["rep_n"] = n_peaks
            lib = row["ingest_lib"]
            libs_present = {r["ingest_lib"] for r in entry["rows"]}
            same_lib = sum(1 for r in entry["rows"] if r["ingest_lib"] == lib)
            if same_lib < max_rows_per_lib and (lib in libs_present or len(libs_present) < max_libs):
                entry["rows"].append(row)
            if panel_lib and lib == panel_lib and len(entry["panel_rows"]) < max_panel_rows:
                entry["panel_rows"].append(row)
        if progress and n_groups % 10 == 0:
            n_rep = sum(1 for e in acq.values() if e["rep"] is not None)
            print(f"[acq] row_group={n_groups} skeletons_with_rows={n_rep}/{len(acq)} {time.time()-t0:.0f}s")
    n_rep = sum(1 for e in acq.values() if e["rep"] is not None)
    print(f"[acq] done skeletons_with_rows={n_rep}/{len(acq)} in {time.time()-t0:.0f}s")
    return acq


# --------------------------------------------------------------------------
# queries
# --------------------------------------------------------------------------
@dataclass
class Truth:
    raw_key: str
    canonical: str
    graph: str
    smiles: str
    mass: float


@dataclass
class Query:
    qid: str
    partition: str
    regime: str  # "U" or "K"
    row: int
    truth: Truth
    feat: dict[str, Any]
    meta: dict[str, Any] = field(default_factory=dict)


def _featurize_row(row: dict[str, Any], cfg: Config) -> dict[str, Any]:
    return featurize_spectrum(
        row["ms2_mzs"],
        row["ms2_normalized_intensities"],
        float(row["precursor_mz"] or 0.0),
        row["adduct"],
        cfg=cfg,
        collision_energy=row.get("collision_energy_ev"),
        ionization_mode=row.get("ionization_mode"),
    )


def _same_peaks(feat: dict[str, Any], index: StructureIndex, row: int) -> bool:
    if index.peak_mz is None:
        return False
    a = np.asarray(feat["peak_mz"], dtype=np.float32)
    b = np.asarray(index.peak_mz[row], dtype=np.float32)
    ma = np.asarray(feat["peak_mask"], dtype=np.float32)
    mb = np.asarray(index.peak_mask[row], dtype=np.float32)
    if a.shape != b.shape or not np.array_equal(ma > 0, mb > 0):
        return False
    sel = ma > 0
    return bool(np.allclose(a[sel], b[sel], atol=1e-3))


def build_queries(
    plan: HoldoutPlan,
    acq: dict[str, dict[str, Any]],
    index: StructureIndex,
    cfg: Config,
) -> tuple[list[Query], dict[str, int]]:
    """Regime U and K queries for every partition. PANEL uses the aggregated test path."""
    queries: list[Query] = []
    counts = {
        "no_acquisition": 0,
        "rep_mismatch": 0,
        "no_alternate": 0,
        "panel_no_rows": 0,
        "panel_rep_in_panel_lib": 0,
        "featurize_failed": 0,
    }
    for part in PARTITIONS:
        for row in plan.partitions.get(part, []):
            raw_key = str(index.inchikey14[row])
            entry = acq.get(raw_key)
            if entry is None or entry["rep"] is None:
                counts["no_acquisition"] += 1
                continue
            truth = Truth(
                raw_key=raw_key,
                canonical=plan.canonical[row],
                graph=plan.graph[row],
                smiles=str(index.smiles[row]),
                mass=float(index.exact_mass[row]),
            )
            rep = entry["rep"]
            try:
                rep_feat = _featurize_row(rep, cfg)
            except Exception:
                counts["featurize_failed"] += 1
                continue
            rep_matches = _same_peaks(rep_feat, index, row)
            if not rep_matches:
                counts["rep_mismatch"] += 1
            rep_lib = rep["ingest_lib"]
            if part == "PANEL":
                panel_rows = entry["panel_rows"]
                if not panel_rows:
                    counts["panel_no_rows"] += 1
                    continue
                try:
                    feat_u = featurize_aggregated(panel_rows, cfg=cfg)
                except Exception:
                    counts["featurize_failed"] += 1
                    continue
                meta = {
                    "query_lib": plan.panel_lib,
                    "rep_lib": rep_lib,
                    "n_spectra": len(panel_rows),
                    "instrument": panel_rows[0].get("instrument_type", ""),
                    "rep_matches_library": rep_matches,
                }
                queries.append(Query(f"PANEL_U_{row}", part, "U", row, truth, feat_u, dict(meta)))
                if rep_lib != plan.panel_lib:
                    queries.append(Query(f"PANEL_K_{row}", part, "K", row, truth, feat_u, dict(meta)))
                else:
                    counts["panel_rep_in_panel_lib"] += 1
                continue
            meta_u = {
                "query_lib": rep_lib,
                "rep_lib": rep_lib,
                "n_spectra": 1,
                "instrument": rep.get("instrument_type", ""),
                "rep_matches_library": rep_matches,
            }
            queries.append(Query(f"{part}_U_{row}", part, "U", row, truth, rep_feat, meta_u))
            # Regime K: a different acquisition, preferring another library.
            alt_feat = None
            alt_meta: dict[str, Any] = {}
            candidates = sorted(
                entry["rows"],
                key=lambda r: (0 if r["ingest_lib"] != rep_lib else 1, -int(r["num_peaks"])),
            )
            for alt in candidates:
                try:
                    feat_k = _featurize_row(alt, cfg)
                except Exception:
                    continue
                if _same_peaks(feat_k, index, row):
                    continue
                if int(np.sum(feat_k["peak_mask"])) == 0:
                    continue
                alt_feat = feat_k
                alt_meta = {
                    "query_lib": alt["ingest_lib"],
                    "rep_lib": rep_lib,
                    "n_spectra": 1,
                    "instrument": alt.get("instrument_type", ""),
                    "cross_library": alt["ingest_lib"] != rep_lib,
                    "rep_matches_library": rep_matches,
                }
                break
            if alt_feat is None:
                counts["no_alternate"] += 1
                continue
            queries.append(Query(f"{part}_K_{row}", part, "K", row, truth, alt_feat, alt_meta))
    summary = {}
    for q in queries:
        summary[f"{q.partition}_{q.regime}"] = summary.get(f"{q.partition}_{q.regime}", 0) + 1
    print(f"[queries] built={len(queries)} by_partition_regime={summary} skipped={counts}")
    return queries, counts


# --------------------------------------------------------------------------
# validation libraries
# --------------------------------------------------------------------------
def masked_library(index: StructureIndex, drop_rows: set[int]) -> StructureIndex:
    """Copy of the spectral library without ``drop_rows`` (identities + aliases)."""
    n = int(index.smiles.shape[0])
    keep = np.ones(n, dtype=bool)
    if drop_rows:
        keep[np.fromiter((int(r) for r in drop_rows), dtype=np.int64)] = False

    def _take(arr):
        return None if arr is None else arr[keep]

    out = StructureIndex.build(
        smiles=index.smiles[keep],
        inchikey14=index.inchikey14[keep],
        exact_mass=index.exact_mass[keep],
        fingerprints=index.fingerprints[keep],
        peak_mz=_take(index.peak_mz),
        peak_intensity=_take(index.peak_intensity),
        peak_mask=_take(index.peak_mask),
        formulas=_take(index.formulas),
        fp_packed=_take(index.fp_packed),
        ionization_mode=_take(index.ionization_mode),
        peak_nl=_take(index.peak_nl),
        precursor_feat=_take(index.precursor_feat),
        adduct_id=_take(index.adduct_id),
    )
    print(f"[library] masked copy n={len(out.smiles)} (dropped {n - len(out.smiles)} rows)")
    return out


def external_with_truths(
    external: StructureIndex,
    truths: list[Truth],
    cfg: Config,
) -> tuple[StructureIndex, dict[str, int]]:
    """Validation copy of the Class 2 pool with the holdout truths inserted."""
    uniq: dict[str, Truth] = {}
    for t in truths:
        uniq.setdefault(t.canonical, t)
    smiles = [t.smiles for t in uniq.values()]
    keys = [t.raw_key for t in uniq.values()]
    masses = np.array([exact_mass_from_smiles(s) or t.mass for s, t in zip(smiles, uniq.values())], dtype=np.float64)
    stats = {"inserted": len(smiles), "already_present_raw_key": 0}
    present = set(str(k) for k in external.inchikey14.tolist())
    stats["already_present_raw_key"] = sum(1 for k in keys if k in present)
    if external.fp_packed is not None:
        fps_new = np.stack([morgan_fingerprint(s, n_bits=cfg.fp_bits, radius=cfg.morgan_radius) for s in smiles])
        packed = np.concatenate([external.fp_packed, pack_fingerprints(fps_new)], axis=0)
        fingerprints = external.fingerprints
    else:
        fps_new = np.stack([morgan_fingerprint(s, n_bits=cfg.fp_bits, radius=cfg.morgan_radius) for s in smiles])
        packed = None
        fingerprints = np.concatenate([external.fingerprints, fps_new], axis=0)
    formulas = None
    if external.formulas is not None:
        formulas = np.concatenate([external.formulas, np.array([""] * len(smiles), dtype=object)])
    out = StructureIndex.build(
        smiles=np.concatenate([external.smiles, np.array(smiles, dtype=object)]),
        inchikey14=np.concatenate([external.inchikey14, np.array(keys, dtype=object)]),
        exact_mass=np.concatenate([external.exact_mass, masses]),
        fingerprints=fingerprints,
        formulas=formulas,
        fp_packed=packed,
    )
    print(f"[class2] validation copy n={len(out.smiles)} inserted_truths={len(smiles)}")
    return out, stats


def natural_coverage(external: StructureIndex, truths: list[Truth]) -> dict[str, int]:
    """How many truths already sit in the production pool (raw key or tautomer alias)."""
    present = set(str(k) for k in external.inchikey14.tolist())
    raw_hits = 0
    alias_hits = 0
    for t in truths:
        if t.raw_key in present:
            raw_hits += 1
            continue
        for other in _rows_near_mass(external, t.mass):
            smi = str(external.smiles[int(other)])
            if heavy_atom_graph_key(smi) != t.graph:
                continue
            if inchikey14_from_smiles(smi) == t.canonical:
                alias_hits += 1
                break
    return {"n": len(truths), "raw_key_present": raw_hits, "tautomer_alias_present": alias_hits}


# --------------------------------------------------------------------------
# candidate generation (shared production code, optional fork pool)
# --------------------------------------------------------------------------
_WORKER: dict[str, Any] = {}


def _limit_threads() -> None:
    """Fork-pool initializer. Touches no torch state: a forked child that calls
    into an OpenMP runtime initialised by the parent can deadlock."""
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(1)
    except Exception:
        pass


def _rank_payload(payload: tuple[str, dict[str, Any], np.ndarray]) -> tuple[str, dict[str, Any]]:
    from src.infer import _rank_query

    qid, feat, fp = payload
    t0 = time.time()
    smiles, stats, trace = _rank_query(
        feat,
        fp,
        _WORKER["index"],
        _WORKER["cfg"],
        external_index=_WORKER["external"],
        reranker=None,
        want_trace=True,
    )
    return qid, {"final_smiles": smiles, "stats": stats, "trace": trace, "seconds": time.time() - t0}


class _PoolStalled(RuntimeError):
    pass


def rank_queries(
    queries: list[Query],
    pred_fps: dict[str, np.ndarray],
    index: StructureIndex,
    external_index: StructureIndex | None,
    cfg: Config,
    *,
    workers: int = 1,
    time_budget_s: float | None = None,
    progress_every: int = 100,
    label: str = "",
    probe_timeout_s: float = 180.0,
    stall_s: float = 300.0,
    poll_s: float = 10.0,
) -> dict[str, dict[str, Any]]:
    """Baseline ranking + candidate trace for each query, in list order.

    Stops at ``time_budget_s`` (queries are passed in priority order). Uses a
    fork pool when ``workers > 1``. The pool must first finish one real query
    within ``probe_timeout_s`` and must then keep producing results at least
    every ``stall_s``; otherwise it is terminated and the remaining queries run
    sequentially in the parent, with the reason logged. The sequential path is
    the reference; the pool only changes wall time, never results.
    """
    results: dict[str, dict[str, Any]] = {}
    if not queries:
        return results
    _WORKER.update({"index": index, "external": external_index, "cfg": cfg})
    payloads = [(q.qid, q.feat, np.asarray(pred_fps[q.qid], dtype=np.float32)) for q in queries]
    t0 = time.time()
    n_total = len(payloads)

    def _over_budget() -> bool:
        return time_budget_s is not None and (time.time() - t0) > float(time_budget_s)

    def _progress() -> None:
        el = time.time() - t0
        print(f"[rank:{label}] {len(results)}/{n_total} elapsed={el:.0f}s rate={el / max(len(results), 1):.2f}s/query")

    def _store(qid: str, res: dict[str, Any]) -> None:
        results[qid] = res
        if len(results) % progress_every == 0:
            _progress()

    def _sequential(items) -> None:
        for payload in items:
            if _over_budget():
                print(f"[rank:{label}] time budget {time_budget_s:.0f}s reached after {len(results)}/{n_total}")
                break
            qid, res = _rank_payload(payload)
            _store(qid, res)

    use_pool = int(workers) > 1 and hasattr(os, "fork") and n_total > 1
    if use_pool:
        pool = None
        try:
            ctx = mp.get_context("fork")
            pool = ctx.Pool(int(workers), initializer=_limit_threads)
            # Probe: one real query must come back; a fork-unsafe runtime hangs here, not later.
            probe = pool.apply_async(_rank_payload, (payloads[0],))
            try:
                qid, res = probe.get(timeout=float(probe_timeout_s))
            except mp.TimeoutError as exc:
                raise _PoolStalled(f"probe query produced nothing in {probe_timeout_s:.0f}s") from exc
            _store(qid, res)
            # chunksize must stay 1: larger chunks make imap_unordered return a
            # plain generator without the ``next(timeout)`` used below.
            it = pool.imap_unordered(_rank_payload, payloads[1:], chunksize=1)
            last_result = time.time()
            while True:
                if _over_budget():
                    print(f"[rank:{label}] time budget {time_budget_s:.0f}s reached after {len(results)}/{n_total}")
                    break
                try:
                    qid, res = it.next(timeout=float(poll_s))
                except StopIteration:
                    break
                except mp.TimeoutError:
                    if time.time() - last_result > float(stall_s):
                        raise _PoolStalled(f"no result for {stall_s:.0f}s after {len(results)}/{n_total}")
                    continue
                last_result = time.time()
                _store(qid, res)
        except Exception as exc:  # pragma: no cover - environment dependent
            print(f"[rank:{label}] fork pool abandoned ({type(exc).__name__}: {exc}); continuing sequentially")
            if pool is not None:
                pool.terminate()
                pool.join()
                pool = None
            remaining = [p for p in payloads if p[0] not in results]
            _sequential(remaining)
        finally:
            if pool is not None:
                pool.terminate()
                pool.join()
    else:
        _sequential(payloads)
    el = time.time() - t0
    print(f"[rank:{label}] done {len(results)}/{n_total} in {el:.0f}s ({el / max(len(results), 1):.2f}s/query)")
    return results


# --------------------------------------------------------------------------
# labels, arms, metrics
# --------------------------------------------------------------------------
def same_skeleton(smiles: str, raw_key: str | None, truth: Truth) -> bool:
    """Metric identity: raw key match, or tautomer-canonical match (graph key prefilter)."""
    if raw_key and raw_key == truth.raw_key:
        return True
    if heavy_atom_graph_key(smiles) != truth.graph:
        return False
    return inchikey14_from_smiles(smiles) == truth.canonical


def reciprocal_rank_smiles(smiles_list: list[str], truth: Truth, k: int = 25) -> float:
    for i, smi in enumerate(smiles_list[:k]):
        if same_skeleton(smi, None, truth):
            return 1.0 / float(i + 1)
    return 0.0


def label_window(trace: dict[str, Any], truth: Truth) -> np.ndarray:
    win = trace["window"]
    y = np.zeros(len(win["smiles"]), dtype=np.int64)
    for i, (smi, raw) in enumerate(zip(win["smiles"], win["raw_key"])):
        if abs(float(win["exact_mass"][i]) - truth.mass) > 0.5 and raw != truth.raw_key:
            continue  # cannot be the same skeleton (mass differs)
        if same_skeleton(smi, raw, truth):
            y[i] = 1
    return y


def truth_location(res: dict[str, Any], truth: Truth) -> dict[str, Any]:
    """Where the truth sits in the generated pool for diagnostics."""
    trace = res["trace"]
    locked_keys = trace.get("locked", {}).get("key", [])
    locked_smiles = trace.get("locked", {}).get("smiles", [])
    truth_locked = any(
        k == truth.canonical or same_skeleton(s, None, truth) for s, k in zip(locked_smiles, locked_keys)
    )
    y = label_window(trace, truth)
    in_window = bool(y.any())
    in_tail = False
    tail = trace.get("tail", {})
    for smi, raw, m in zip(tail.get("smiles", []), tail.get("raw_key", []), tail.get("exact_mass", [])):
        if abs(float(m) - truth.mass) <= 0.5 and same_skeleton(smi, raw, truth):
            in_tail = True
            break
    crc = trace.get("pool_key_crc")
    in_pool_raw = bool(crc is not None and np.any(crc == np.uint32(zlib.crc32(truth.raw_key.encode()))))
    return {
        "truth_locked": bool(truth_locked),
        "truth_in_window": in_window,
        "truth_in_tail": in_tail,
        "truth_in_pool_raw_key": in_pool_raw or in_window or in_tail,
        "false_lock": bool(locked_keys) and not truth_locked,
        "n_locked": len(locked_keys),
        "window_labels": y,
    }


def apply_reranker_to_trace(
    trace: dict[str, Any],
    reranker: Any,
    cfg: Config,
    *,
    baseline_final: list[str] | None = None,
) -> list[str]:
    """Rerank arm reconstructed offline from a baseline trace (same pool)."""
    from src.infer import _Hit, _emit_unique
    from src.rerank import rerank_scores, reranked_order

    top_k = int(cfg.top_k)
    win = trace["window"]
    tail = trace.get("tail", {})
    locked_info = trace.get("locked", {"smiles": [], "key": []})
    locked = [
        _Hit(s, k, 1.0, spec=float(sp), n_match=float(nm))
        for s, k, sp, nm in zip(
            locked_info["smiles"], locked_info["key"], locked_info.get("spec", [1.0] * len(locked_info["smiles"])),
            locked_info.get("n_match", [0.0] * len(locked_info["smiles"])),
        )
    ]
    window = [
        _Hit(s, k, float(sc), exact_mass=float(m), source=int(src))
        for s, k, sc, m, src in zip(win["smiles"], win["raw_key"], win["lin_score"], win["exact_mass"], win["source"])
    ]
    tail_hits = [
        _Hit(s, k, float(sc), exact_mass=float(m), source=int(src))
        for s, k, sc, m, src in zip(
            tail.get("smiles", []), tail.get("raw_key", []), tail.get("lin_score", []), tail.get("exact_mass", []), tail.get("source", [])
        )
    ]
    if window:
        scores = rerank_scores(reranker, win["features"])
        order = reranked_order(scores)
        window = [window[int(i)] for i in order]
    remain = max(top_k - len(locked), 0)
    competed = _emit_unique(window + tail_hits, locked, remain, locked_keys=list(locked_info["key"]))
    out = [h.smiles for h in locked] + [h.smiles for h in competed]
    if baseline_final and len(out) < top_k:
        have = set(out)
        for smi in baseline_final:
            if smi not in have:
                out.append(smi)
                have.add(smi)
            if len(out) >= top_k:
                break
    return out[:top_k]


class BaselineOrderScorer:
    """Scorer that reproduces the baseline order (used to recover the other arm from a trace)."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        from src.rerank import FEATURE_NAMES

        X = np.asarray(X, dtype=np.float32)
        return -X[:, FEATURE_NAMES.index("log_linear_rank")]


def baseline_from_trace(trace: dict[str, Any], cfg: Config, *, final: list[str] | None = None) -> list[str]:
    return apply_reranker_to_trace(trace, BaselineOrderScorer(), cfg, baseline_final=final)


def arm_metrics(rr: np.ndarray) -> dict[str, float | int]:
    rr = np.asarray(rr, dtype=np.float64)
    n = int(rr.size)
    if n == 0:
        return {"n": 0, "mrr": 0.0, "recall@1": 0.0, "recall@5": 0.0, "recall@25": 0.0}
    return {
        "n": n,
        "mrr": float(rr.mean()),
        "recall@1": float((rr >= 1.0).mean()),
        "recall@5": float((rr >= 0.2).mean()),
        "recall@25": float((rr > 0.0).mean()),
    }


def paired_bootstrap(delta: np.ndarray, *, n_resamples: int = 4096, seed: int = 0) -> dict[str, float]:
    delta = np.asarray(delta, dtype=np.float64)
    if delta.size == 0:
        return {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "n": 0}
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, delta.size, size=(int(n_resamples), delta.size))
    means = delta[idx].mean(axis=1)
    return {
        "mean": float(delta.mean()),
        "ci_lo": float(np.percentile(means, 2.5)),
        "ci_hi": float(np.percentile(means, 97.5)),
        "n": int(delta.size),
    }


def evaluate_arms(
    queries: list[Query],
    results: dict[str, dict[str, Any]],
    reranker: Any | None,
    cfg: Config,
    *,
    rule: dict[str, Any] = PROMOTION_RULE,
) -> dict[str, Any]:
    """Baseline vs rerank on identical pools, per regime and pooled."""
    per_regime: dict[str, dict[str, list[float]]] = {}
    coverage: dict[str, dict[str, int]] = {}
    rows: list[dict[str, Any]] = []
    for q in queries:
        res = results.get(q.qid)
        if res is None:
            continue
        loc = truth_location(res, q.truth)
        base_rr = reciprocal_rank_smiles(res["final_smiles"], q.truth, k=cfg.top_k)
        if reranker is not None:
            rer_smiles = apply_reranker_to_trace(res["trace"], reranker, cfg, baseline_final=res["final_smiles"])
            rer_rr = reciprocal_rank_smiles(rer_smiles, q.truth, k=cfg.top_k)
        else:
            rer_smiles = list(res["final_smiles"])
            rer_rr = base_rr
        reg = per_regime.setdefault(q.regime, {"base": [], "rerank": []})
        reg["base"].append(base_rr)
        reg["rerank"].append(rer_rr)
        cov = coverage.setdefault(q.regime, {"n": 0, "truth_locked": 0, "truth_in_window": 0, "truth_in_pool": 0, "false_lock": 0, "n_locked_queries": 0})
        cov["n"] += 1
        cov["truth_locked"] += int(loc["truth_locked"])
        cov["truth_in_window"] += int(loc["truth_in_window"])
        cov["truth_in_pool"] += int(loc["truth_in_pool_raw_key"])
        cov["false_lock"] += int(loc["false_lock"])
        cov["n_locked_queries"] += int(loc["n_locked"] > 0)
        rows.append(
            {
                "qid": q.qid,
                "partition": q.partition,
                "regime": q.regime,
                "query_lib": q.meta.get("query_lib", ""),
                "instrument": q.meta.get("instrument", ""),
                "n_locked": loc["n_locked"],
                "truth_locked": loc["truth_locked"],
                "truth_in_window": loc["truth_in_window"],
                "pool_size": int(res["trace"].get("pool_size", 0)),
                "rr_baseline": base_rr,
                "rr_rerank": rer_rr,
                "baseline_top1_source": (res["trace"].get("merged_source") or [0])[0],
                "rerank_smiles_differs": rer_smiles != list(res["final_smiles"]),
            }
        )
    out: dict[str, Any] = {"regimes": {}, "coverage": coverage, "rows": rows}
    all_base: list[float] = []
    all_rer: list[float] = []
    for regime, arms in per_regime.items():
        b = np.asarray(arms["base"])
        r = np.asarray(arms["rerank"])
        all_base.extend(b.tolist())
        all_rer.extend(r.tolist())
        out["regimes"][regime] = {
            "baseline": arm_metrics(b),
            "rerank": arm_metrics(r),
            "paired_delta": paired_bootstrap(r - b, n_resamples=rule["bootstrap_resamples"], seed=rule["bootstrap_seed"]),
        }
        in_window = np.array([row["truth_in_window"] for row in rows if row["regime"] == regime], dtype=bool)
        if in_window.any():
            # Queries whose truth reached the unlocked window: the slice the reranker can act on.
            out["regimes"][regime]["truth_in_window_only"] = {
                "baseline": arm_metrics(b[in_window]),
                "rerank": arm_metrics(r[in_window]),
            }
    b = np.asarray(all_base)
    r = np.asarray(all_rer)
    out["pooled"] = {
        "baseline": arm_metrics(b),
        "rerank": arm_metrics(r),
        "paired_delta": paired_bootstrap(r - b, n_resamples=rule["bootstrap_resamples"], seed=rule["bootstrap_seed"]),
        "n_rankings_changed": int(sum(1 for row in rows if row["rerank_smiles_differs"])),
    }
    return out


def decide_promotion(
    audit: dict[str, Any] | None,
    panel: dict[str, Any] | None,
    *,
    reranker_fitted: bool,
    rule: dict[str, Any] = PROMOTION_RULE,
) -> dict[str, Any]:
    """Apply the pre-declared rule. Any failed or unmeasurable check keeps the baseline."""
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"check": name, "passed": bool(ok), "detail": detail})

    add("reranker_fitted", reranker_fitted, "a reranker was fitted on RR_TRAIN")
    if audit is None or not audit.get("pooled") or audit["pooled"]["baseline"]["n"] == 0:
        add("audit_available", False, "no audit queries were ranked")
    else:
        pooled = audit["pooled"]
        n = int(pooled["baseline"]["n"])
        add("min_audit_queries", n >= int(rule["min_audit_queries"]), f"n={n} >= {rule['min_audit_queries']}")
        delta = pooled["paired_delta"]
        add("min_delta_mrr", delta["mean"] >= float(rule["min_delta_mrr"]), f"delta={delta['mean']:+.4f} >= {rule['min_delta_mrr']:+.3f}")
        add("ci_lower", delta["ci_lo"] > float(rule["ci_lower_must_exceed"]), f"ci=[{delta['ci_lo']:+.4f}, {delta['ci_hi']:+.4f}]")
        known = audit["regimes"].get("K")
        if known is None or known["baseline"]["n"] == 0:
            add("known_regime", False, "no regime-K audit queries")
        else:
            kd = known["paired_delta"]["mean"]
            add("known_regime_delta", kd >= float(rule["min_known_regime_delta"]), f"K delta={kd:+.4f} >= {rule['min_known_regime_delta']:+.3f}")
    if panel is not None and panel.get("pooled") and panel["pooled"]["baseline"]["n"] >= int(rule["panel_min_n"]):
        pd_ = panel["pooled"]["paired_delta"]["mean"]
        add("panel_delta", pd_ >= float(rule["min_panel_delta"]), f"panel delta={pd_:+.4f} (n={panel['pooled']['baseline']['n']})")
    else:
        n_panel = 0 if panel is None or not panel.get("pooled") else panel["pooled"]["baseline"]["n"]
        checks.append({"check": "panel_delta", "passed": True, "detail": f"not gating (n={n_panel} < {rule['panel_min_n']})"})
    passed = all(c["passed"] for c in checks)
    return {
        "selected_arm": "rerank" if passed else "baseline",
        "promoted": bool(passed),
        "checks": checks,
        "rule": dict(rule),
    }


# --------------------------------------------------------------------------
# training rows
# --------------------------------------------------------------------------
def training_rows(
    queries: list[Query],
    results: dict[str, dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Window features + labels for queries whose truth is in the unlocked window."""
    Xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    qids: list[np.ndarray] = []
    counts = {"queries": 0, "used": 0, "truth_locked": 0, "truth_outside_window": 0, "empty_window": 0}
    for q in queries:
        res = results.get(q.qid)
        if res is None:
            continue
        counts["queries"] += 1
        loc = truth_location(res, q.truth)
        if loc["truth_locked"]:
            counts["truth_locked"] += 1
            continue
        y = loc["window_labels"]
        if y.size == 0:
            counts["empty_window"] += 1
            continue
        if not y.any():
            counts["truth_outside_window"] += 1
            continue
        X = np.asarray(res["trace"]["window"]["features"], dtype=np.float32)
        Xs.append(X)
        ys.append(y)
        qids.append(np.full(y.size, counts["used"], dtype=np.int64))
        counts["used"] += 1
    if not Xs:
        from src.rerank import FEATURE_NAMES

        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32), np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), counts
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(qids), counts


# --------------------------------------------------------------------------
# artifacts
# --------------------------------------------------------------------------
def write_json(path: Path | str, obj: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default))
    return path


def _json_default(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, (set, tuple)):
        return list(o)
    if hasattr(o, "__dataclass_fields__"):
        return asdict(o)
    return str(o)


def trace_table(queries: list[Query], results: dict[str, dict[str, Any]], reranker: Any | None, cfg: Config, *, top_n: int = 50):
    """Compact per-candidate table (window top-N) for the audit/panel partitions."""
    import pandas as pd

    from src.rerank import rerank_scores

    recs: list[dict[str, Any]] = []
    for q in queries:
        res = results.get(q.qid)
        if res is None:
            continue
        win = res["trace"]["window"]
        n = min(int(top_n), len(win["smiles"]))
        if n == 0:
            continue
        y = label_window(res["trace"], q.truth)
        scores = rerank_scores(reranker, win["features"]) if reranker is not None else np.full(len(win["smiles"]), np.nan)
        order = np.argsort(-np.nan_to_num(scores, nan=-1.0), kind="stable") if reranker is not None else np.arange(len(win["smiles"]))
        rerank_rank = np.empty(len(order), dtype=np.int64)
        rerank_rank[order] = np.arange(len(order))
        for i in range(n):
            recs.append(
                {
                    "qid": q.qid,
                    "partition": q.partition,
                    "regime": q.regime,
                    "smiles": win["smiles"][i],
                    "source": int(win["source"][i]),
                    "baseline_rank": i,
                    "rerank_rank": int(rerank_rank[i]),
                    "linear_score": float(win["lin_score"][i]),
                    "rerank_score": float(scores[i]) if reranker is not None else None,
                    "spec": float(win["spec"][i]),
                    "n_match": float(win["n_match"][i]),
                    "tanimoto": float(win["tani"][i]),
                    "is_truth": bool(y[i]),
                }
            )
    return pd.DataFrame(recs)


def zip_directory(src_dir: Path | str, zip_path: Path | str, *, exclude: Iterable[str] = ()) -> Path:
    src_dir = Path(src_dir)
    zip_path = Path(zip_path)
    skip = set(exclude)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(src_dir.rglob("*")):
            if p.is_file() and p.name not in skip and p.resolve() != zip_path.resolve():
                zf.write(p, arcname=str(p.relative_to(src_dir)))
    return zip_path


def strict_submission_check(path: Path | str, sample_path: Path | str | None, *, top_k: int = 25) -> dict[str, Any]:
    """Format validation plus RDKit parseability and per-row skeleton uniqueness."""
    import pandas as pd

    from src.chem import smiles_to_mol
    from src.ranker import validate_submission

    validate_submission(path, sample_path)
    df = pd.read_csv(path, dtype=str)
    n_guesses: list[int] = []
    n_unparseable = 0
    n_dup_rows = 0
    for raw in df["smiles"].astype(str).tolist():
        parts = [p for p in raw.split(";") if p]
        n_guesses.append(len(parts))
        by_graph: dict[str, list[str]] = {}
        for p in parts:
            if smiles_to_mol(p) is None:
                n_unparseable += 1
                continue
            by_graph.setdefault(heavy_atom_graph_key(p), []).append(p)
        duplicate = False
        for group in by_graph.values():
            if len(group) < 2:
                continue
            keys = [inchikey14_from_smiles(p) for p in group]
            if len(set(keys)) != len(keys):
                duplicate = True
                break
        n_dup_rows += int(duplicate)
    report = {
        "rows": int(len(df)),
        "min_guesses": int(min(n_guesses)) if n_guesses else 0,
        "max_guesses": int(max(n_guesses)) if n_guesses else 0,
        "rows_with_full_top_k": int(sum(1 for n in n_guesses if n == int(top_k))),
        "unparseable_smiles": int(n_unparseable),
        "rows_with_duplicate_skeleton": int(n_dup_rows),
    }
    if n_unparseable:
        raise ValueError(f"submission has {n_unparseable} unparseable SMILES")
    if n_dup_rows:
        raise ValueError(f"submission has {n_dup_rows} rows with duplicate skeletons")
    return report
