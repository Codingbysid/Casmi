"""Required-reranker execution path for a Kaggle Run All.

Train-and-predict:

    preflight → library → three Spec2FP checkpoints → RR_TRAIN pairs
    → fit the reranker → predict test → atomically publish submission.csv

Predict-only loads a verified bundle and never trains. There is no local
promotion rule and no fallback to the linear score.

The precursor-m/z shift (``precursor-mz-shift-v1``) and the official metric
identity (``official-inchikey14-v14``) are recorded here and left off.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.config import Config, get_config
from src.rerank import FEATURE_NAMES, MIN_TRAIN_QUERIES, RERANKER_PARAMS, fit_reranker, load_reranker, save_reranker

STRATEGY_ID = "required-reranker-v1"
PRECURSOR_EXPERIMENT_ID = "precursor-mz-shift-v1"
IDENTITY_EXPERIMENT_ID = "official-inchikey14-v14"
INACTIVE_EXPERIMENTS = (PRECURSOR_EXPERIMENT_ID, IDENTITY_EXPERIMENT_ID)

PRODUCTION_SEEDS = (42, 7, 123)
EXPECTED_CLASS2_ROWS = 452_608
LIBRARY_SCHEMA = "casmi-spectral-library-v1"
NEIGHBOR_SCHEMA = "casmi-neighbor-bins-v1"
PAIRS_SCHEMA = "casmi-reranker-pairs-v1"
QUERY_SHARD_SCHEMA = "casmi-reranker-query-v1"
MANIFEST_SCHEMA = "casmi-run-manifest-v1"

CHECKPOINT_SOURCE = (
    "src/chem.py",
    "src/config.py",
    "src/data.py",
    "src/model.py",
    "src/preprocessing.py",
    "src/tpu_trainer.py",
)
PAIRS_SOURCE = (
    "src/analogs.py",
    "src/chem.py",
    "src/infer.py",
    "src/neighbors.py",
    "src/ranker.py",
    "src/rerank.py",
    "src/retrieval.py",
    "src/validation.py",
)

RECIPE = {
    "num_epochs": 4,
    "max_train_spectra": 120_000,
    "batch_size": 64,
    "d_model": 128,
    "n_heads": 4,
    "n_transformer_layers": 2,
    "n_conv_channels": 64,
    "top_n_peaks": 128,
    "n_mz_bins": 512,
    "fp_bits": 2048,
    "neighbor_k": 20,
    "neighbor_blend": 0.5,
    "use_neighbor_fp": True,
    "rerank_window": 200,
    "use_representative_metadata": False,
    "restore_best_checkpoint": False,
    "use_observed_precursor": False,
    "ensemble_seeds": 3,
    "max_mass_candidates": 8192,
}


class PipelineError(RuntimeError):
    """A required step failed. The run must not substitute another strategy."""


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def bound_threads(n: int = 4) -> int:
    """Keep numeric libraries from spawning one thread per core."""
    limit = max(1, min(int(n), os.cpu_count() or 1))
    value = str(limit)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = value
    try:
        import torch

        torch.set_num_threads(limit)
    except Exception:
        pass
    print(f"[threads] numeric libraries limited to {limit}")
    return limit


def rss_mb() -> float:
    try:
        import resource

        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return 0.0
    if sys.platform == "darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def atomic_write_text(path: Path | str, text: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(text)
    os.replace(tmp, path)
    return path


def atomic_write_json(path: Path | str, obj: Any) -> Path:
    return atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=True, default=_json_default))


def atomic_savez(path: Path | str, **arrays: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".partial.npz")
    if tmp.exists():
        tmp.unlink()
    np.savez(tmp, **arrays)
    os.replace(tmp, path)
    return path


def _json_default(value: Any):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def file_sha256(path: Path | str) -> str:
    from src.validation import sha256_file

    return sha256_file(path, full=True)


def content_id(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    return {"bytes": int(path.stat().st_size), "sha256": file_sha256(path)}


def source_hash(files: tuple[str, ...] = CHECKPOINT_SOURCE) -> str:
    digest = hashlib.sha256()
    root = repo_root()
    for rel in files:
        digest.update(rel.encode())
        digest.update((root / rel).read_bytes())
    return digest.hexdigest()[:16]


def dependency_versions() -> dict[str, str]:
    import pandas
    import torch
    from rdkit import Chem

    import sklearn

    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pandas.__version__,
        "torch": torch.__version__,
        "sklearn": sklearn.__version__,
        "rdkit": str(Chem.rdBase.rdkitVersion),
    }


def production_config(**overrides: Any) -> Config:
    recipe = dict(RECIPE)
    recipe.update(overrides)
    cfg = get_config(
        **recipe,
        train_time_limit_s=1.5 * 3600.0,
        log_every=50,
        num_workers=0,
        smiles_num_epochs=1,
        smiles_batch_size=64,
    )
    assert_production_recipe(cfg)
    return cfg


def assert_production_recipe(cfg: Config) -> None:
    for key, expected in RECIPE.items():
        got = getattr(cfg, key)
        if isinstance(expected, float):
            ok = abs(float(got) - expected) < 1e-9
        else:
            ok = got == expected
        if not ok:
            raise PipelineError(
                f"production recipe forbids {key}={got!r}; required-reranker-v1 expects {expected!r}"
            )
    if tuple(PRODUCTION_SEEDS) != (42, 7, 123):
        raise PipelineError("production seeds drifted")


def find_competition_dir() -> Path:
    candidates = [
        Path("/kaggle/input/enveda-casmi26-molecule-id-mass-spectra"),
        Path("/kaggle/input/enveda-CASMI26-molecule-id-mass-spectra"),
        repo_root() / "data",
        Path("data"),
    ]
    input_root = Path("/kaggle/input")
    if input_root.exists():
        for parquet in input_root.rglob("train.parquet"):
            if (parquet.parent / "test.parquet").exists():
                candidates.insert(0, parquet.parent)
    for cand in candidates:
        if cand.exists() and (cand / "train.parquet").exists() and (cand / "test.parquet").exists():
            return cand
    raise PipelineError("Could not find train.parquet and test.parquet. Attach the competition dataset.")


def find_class2_table() -> Path | None:
    names = {"class2_candidates.parquet", "class2_candidates.csv.gz", "class2_candidates.csv"}
    roots = [Path("/kaggle/input"), repo_root() / "data", Path("data"), repo_root() / "kaggle_dataset"]
    hits: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and (path.name.lower() in names or ("class2" in path.name.lower() and path.suffix == ".parquet")):
                hits.append(path)
    hits.sort(key=lambda path: (0 if path.suffix == ".parquet" else 1, -path.stat().st_size))
    return hits[0] if hits else None


def prepare_kaggle_config(*, require_class2: bool = True) -> Config:
    data_dir = find_competition_dir()
    class2 = find_class2_table()
    if class2 is None and require_class2:
        raise PipelineError(
            "class2_candidates.parquet is not attached. "
            "The required-reranker run needs the 452,608-row COCONUT ∪ LOTUS pool."
        )
    cfg = production_config()
    cfg.data_dir = data_dir
    cfg.external_candidates_path = class2
    print("[config] data", data_dir)
    print("[config] class2", class2)
    return cfg


def _require_class2_pool(path: Path) -> int:
    import pyarrow.parquet as pq

    rows = int(pq.ParquetFile(str(path)).metadata.num_rows)
    if rows != EXPECTED_CLASS2_ROWS:
        raise PipelineError(
            f"Class 2 pool has {rows} rows; required-reranker-v1 needs the full "
            f"{EXPECTED_CLASS2_ROWS}-row COCONUT ∪ LOTUS table. Refusing a filtered library."
        )
    print(f"[class2] {path} rows={rows}")
    return rows


def retire_previous_submission(path: Path | str) -> str | None:
    """Move an existing submission aside so a failed run cannot look fresh."""
    path = Path(path)
    if not path.exists():
        return None
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    dest = path.with_name(f"submission_previous_{stamp}.csv")
    if dest.exists():
        dest = path.with_name(f"submission_previous_{stamp}_{os.getpid()}.csv")
    os.replace(path, dest)
    print(f"[submit] moved previous submission aside to {dest.name}")
    return str(dest)


def _row_order_hash(keys: np.ndarray) -> str:
    digest = hashlib.sha256()
    for key in keys.tolist():
        digest.update(str(key).encode())
        digest.update(b"\n")
    return digest.hexdigest()[:16]


def _array_shapes(arrays: dict[str, np.ndarray]) -> dict[str, list[int]]:
    return {name: list(np.asarray(value).shape) for name, value in arrays.items()}


def save_array_cache(path: Path | str, arrays: dict[str, np.ndarray], sidecar: dict[str, Any]) -> None:
    path = Path(path)
    sidecar = dict(sidecar)
    sidecar["shapes"] = _array_shapes(arrays)
    atomic_savez(path, **arrays)
    atomic_write_json(path.with_suffix(".json"), sidecar)


def load_array_cache(path: Path | str, *, expect_schema: str) -> tuple[Any, dict[str, Any]]:
    path = Path(path)
    side_path = path.with_suffix(".json")
    if not path.exists() or not side_path.exists():
        raise PipelineError(f"cache missing: {path}")
    sidecar = json.loads(side_path.read_text())
    if sidecar.get("schema") != expect_schema:
        raise PipelineError(f"cache schema {sidecar.get('schema')!r} != {expect_schema}")
    data = np.load(path, allow_pickle=True)
    for name, shape in sidecar.get("shapes", {}).items():
        if name not in data.files or list(data[name].shape) != list(shape):
            raise PipelineError(f"cache shape mismatch for {name}")
    row_hash = sidecar.get("row_order_sha256")
    if row_hash is not None and "inchikey14" in data.files:
        got = _row_order_hash(data["inchikey14"])
        if got != row_hash:
            raise PipelineError("cache row order does not match its sidecar hash")
    return data, sidecar


def library_arrays(index) -> dict[str, np.ndarray]:
    arrays = {
        "smiles": np.asarray(index.smiles, dtype=object),
        "inchikey14": np.asarray(index.inchikey14, dtype=object),
        "exact_mass": np.asarray(index.exact_mass),
        "fingerprints": np.asarray(index.fingerprints),
        "peak_mz": np.asarray(index.peak_mz),
        "peak_intensity": np.asarray(index.peak_intensity),
        "peak_mask": np.asarray(index.peak_mask),
        "peak_nl": np.asarray(index.peak_nl),
        "precursor_feat": np.asarray(index.precursor_feat),
        "adduct_id": np.asarray(index.adduct_id),
        "precursor_mz": np.asarray(index.precursor_mz if index.precursor_mz is not None else np.full(len(index.smiles), np.nan)),
    }
    if index.formulas is not None:
        arrays["formulas"] = np.asarray(index.formulas, dtype=object)
    if index.ionization_mode is not None:
        arrays["ionization_mode"] = np.asarray(index.ionization_mode, dtype=object)
    return arrays


def library_from_cache(data) -> Any:
    from src.retrieval import StructureIndex

    def _opt(name):
        return data[name] if name in data.files else None

    return StructureIndex.build(
        smiles=data["smiles"],
        inchikey14=data["inchikey14"],
        exact_mass=data["exact_mass"],
        fingerprints=data["fingerprints"],
        peak_mz=data["peak_mz"],
        peak_intensity=data["peak_intensity"],
        peak_mask=data["peak_mask"],
        formulas=_opt("formulas"),
        ionization_mode=_opt("ionization_mode"),
        peak_nl=data["peak_nl"],
        precursor_feat=data["precursor_feat"],
        adduct_id=data["adduct_id"],
        precursor_mz=data["precursor_mz"],
    )


def neighbor_from_cache(data):
    from src.neighbors import SpectralNeighborIndex

    ion = data["ion_code"] if "ion_code" in data.files else None
    return SpectralNeighborIndex(
        binned=data["binned"],
        fingerprints=data["fingerprints"],
        ion_code=ion,
        chunk=int(data["chunk"][0]) if "chunk" in data.files else 4096,
    )


def slice_neighbor(full, keep: np.ndarray):
    from src.neighbors import SpectralNeighborIndex

    return SpectralNeighborIndex(
        binned=full.binned[keep],
        fingerprints=full.fingerprints[keep],
        ion_code=None if full.ion_code is None else full.ion_code[keep],
        chunk=full.chunk,
    )


def training_signature(
    *,
    fit_hash: str,
    library_row_hash: str,
    checkpoint_sha256: dict[str, str],
    class2_sha256: str,
    cfg: Config,
) -> str:
    payload = {
        "schema": PAIRS_SCHEMA,
        "fit_hash": fit_hash,
        "library_row_hash": library_row_hash,
        "checkpoint_sha256": {str(k): v for k, v in sorted(checkpoint_sha256.items())},
        "class2_sha256": class2_sha256,
        "feature_names": list(FEATURE_NAMES),
        "params": RERANKER_PARAMS,
        "window": int(cfg.rerank_window),
        "n_rr_train": 1200,
        "validation_seed": 0,
        "pairs_source_hash": source_hash(PAIRS_SOURCE),
        "model_config_hash": __import__("src.validation", fromlist=["model_config_hash"]).model_config_hash(cfg),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _query_identity(query) -> dict[str, Any]:
    truth = query.truth
    return {
        "qid": str(query.qid),
        "partition": str(query.partition),
        "regime": str(query.regime),
        "row": int(query.row),
        "truth_raw_key": str(truth.raw_key),
        "truth_canonical": str(truth.canonical),
        "truth_smiles": str(truth.smiles),
    }


def _query_shard_path(run_dir: Path, signature: str, qid: str) -> Path:
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(qid))
    return Path(run_dir) / "pair_shards" / signature / f"{safe}.npz"


def _load_query_shard(path: Path, identity: dict[str, Any], signature: str):
    """Return ``(X, y)`` only when the shard matches this signature and identity."""
    side = path.with_suffix(".json")
    if not path.exists() or not side.exists():
        return None
    try:
        meta = json.loads(side.read_text())
        if meta.get("schema") != QUERY_SHARD_SCHEMA or meta.get("signature") != signature:
            return None
        if meta.get("identity") != identity:
            return None
        data = np.load(path)
        X = np.asarray(data["X"], dtype=np.float32)
        y = np.asarray(data["y"], dtype=np.int64).ravel()
        if X.ndim != 2 or X.shape[1] != len(FEATURE_NAMES) or y.shape[0] != X.shape[0]:
            raise PipelineError(f"query shard {path.name} has shape {getattr(X, 'shape', None)}")
        if X.size and not np.isfinite(X).all():
            raise PipelineError(f"query shard {path.name} has non-finite features")
    except Exception as exc:
        print(f"[pairs] shard {path.name} rejected: {exc}")
        return None
    return X, y


def _save_query_shard(path: Path, identity: dict[str, Any], signature: str, X: np.ndarray, y: np.ndarray) -> None:
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64).ravel()
    if X.ndim != 2 or X.shape[1] != len(FEATURE_NAMES) or y.shape[0] != X.shape[0]:
        raise PipelineError(f"refusing to store query shard with shape {X.shape} labels {y.shape}")
    atomic_savez(path, X=X, y=y)
    atomic_write_json(path.with_suffix(".json"), {
        "schema": QUERY_SHARD_SCHEMA,
        "signature": signature,
        "identity": identity,
        "n_rows": int(X.shape[0]),
    })


def _stack_query_rows(queries, saved: dict[str, tuple[np.ndarray, np.ndarray]]):
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    qids: list[np.ndarray] = []
    used = 0
    for query in queries:
        X, y = saved[query.qid]
        if X.shape[0] == 0:
            continue
        xs.append(X)
        ys.append(y)
        qids.append(np.full(X.shape[0], used, dtype=np.int64))
        used += 1
    if not xs:
        empty_x = np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32)
        empty = np.zeros((0,), dtype=np.int64)
        return empty_x, empty, empty
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(qids)


def assemble_saved_query_rows(run_dir: Path | str, signature: str, queries, produce):
    """Load matching shards and call ``produce`` only for the queries still missing.

    ``produce(query)`` returns that query's feature matrix and labels. A raise
    leaves earlier shards in place. A different signature is a different
    directory, so those shards are not reused.
    """
    run_dir = Path(run_dir)
    saved: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    n_resumed = 0
    for query in queries:
        identity = _query_identity(query)
        path = _query_shard_path(run_dir, signature, query.qid)
        loaded = _load_query_shard(path, identity, signature)
        if loaded is not None:
            saved[query.qid] = loaded
            n_resumed += 1
            continue
        X, y = produce(query)
        _save_query_shard(path, identity, signature, X, y)
        saved[query.qid] = (np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64).ravel())
        print(f"[pairs] saved {query.qid} rows={saved[query.qid][0].shape[0]} signature={signature}")
    print(f"[pairs] resumed {n_resumed}/{len(queries)} signature={signature}")
    return _stack_query_rows(queries, saved)


def _matrix_for_query(query, result) -> tuple[np.ndarray, np.ndarray]:
    from src.validation import training_rows

    X, y, _qid, _counts = training_rows([query], {query.qid: result})
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.int64).ravel()


def reranker_provenance_problems(
    manifest: dict[str, Any],
    reranker_path: Path,
    class2_sha256: str | None,
) -> list[str]:
    """Compare the attached reranker, pair source, and Class 2 file with the manifest."""
    problems: list[str] = []
    if manifest.get("pairs_source_hash") != source_hash(PAIRS_SOURCE):
        problems.append("pairs source hash does not match the manifest")
    recorded_class2 = manifest.get("class2_sha256")
    if not class2_sha256 or not recorded_class2 or class2_sha256 != recorded_class2:
        problems.append("class 2 content hash does not match the manifest")
    expected = manifest.get("pairs_signature")
    recorded = (manifest.get("reranker") or {}).get("signature")
    recorded_hash = (manifest.get("reranker") or {}).get("sha256")
    meta_path = reranker_path.with_suffix(".json")
    file_sig = None
    if meta_path.exists():
        try:
            file_sig = json.loads(meta_path.read_text()).get("signature")
        except Exception:
            file_sig = None
    if not expected or recorded != expected or file_sig != expected:
        problems.append("reranker signature does not match the manifest")
    if not recorded_hash or not reranker_path.exists() or file_sha256(reranker_path) != recorded_hash:
        problems.append("reranker artifact hash does not match the manifest")
    return problems


def fit_required_reranker(X: np.ndarray, y: np.ndarray, query_id: np.ndarray, *, min_queries: int = MIN_TRAIN_QUERIES):
    reranker, report = fit_reranker(X, y, query_id, min_queries=min_queries)
    if reranker is None or report.get("status") != "fitted":
        raise PipelineError(f"required reranker fit failed: {report.get('status')}. Not using the linear score.")
    print("[reranker] fitted", {k: report[k] for k in ("status", "n_rows", "n_pos", "n_queries", "deterministic", "fit_seconds") if k in report})
    return reranker, report


def _write_manifest(run_dir: Path, state: dict[str, Any]) -> None:
    payload = {key: value for key, value in state.items() if key != "fit_take"}
    atomic_write_json(run_dir / "run_manifest.json", payload)


def _mark(state: dict[str, Any], run_dir: Path, stage: str, t0: float) -> None:
    state["stage"] = stage
    state["timings"][stage] = {"elapsed_s": round(time.time() - t0, 1), "peak_rss_mb": round(rss_mb(), 1)}
    _write_manifest(run_dir, state)
    info = state["timings"][stage]
    print(f"[time] {stage} elapsed_s={info['elapsed_s']} peak_rss_mb={info['peak_rss_mb']}")


def _checkpoint_dirs(run_dir: Path) -> list[Path]:
    dirs = [run_dir / "checkpoints"]
    if Path("/kaggle/input").exists():
        dirs.extend(path.parent for path in Path("/kaggle/input").rglob("spec2fp_s42.pt"))
    return dirs


def _load_or_train_seeds(cfg: Config, ds, run_dir: Path, state: dict[str, Any], t0: float) -> list:
    from src.tpu_trainer import get_device, load_model, train_spec2fp
    from src.validation import checkpoint_extra, checkpoint_is_complete

    device, _ = get_device(prefer_xla=False)
    if device.type != "cuda":
        raise PipelineError(f"required-reranker-v1 needs a CUDA GPU; got {device}")
    models = []
    fit_hash = state["fit_subset_hash"]
    config_hash = state["model_config_hash"]
    src_hash = state["checkpoint_source_hash"]
    versions = state["versions"]
    for seed in PRODUCTION_SEEDS:
        name = f"spec2fp_s{seed}.pt"
        dest = run_dir / "checkpoints" / name
        reused = None
        for folder in _checkpoint_dirs(run_dir):
            cand = folder / name
            if not cand.exists():
                continue
            extra = checkpoint_extra(cand)
            ok, why = checkpoint_is_complete(
                extra, cfg, fit_hash=fit_hash, config_hash=config_hash, seed=seed,
                source_hash=src_hash, versions=versions,
            )
            print(f"[ckpt] seed={seed} {cand} {'ok' if ok else why}")
            if ok:
                reused = cand
                break
        cfg.seed = int(seed)
        cfg.ckpt_name = name
        cfg.artifact_dir = run_dir / "checkpoints"
        if reused is None:
            print(f"[ckpt] seed={seed} training epochs={cfg.num_epochs}")
            model = train_spec2fp(
                ds, cfg, device=device, seed=int(seed), ckpt_name=name, time_limit_s=cfg.train_time_limit_s,
                extra_meta={
                    "fit_subset_hash": fit_hash,
                    "model_config_hash": config_hash,
                    "source_hash": src_hash,
                    "versions": versions,
                    "strategy_id": STRATEGY_ID,
                    "library_n": state["library_n"],
                },
            )
            extra = checkpoint_extra(cfg.checkpoint_path)
            ok, why = checkpoint_is_complete(
                extra, cfg, fit_hash=fit_hash, config_hash=config_hash, seed=seed,
                source_hash=src_hash, versions=versions,
            )
            if not ok:
                raise PipelineError(f"seed={seed} checkpoint is incomplete ({why})")
        else:
            model = load_model(reused, cfg)
            if reused.resolve() != dest.resolve():
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_name(dest.name + ".partial")
                tmp.write_bytes(reused.read_bytes())
                os.replace(tmp, dest)
        models.append(model)
        state["checkpoints"][str(seed)] = {"path": str(dest if dest.exists() else reused), "sha256": file_sha256(dest if dest.exists() else reused), "complete": True}
        _mark(state, run_dir, f"seed_{seed}", t0)
    if len(models) != 3:
        raise PipelineError(f"expected 3 checkpoints, got {len(models)}")
    return models


def _rr_train_pairs(cfg: Config, index, external, neighbor, models, state: dict[str, Any], run_dir: Path, t0: float):
    from src.infer import predict_fingerprint_ensemble
    from src.neighbors import blend_predicted_fingerprints
    from src.validation import (
        build_queries,
        check_plan_disjoint,
        collect_acquisitions,
        external_with_truths,
        masked_library,
        plan_holdouts,
        rank_queries,
    )

    fit_take = state["fit_take"]
    signature = state["pairs_signature"]
    cache = run_dir / "reranker_pairs.npz"
    side = cache.with_suffix(".json")
    if cache.exists() and side.exists():
        meta = json.loads(side.read_text())
        if meta.get("schema") == PAIRS_SCHEMA and meta.get("signature") == signature:
            data = np.load(cache)
            print(f"[pairs] reusing {cache.name} signature={signature}")
            _mark(state, run_dir, "reranker_pairs_cache", t0)
            return data["X"], data["y"], data["qid"]
        print("[pairs] cached pairs signature does not match; regenerating")
    plan = plan_holdouts(
        index, fit_take, sizes={"RR_TRAIN": 1200, "DEV": 0, "AUDIT": 0, "PANEL": 0},
        panel_rows=[], seed=0,
    )
    check_plan_disjoint(plan, fit_take)
    if plan.partitions["DEV"] or plan.partitions["AUDIT"] or plan.partitions["PANEL"]:
        raise PipelineError("only the RR_TRAIN partition may be built")
    wanted = {str(index.inchikey14[r]) for r in plan.partitions["RR_TRAIN"]}
    acq = collect_acquisitions(cfg, wanted, panel_lib=None, panel_raw_keys=set())
    queries, skipped = build_queries(plan, acq, index, cfg)
    del acq
    queries = [q for q in queries if q.partition == "RR_TRAIN"]
    if not queries:
        raise PipelineError(f"RR_TRAIN produced no queries (skipped={skipped})")
    atomic_write_json(run_dir / "splits.json", {"partition": "RR_TRAIN", "n": len(queries), "skipped": skipped, "signature": signature})
    u_queries = [q for q in queries if q.regime == "U"]
    k_queries = [q for q in queries if q.regime == "K"]
    masked = masked_library(index, plan.masked_rows())
    keep = np.ones(len(index.smiles), dtype=bool)
    keep[np.fromiter(plan.masked_rows(), dtype=np.int64)] = False
    ext_val, _inserted = external_with_truths(external, [q.truth for q in u_queries], cfg)
    pending = []
    for query in queries:
        if _load_query_shard(_query_shard_path(run_dir, signature, query.qid), _query_identity(query), signature) is None:
            pending.append(query)
    pending_ids = {query.qid for query in pending}
    pred = {}
    for label, qs, nbr in (
        ("U", [q for q in u_queries if q.qid in pending_ids], slice_neighbor(neighbor, keep)),
        ("K", [q for q in k_queries if q.qid in pending_ids], neighbor),
    ):
        if not qs:
            continue
        fps = predict_fingerprint_ensemble(models, [q.feat for q in qs], cfg, allow_uniform=False)
        fps = blend_predicted_fingerprints(fps, [q.feat for q in qs], nbr, k=int(cfg.neighbor_k), alpha=float(cfg.neighbor_blend))
        if fps.shape[0] != len(qs) or not np.isfinite(fps).all():
            raise PipelineError(f"neighbor blend failed for regime {label}")
        for query, fp in zip(qs, fps):
            pred[query.qid] = fp
    print(f"[pairs] ranking {len(pending_ids)} remaining queries in one process")
    ranked: dict[str, dict[str, Any]] = {}

    def produce(query):
        if query.qid in ranked:
            return _matrix_for_query(query, ranked[query.qid])
        group = u_queries if query.regime == "U" else k_queries
        pool = masked if query.regime == "U" else index
        ext = ext_val if query.regime == "U" else external
        if query.qid not in {item.qid for item in group}:
            raise PipelineError(f"{query.qid} is not an RR_TRAIN U or K query")
        out = rank_queries([query], pred, pool, ext, cfg, workers=1, progress_every=1, label=query.regime)
        if query.qid not in out:
            raise PipelineError(f"candidate generation missed {query.qid}")
        ranked[query.qid] = out[query.qid]
        return _matrix_for_query(query, out[query.qid])

    X, y, qid = assemble_saved_query_rows(run_dir, signature, queries, produce)
    counts = {"rows": int(X.shape[0]), "queries_with_rows": int(np.unique(qid).size) if qid.size else 0}
    print("[pairs] training rows", X.shape, "counts", counts)
    atomic_savez(cache, X=X, y=y, qid=qid)
    atomic_write_json(side, {"schema": PAIRS_SCHEMA, "signature": signature, "counts": counts})
    _mark(state, run_dir, "reranker_pairs", t0)
    return X, y, qid


def _predict_and_publish(cfg: Config, index, external, neighbor, models, reranker, run_dir: Path, state: dict[str, Any], t0: float, submission_path: Path) -> dict[str, Any]:
    from src.infer import predict_test
    from src.validation import strict_submission_check

    if reranker is None:
        raise PipelineError("required reranker is missing")
    print(
        f"STRATEGY_ACTIVE {STRATEGY_ID} reranker=required window={int(cfg.rerank_window)} "
        f"seeds={list(PRODUCTION_SEEDS)} inactive={list(INACTIVE_EXPERIMENTS)}"
    )
    partial = submission_path.with_name("submission.csv.partial")
    if partial.exists():
        partial.unlink()
    predict_test(
        cfg, index=index, models=models, external_index=external, output_path=partial, reranker=reranker,
        allow_decoder=False, strict=True, require_reranker=True, require_neighbors=True,
        allow_uniform=False, neighbor_index=neighbor,
    )
    report = strict_submission_check(partial, cfg.sample_submission_path, top_k=int(cfg.top_k))
    os.replace(partial, submission_path)
    digest = file_sha256(submission_path)
    print(f"SUBMISSION_READY {submission_path} sha256={digest} rows={report['rows']}")
    summary = {
        "strategy_id": STRATEGY_ID,
        "inactive_experiments": list(INACTIVE_EXPERIMENTS),
        "source_hash": state.get("checkpoint_source_hash"),
        "pairs_source_hash": state.get("pairs_source_hash"),
        "checkpoint_sha256": {seed: info.get("sha256") for seed, info in state.get("checkpoints", {}).items()},
        "complete_seeds": [int(seed) for seed in state.get("checkpoints", {})],
        "reranker_status": state.get("reranker", {}).get("status"),
        "submission_sha256": digest,
        "rows": report["rows"],
        "timings": state.get("timings", {}),
        "use_observed_precursor": False,
    }
    atomic_write_json(run_dir / "summary.json", summary)
    _mark(state, run_dir, "submission", t0)
    summary["timings"] = state["timings"]
    atomic_write_json(run_dir / "summary.json", summary)
    print("SUMMARY_JSON " + json.dumps(summary, default=str))
    return summary


def run_train_and_predict(
    cfg: Config,
    *,
    run_dir: Path | str,
    submission_path: Path | str | None = None,
    source_commit: str = "",
    require_cuda: bool = True,
) -> dict[str, Any]:
    """Full required-reranker run. A failed fit or a missing seed raises."""
    del require_cuda  # CUDA is checked when the seeds load; a CPU device raises
    from src.data import dataset_from_structure_index, fitting_subset_indices
    from src.infer import build_library_from_train
    from src.neighbors import SpectralNeighborIndex
    from src.retrieval import load_external_structure_index
    from src.validation import fit_subset_hash, model_config_hash

    if bool(getattr(cfg, "use_observed_precursor", False)):
        raise PipelineError(f"{PRECURSOR_EXPERIMENT_ID} is not part of {STRATEGY_ID}")
    assert_production_recipe(cfg)
    bound_threads(4)
    run_dir = Path(run_dir)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    cfg.artifact_dir = run_dir / "checkpoints"
    submission_path = Path(submission_path or (Path("/kaggle/working") if Path("/kaggle/working").exists() else run_dir.parent) / "submission.csv")
    retire_previous_submission(submission_path)
    t0 = time.time()
    versions = dependency_versions()
    state: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA,
        "strategy_id": STRATEGY_ID,
        "inactive_experiments": list(INACTIVE_EXPERIMENTS),
        "source_commit": source_commit,
        "checkpoint_source_hash": source_hash(CHECKPOINT_SOURCE),
        "pairs_source_hash": source_hash(PAIRS_SOURCE),
        "versions": versions,
        "checkpoints": {},
        "reranker": {"status": "missing"},
        "timings": {},
        "fast_dev_run": False,
    }
    _mark(state, run_dir, "preflight", t0)
    if cfg.external_candidates_path is None or not Path(cfg.external_candidates_path).exists():
        raise PipelineError("Class 2 library path is missing")
    _require_class2_pool(Path(cfg.external_candidates_path))
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: F401
    except Exception as exc:
        raise PipelineError(f"sklearn HistGradientBoostingClassifier is required: {exc}") from exc

    lib_path = run_dir / "library.npz"
    train_id = content_id(cfg.train_path)
    index = None
    if lib_path.exists():
        try:
            data, side = load_array_cache(lib_path, expect_schema=LIBRARY_SCHEMA)
            if side.get("train") == train_id and side.get("checkpoint_source_hash") == state["checkpoint_source_hash"]:
                index = library_from_cache(data)
                print(f"[library] loaded cache n={len(index.smiles)}")
        except PipelineError as exc:
            print(f"[library] cache rejected: {exc}")
    if index is None:
        index = build_library_from_train(cfg, progress=True)
        arrays = library_arrays(index)
        save_array_cache(lib_path, arrays, {
            "schema": LIBRARY_SCHEMA,
            "row_order_sha256": _row_order_hash(arrays["inchikey14"]),
            "train": train_id,
            "checkpoint_source_hash": state["checkpoint_source_hash"],
        })
    row_hash = _row_order_hash(np.asarray(index.inchikey14))
    state["library_n"] = int(len(index.smiles))
    state["library_row_hash"] = row_hash
    fit_take = fitting_subset_indices(len(index.smiles), int(cfg.max_train_spectra), seed=42)
    state["fit_take"] = fit_take
    state["fit_subset_hash"] = fit_subset_hash(index, fit_take)
    state["model_config_hash"] = model_config_hash(cfg)
    _mark(state, run_dir, "library", t0)

    nbr_path = run_dir / "neighbor.npz"
    neighbor = None
    if nbr_path.exists():
        try:
            data, side = load_array_cache(nbr_path, expect_schema=NEIGHBOR_SCHEMA)
            if side.get("row_order_sha256") == row_hash and int(side.get("n_mz_bins", -1)) == int(cfg.n_mz_bins):
                neighbor = neighbor_from_cache(data)
                print(f"[neighbor] loaded cache n={neighbor.binned.shape[0]}")
        except PipelineError as exc:
            print(f"[neighbor] cache rejected: {exc}")
    if neighbor is None:
        neighbor = SpectralNeighborIndex.from_structure_index(index, cfg)
        save_array_cache(nbr_path, {
            "binned": neighbor.binned,
            "fingerprints": np.asarray(neighbor.fingerprints),
            "ion_code": np.asarray(neighbor.ion_code if neighbor.ion_code is not None else np.zeros(len(neighbor.binned), np.int8)),
            "chunk": np.asarray([neighbor.chunk]),
        }, {"schema": NEIGHBOR_SCHEMA, "row_order_sha256": row_hash, "n_mz_bins": int(cfg.n_mz_bins)})
    _mark(state, run_dir, "neighbor", t0)

    ds = dataset_from_structure_index(index, cfg, max_n=int(cfg.max_train_spectra), seed=42)
    if not np.array_equal(np.asarray(ds.library_indices), np.asarray(fit_take)):
        raise PipelineError("fitting subset drifted from fitting_subset_indices")
    models = _load_or_train_seeds(cfg, ds, run_dir, state, t0)
    del ds
    cfg.seed = 42

    class2_sha = file_sha256(cfg.external_candidates_path)
    state["class2_sha256"] = class2_sha
    state["pairs_signature"] = training_signature(
        fit_hash=state["fit_subset_hash"],
        library_row_hash=row_hash,
        checkpoint_sha256={seed: info["sha256"] for seed, info in state["checkpoints"].items()},
        class2_sha256=class2_sha,
        cfg=cfg,
    )
    external = load_external_structure_index(str(cfg.external_candidates_path), cfg)
    if external is None:
        raise PipelineError("Class 2 library failed to load")
    X, y, qid = _rr_train_pairs(cfg, index, external, neighbor, models, state, run_dir, t0)
    reranker_path = run_dir / "reranker.pkl"
    meta_path = run_dir / "reranker.json"
    reranker = None
    if reranker_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("signature") == state["pairs_signature"]:
            reranker = load_reranker(reranker_path)
            state["reranker"] = {
                "status": "reused",
                "signature": state["pairs_signature"],
                "sha256": file_sha256(reranker_path),
            }
            print("[reranker] reusing verified artifact")
    if reranker is None:
        reranker, report = fit_required_reranker(X, y, qid)
        save_reranker(reranker, reranker_path)
        atomic_write_json(meta_path, {"signature": state["pairs_signature"], "status": "fitted", "report": report})
        state["reranker"] = {
            "status": "fitted",
            "signature": state["pairs_signature"],
            "sha256": file_sha256(reranker_path),
        }
    _mark(state, run_dir, "reranker", t0)
    return _predict_and_publish(cfg, index, external, neighbor, models, reranker, run_dir, state, t0, submission_path)


def resolve_bundle(bundle_dir: Path | str | None) -> Path:
    if bundle_dir:
        path = Path(bundle_dir)
        if not (path / "run_manifest.json").exists():
            raise PipelineError(
                f"predict-only bundle is missing or incompatible at {path}. "
                "Attach the casmi_run directory saved from a train-and-predict output. "
                "This notebook does not train. Files in a deleted Kaggle session are gone; "
                "only a dataset you uploaded still exists."
            )
        return path
    hits = []
    if Path("/kaggle/input").exists():
        hits = [path.parent for path in Path("/kaggle/input").rglob("run_manifest.json")]
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise PipelineError(
            "predict-only bundle is missing. Attach the casmi_run output of a completed "
            "train-and-predict run as a Kaggle dataset and set BUNDLE_DIR if more than one "
            "manifest is present. This notebook does not train."
        )
    raise PipelineError(
        "multiple run_manifest.json files are attached. Set BUNDLE_DIR to the casmi_run "
        "directory. This notebook does not train."
    )


def run_predict_only(
    cfg: Config,
    *,
    bundle_dir: Path | str | None,
    submission_path: Path | str | None = None,
) -> dict[str, Any]:
    """Load a verified bundle and write submission.csv. Never trains."""
    from src.tpu_trainer import load_model
    from src.retrieval import load_external_structure_index
    from src.validation import checkpoint_extra, checkpoint_is_complete, model_config_hash

    assert_production_recipe(cfg)
    if bool(getattr(cfg, "use_observed_precursor", False)):
        raise PipelineError(f"{PRECURSOR_EXPERIMENT_ID} is not part of {STRATEGY_ID}")
    bound_threads(4)
    bundle = resolve_bundle(bundle_dir)
    manifest = json.loads((bundle / "run_manifest.json").read_text())
    problems: list[str] = []
    if manifest.get("schema") != MANIFEST_SCHEMA or manifest.get("strategy_id") != STRATEGY_ID:
        problems.append("manifest strategy/schema")
    if manifest.get("fast_dev_run"):
        problems.append("bundle was a fast run")
    versions = dependency_versions()
    src_hash = source_hash(CHECKPOINT_SOURCE)
    config_hash = model_config_hash(cfg)
    fit_hash = str(manifest.get("fit_subset_hash") or "")
    models = []
    checkpoints = {}
    for seed in PRODUCTION_SEEDS:
        path = bundle / "checkpoints" / f"spec2fp_s{seed}.pt"
        if not path.exists():
            problems.append(f"missing seed {seed}")
            continue
        extra = checkpoint_extra(path)
        ok, why = checkpoint_is_complete(
            extra, cfg, fit_hash=fit_hash, config_hash=config_hash, seed=seed,
            source_hash=src_hash, versions=versions,
        )
        if not ok:
            problems.append(f"seed {seed}: {why}")
        else:
            models.append(load_model(path, cfg))
            checkpoints[str(seed)] = {"sha256": file_sha256(path), "complete": True}
    reranker_path = bundle / "reranker.pkl"
    class2_sha = None
    if cfg.external_candidates_path is not None and Path(cfg.external_candidates_path).exists():
        class2_sha = file_sha256(cfg.external_candidates_path)
    if not reranker_path.exists():
        problems.append("missing reranker")
        reranker = None
    else:
        provenance = reranker_provenance_problems(manifest, reranker_path, class2_sha)
        problems.extend(provenance)
        if provenance:
            reranker = None
        else:
            try:
                reranker = load_reranker(reranker_path)
            except Exception as exc:
                problems.append(f"reranker unreadable: {exc}")
                reranker = None
    try:
        lib_data, lib_side = load_array_cache(bundle / "library.npz", expect_schema=LIBRARY_SCHEMA)
        nbr_data, nbr_side = load_array_cache(bundle / "neighbor.npz", expect_schema=NEIGHBOR_SCHEMA)
    except PipelineError as exc:
        problems.append(str(exc))
        lib_data = nbr_data = lib_side = nbr_side = None
    if problems:
        raise PipelineError(
            "predict-only bundle is missing or incompatible: " + "; ".join(problems) + ". This notebook does not train."
        )
    if cfg.train_path.exists() and lib_side.get("train") != content_id(cfg.train_path):
        raise PipelineError("library cache does not match the attached train.parquet. This notebook does not train.")
    index = library_from_cache(lib_data)
    if nbr_side.get("row_order_sha256") != lib_side.get("row_order_sha256"):
        raise PipelineError("neighbor cache does not match the library row order. This notebook does not train.")
    neighbor = neighbor_from_cache(nbr_data)
    if cfg.external_candidates_path is None or not Path(cfg.external_candidates_path).exists():
        raise PipelineError("Class 2 library is missing. This notebook does not train.")
    external = load_external_structure_index(str(cfg.external_candidates_path), cfg)
    if external is None:
        raise PipelineError("Class 2 library failed to load. This notebook does not train.")
    _require_class2_pool(Path(cfg.external_candidates_path))
    if submission_path is None:
        submission_path = (Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")) / "submission.csv"
    submission_path = Path(submission_path)
    retire_previous_submission(submission_path)
    work = Path("/kaggle/working") / "casmi_run" if Path("/kaggle/working").exists() else submission_path.parent / "casmi_run"
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    state = {
        "schema": MANIFEST_SCHEMA,
        "strategy_id": STRATEGY_ID,
        "inactive_experiments": list(INACTIVE_EXPERIMENTS),
        "checkpoint_source_hash": src_hash,
        "pairs_source_hash": manifest.get("pairs_source_hash"),
        "versions": versions,
        "checkpoints": checkpoints,
        "reranker": {"status": "loaded"},
        "timings": {},
        "fit_subset_hash": fit_hash,
        "model_config_hash": config_hash,
        "bundle": str(bundle),
        "predict_only": True,
    }
    _write_manifest(work, state)
    return _predict_and_publish(cfg, index, external, neighbor, models, reranker, work, state, t0, submission_path)
