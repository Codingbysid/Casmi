#!/usr/bin/env python3
"""Generate the standalone Kaggle GPU notebook from the src/ package.

Do not hand-edit the generated notebooks. Edit src/ or this script and rerun
``python scripts/make_kaggle_notebook.py``.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
SRC_FILES = [
    "src/__init__.py",
    "src/config.py",
    "src/chem.py",
    "src/preprocessing.py",
    "src/data.py",
    "src/model.py",
    "src/models/__init__.py",
    "src/models/smiles_decoder.py",
    "src/chem_tokenizer.py",
    "src/smiles_tokenizer.py",
    "src/smiles_vocab.json",
    "src/smiles_train.py",
    "src/retrieval.py",
    "src/analogs.py",
    "src/neighbors.py",
    "src/calibration.py",
    "src/calibration_data.json",
    "src/ranker.py",
    "src/rerank.py",
    "src/validation.py",
    "src/formula_head.py",
    "src/metrics.py",
    "src/tpu_trainer.py",
    "src/infer.py",
    "src/train_tpu.py",
]
NOTEBOOK_NAMES = ("kaggle_submission.ipynb", "kaggle_gpu_submission.ipynb", "kaggle_submission_tpu.ipynb")


def cell_md(nb, text: str) -> None:
    nb.cells.append(nbf.v4.new_markdown_cell(text))


def cell_code(nb, text: str) -> None:
    nb.cells.append(nbf.v4.new_code_cell(text.strip() + "\n"))


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def main() -> None:
    files = {}
    for rel in SRC_FILES:
        files[rel] = (ROOT / rel).read_text()
    src_hashes = {rel: hashlib.sha256(text.encode("utf-8")).hexdigest() for rel, text in files.items()}
    source_commit = _git("rev-parse", "--short", "HEAD")
    source_dirty = bool(_git("status", "--porcelain", "--", "src", "scripts"))

    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    # GPU session. Internet stays off. The model size is the 0.139/0.150 net.
    nb.metadata["accelerator"] = "GPU"
    nb.metadata["kaggle"] = {
        "accelerator": "gpu",
        "isInternetEnabled": False,
        "isGpuEnabled": True,
        "language": "python",
        "sourceType": "notebook",
    }

    cell_md(
        nb,
        """# Enveda CASMI 2026 — Molecule ID from Mass Spectra (GPU)

Configured for **Kaggle Run All** on a **CUDA GPU** (Python 3.12) with **internet off**. No TPU.

Attach **three** inputs (the kernel source must stay under **1 MB**, so the NP pool and the RDKit wheel cannot live inside the notebook):

1. Competition data: `train.parquet`, `test.parquet`, `sample_submission.csv`
2. Class 2 COCONUT ∪ LOTUS table: the dataset that contains `class2_candidates.parquet` (~61 MB, ~453k structures). Kaggle may name the folder `class2_candidates.parquet2`.
3. **RDKit wheel (~35 MB):** `kaggle_rdkit_wheels/rdkit-2025.9.5-cp312-cp312-manylinux_2_28_x86_64.whl` uploaded as a dataset. The 20 KB stub named `rdkit-2025-9-5` is not the wheel.

Optional fourth input: a previous run's `casmi_run/checkpoints/` folder. Checkpoints are reused only when their recorded fit-subset hash, model-config hash, seed, and `completed_epochs == 4` all match; otherwise the seeds are trained again.

**Experiment vs the 0.150 submission (one change):** the hand-written compete score for the *unlocked* candidate pool is replaced by a small, deterministic, source-aware gradient-boosted reranker (`sklearn` `HistGradientBoostingClassifier`, fixed configuration) applied to the top-200 unlocked candidates. Locked Class 1 hits (cosine ≥ 0.75 and ≥ 5 matched peaks) never move. The reranker is fitted and audited inside this run on train skeletons Spec2FP never saw, with the promotion rule declared before the audit. If the rule fails, `submission.csv` is produced by the exact 0.150 ordering and the run reports why.

Frozen (not experiments): Spec2FP training recipe (focal BCE, 3 seeds, 120k structures, 4 epochs), neighbor blend (k=20, a=0.5), spectral gates, Class 1 lock, candidate generation, analog generation, InChIKey14 dedup, RDKit 2025.09.5.

Outputs: `/kaggle/working/submission.csv` plus `/kaggle/working/casmi_run/` (checkpoints, reranker, manifest, splits, validation results, candidate traces, diagnostics archive). Nothing is deleted at the end.

Keep the existing 0.150 Kaggle submission selected unless this run scores strictly higher.
""",
    )

    cell_code(
        nb,
        """
# Offline RDKit install — MUST be the first code cell (internet OFF).
# Need a ~35 MB .whl, not the 20 KB stub dataset named rdkit-2025-9-5.
import os, sys, shutil, subprocess
from importlib import invalidate_caches
from pathlib import Path

def _rdkit_ok():
    try:
        from rdkit import Chem
        ver = getattr(getattr(Chem, "rdBase", Chem), "rdkitVersion", None) or "unknown"
        mol = Chem.MolFromSmiles("CCO")
        return bool(mol is not None), str(ver)
    except Exception:
        return False, ""

def _looks_like_wheel(path: Path) -> bool:
    name = path.name.lower()
    if name.endswith(".whl"):
        return True
    if "rdkit" in name and any(x in name for x in ("cp31", "manylinux", "win", "macosx")):
        return True
    try:
        with path.open("rb") as fh:
            magic = fh.read(4)
        # ZIP / wheel local file header
        if magic[:2] == b"PK":
            return "rdkit" in name or path.stat().st_size > 1_000_000
    except Exception:
        return False
    return False

def find_rdkit_wheels() -> list[Path]:
    inp = Path("/kaggle/input")
    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
    print("listing /kaggle/input ...")
    if inp.exists():
        for p in sorted(inp.rglob("*")):
            kind = "dir" if p.is_dir() else "file"
            size = p.stat().st_size if p.is_file() else 0
            print(f"  {kind} {p} bytes={size}")
    small: list[tuple[Path, int]] = []
    hits: list[Path] = []
    if not inp.exists():
        return hits
    for p in inp.rglob("*"):
        if not p.is_file():
            continue
        parent = str(p.parent).lower()
        if "rdkit" not in parent and "rdkit" not in p.name.lower():
            continue
        size = p.stat().st_size
        if size < 5_000_000:
            small.append((p, size))
            continue
        if _looks_like_wheel(p) or "rdkit" in p.name.lower() or p.suffix.lower() == ".whl":
            hits.append(p)
    if small:
        print("rdkit-named files too small to be a wheel (need ~35 MB):")
        for p, size in small:
            print(f"  {p} bytes={size}")
    uniq: list[Path] = []
    seen: set[str] = set()
    for p in hits:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    uniq.sort(
        key=lambda p: (
            0 if p.suffix.lower() == ".whl" else 1,
            0 if tag in p.name else 1,
            0 if "rdkit" in p.name.lower() else 1,
            -p.stat().st_size,
        )
    )
    return uniq

RDKIT_WHEEL_PATH = None
ok, ver = _rdkit_ok()
if ok:
    print("rdkit already installed", ver)
else:
    print("python", sys.version)
    wheels = find_rdkit_wheels()
    print("rdkit wheels found", [str(w) for w in wheels])
    if not wheels:
        raise FileNotFoundError(
            "No ~35 MB RDKit wheel under /kaggle/input. "
            "The attached 'rdkit-2025-9-5' file is only ~20 KB — that is not the wheel. "
            "On your laptop, upload this file as a NEW Kaggle dataset (do not reuse the stub): "
            "kaggle_rdkit_wheels/rdkit-2025.9.5-cp312-cp312-manylinux_2_28_x86_64.whl "
            "(~35 MB). Then Add Input that dataset. Keep internet OFF."
        )
    src = wheels[0]
    RDKIT_WHEEL_PATH = str(src)
    work = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("/tmp")
    dest_dir = work / "_rdkit_wheels"
    dest_dir.mkdir(parents=True, exist_ok=True)
    py = f"cp{sys.version_info.major}{sys.version_info.minor}"
    dest = dest_dir / f"rdkit-2025.9.5-{py}-{py}-manylinux_2_28_x86_64.whl"
    print("copying", src, "->", dest, "src_size", src.stat().st_size)
    shutil.copy2(src, dest)
    cmd = [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", str(dest)]
    print("install", " ".join(cmd))
    subprocess.check_call(cmd)
    invalidate_caches()
    ok, ver = _rdkit_ok()
    if not ok:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
             f"--find-links={dest_dir}", "rdkit"]
        )
        invalidate_caches()
        ok, ver = _rdkit_ok()
    if not ok:
        raise ImportError(f"pip installed {src} but `from rdkit import Chem` still fails")
    print("rdkit", ver)
assert _rdkit_ok()[0], "HAS_RDKIT must be True after the offline wheel install"
print("HAS_RDKIT", True, "rdkit", _rdkit_ok()[1])
""",
    )

    cell_code(
        nb,
        f"""
# ===== Fully configured for Run All on Kaggle GPU =====
# Session: CUDA GPU (P100 or T4). Do not start a TPU or CPU session.
# Internet: OFF.
# Add Data:
#   1) Enveda CASMI 2026 competition
#   2) class2_candidates.parquet (folder may be class2_candidates.parquet2)
#   3) rdkit-cp312-wheel (~35 MB .whl, not the 20 KB stub)
#   4) optional: a previous run's casmi_run/checkpoints folder (verified before reuse)
SOURCE_COMMIT = {source_commit!r}       # git HEAD when this notebook was generated
SOURCE_DIRTY = {source_dirty!r}         # True if src/ or scripts/ had uncommitted edits
REQUIRE_CLASS2 = True         # fail fast if class2_candidates.parquet is not attached
REQUIRE_CUDA = True           # fail before training when no CUDA device is visible
FAST_DEV_RUN = False          # True = small debug pass; keep False for a real submit
PREFER_XLA = False            # TPU off: CPU/GPU path
TRAIN_DECODER = False         # Class 3 decoder stays off (emits garbage)
ENSEMBLE_SEEDS = 3            # average pre-sigmoid Spec2FP logits; all seeds must finish
MAX_TRAIN_SPECTRA = 120_000   # 0.139/0.150 recipe; 280k/6ep scored 0.126
USE_REPRESENTATIVE_METADATA = False  # 0.139 feature path; that experiment already ran (0.134)
RESTORE_BEST_CHECKPOINT = False      # no demonstrated gain; keep last-epoch selection
NUM_EPOCHS = 4
BATCH_SIZE = 64
TRAIN_TIME_LIMIT_S = 1.5 * 3600.0    # per seed; a timeout is a failure, not a quiet fallback
USE_NEIGHBOR_FP = True
NEIGHBOR_BLEND = 0.5
TOP_N_PEAKS = 128
N_MZ_BINS = 512
FP_BITS = 2048
SESSION_BUDGET_S = 8.5 * 3600.0
RANK_RESERVE_S = 1.0 * 3600.0        # kept free for production ranking + artifacts

# ----- the experiment: learned reranker for the unlocked pool -----
RERANKER_EXPERIMENT = True    # False = skip validation and ship the exact 0.150 ordering
RERANK_WINDOW = 200           # unlocked candidates (by 0.150 score) the reranker may reorder
REUSE_CHECKPOINTS = True      # reuse attached spec2fp_s{{seed}}.pt only after hash/epoch checks
RANK_WORKERS = 4              # fork workers for validation candidate generation
VALIDATION_SEED = 0
N_RR_TRAIN = 1200             # skeletons used to fit the reranker
N_DEV = 400                   # sanity only; never used for selection
N_AUDIT = 800                 # promotion decision
N_PANEL = 400                 # enveda-np-examples (Bruker) skeletons outside the fit subset
VALIDATION_BUDGET_S = 3.0 * 3600.0
""",
    )

    cell_code(
        nb,
        """
import os, sys, json, time, math, traceback, hashlib
from pathlib import Path

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("OMP_NUM_THREADS", str(min(8, os.cpu_count() or 4)))
os.environ.setdefault("MKL_NUM_THREADS", os.environ["OMP_NUM_THREADS"])

KAGGLE_WORKING = Path("/kaggle/working")
IS_KAGGLE = KAGGLE_WORKING.exists()
ROOT = KAGGLE_WORKING if IS_KAGGLE else Path(".").resolve()
(ROOT / "src" / "models").mkdir(parents=True, exist_ok=True)
(ROOT / "artifacts").mkdir(parents=True, exist_ok=True)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

T_START = time.time()
TIMINGS = {}
RUN_ID = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
RUN_DIR = (KAGGLE_WORKING if IS_KAGGLE else ROOT / "artifacts") / "casmi_run"
for sub in ("checkpoints", "diagnostics"):
    (RUN_DIR / sub).mkdir(parents=True, exist_ok=True)

def mark(stage):
    TIMINGS[stage] = round(time.time() - T_START, 1)
    print(f"[time] {stage} at elapsed_s={TIMINGS[stage]}")

print("ROOT", ROOT, "kaggle", IS_KAGGLE, "run_id", RUN_ID, "run_dir", RUN_DIR)
print("python", sys.version)
print("cpu_count", os.cpu_count())
print("source_commit", SOURCE_COMMIT, "dirty", SOURCE_DIRTY)
print("internet: OFF (notebook metadata isInternetEnabled=False). Toggle Internet off in the editor.")
if Path("/kaggle/input").exists():
    attached = sorted(p.name for p in Path("/kaggle/input").iterdir())
    print("attached /kaggle/input datasets:", attached)
    if not attached:
        print(
            "WARNING: /kaggle/input is empty — Add Data for competition + "
            "class2 parquet + a ~35 MB rdkit .whl"
        )
else:
    print("not on Kaggle; using local data/")
""",
    )

    payload = repr(json.dumps(files))
    cell_md(nb, "## 1. Materialize the `src/` package (offline)")
    cell_code(
        nb,
        f"""
import json
from pathlib import Path

FILES = json.loads({payload})
SRC_HASHES = {{}}
for rel, content in FILES.items():
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    SRC_HASHES[rel] = hashlib.sha256(content.encode("utf-8")).hexdigest()
    print("wrote", path, "bytes", len(content))
""",
    )

    cell_md(nb, "## 1b. Locate the attached Class 2 parquet (COCONUT ∪ LOTUS)")
    cell_code(
        nb,
        """
def find_class2_table():
    names = {
        "class2_candidates.parquet",
        "class2_candidates.csv.gz",
        "class2_candidates.csv",
        "lotus.parquet",
    }
    roots = [Path("/kaggle/input"), ROOT / "data", Path("data")]
    explicit = [
        Path("/kaggle/input/class2_candidates.parquet2/class2_candidates.parquet"),
        Path("/kaggle/input/class2-candidates-parquet2/class2_candidates.parquet"),
        Path("/kaggle/input/class2_candidates.parquet/class2_candidates.parquet"),
    ]
    hits = [p for p in explicit if p.exists() and p.is_file()]
    for root in roots:
        if not root.exists():
            continue
        try:
            for p in root.rglob("*"):
                if not p.is_file():
                    continue
                low = p.name.lower()
                parent = p.parent.name.lower()
                if low in names or ("class2" in low and p.suffix.lower() in {".parquet", ".csv", ".gz"}):
                    hits.append(p)
                elif "class2" in parent and low.endswith(".parquet"):
                    hits.append(p)
        except Exception:
            continue
    uniq = []
    seen = set()
    for p in hits:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(p)
    # Prefer the largest parquet so the 453k COCONUT pool wins over old LOTUS (~19 MB).
    uniq.sort(key=lambda p: (0 if p.suffix.lower() == ".parquet" else 1, -p.stat().st_size))
    if uniq:
        print("class2 candidates found:")
        for p in uniq:
            print(" ", p, "bytes", p.stat().st_size)
    return uniq[0] if uniq else None

CLASS2_PATH = find_class2_table()
if CLASS2_PATH is None:
    msg = (
        "class2_candidates.parquet not attached. Add the dataset whose folder may be "
        "class2_candidates.parquet2 and whose file is class2_candidates.parquet "
        "(~61 MB COCONUT ∪ LOTUS, ~453k structures). Kernel source must stay < 1 MB."
    )
    if IS_KAGGLE and REQUIRE_CLASS2:
        raise FileNotFoundError(msg)
    print("WARNING:", msg, "Ranking will be Class 1 only.")
else:
    print("class2 table", CLASS2_PATH, "bytes", CLASS2_PATH.stat().st_size)
    nbytes = CLASS2_PATH.stat().st_size
    if nbytes < 5_000_000:
        print(
            "WARNING: Class 2 file is < 5 MB — this may be the old 57k test-mass slice. "
            "Attach the ~61 MB parquet (~453k structures, 50-2000 Da)."
        )
    elif nbytes < 40_000_000:
        print(
            "WARNING: Class 2 file is < 40 MB — this looks like LOTUS-only (~133k). "
            "V6 expects COCONUT ∪ LOTUS ≈ 453k (~61 MB)."
        )
    else:
        print("class2 parquet size looks like the V6 COCONUT ∪ LOTUS pool")
""",
    )

    cell_md(nb, "## 1c. Dependency and accelerator preflight (before any training time is spent)")
    cell_code(
        nb,
        """
import numpy as np
import pandas as pd
import torch

print("numpy", np.__version__, "pandas", pd.__version__, "torch", torch.__version__)
try:
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))
except Exception:
    pass

try:
    from rdkit import Chem
    RDKIT_VERSION = Chem.rdBase.rdkitVersion
    print("rdkit", RDKIT_VERSION)
    HAS_RDKIT = True
    _probe = Chem.MolFromSmiles("CCO")
    assert _probe is not None, "RDKit imported but MolFromSmiles('CCO') returned None"
except Exception as e:
    HAS_RDKIT = False
    raise ImportError(
        "HAS_RDKIT is False. The first cell must pip-install the attached "
        "~35 MB rdkit .whl. Attaching a 20 KB stub does not install the package."
    ) from e

import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier  # the only estimator family used
print("sklearn", sklearn.__version__, "HistGradientBoostingClassifier available")
try:
    import threadpoolctl
    print("threadpoolctl", threadpoolctl.__version__)
except Exception:
    print("threadpoolctl missing (worker thread limiting will rely on torch only)")

CUDA_OK = torch.cuda.is_available()
GPU_NAME = torch.cuda.get_device_name(0) if CUDA_OK else None
print("cuda", CUDA_OK, "gpu", GPU_NAME)
if REQUIRE_CUDA and not FAST_DEV_RUN and not CUDA_OK:
    raise RuntimeError(
        "No CUDA device. This notebook must run on a Kaggle GPU session; "
        "3 seeds x 4 epochs on CPU would exceed the session and silently degrade the run."
    )

try:
    import torch_xla.core.xla_model as xm  # noqa: F401
    HAS_XLA = True
    print("torch_xla present (not used unless PREFER_XLA=True)")
except Exception:
    HAS_XLA = False
    print("torch_xla not available (expected on CPU/GPU sessions)")
VERSIONS = {
    "python": sys.version.split()[0],
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "torch": torch.__version__,
    "sklearn": sklearn.__version__,
    "rdkit": RDKIT_VERSION,
    "rdkit_wheel": RDKIT_WHEEL_PATH,
    "cuda": CUDA_OK,
    "gpu": GPU_NAME,
}
mark("preflight")
""",
    )

    cell_md(nb, "## 2. Config")
    cell_code(
        nb,
        """
from src.config import get_config
from src.chem import has_rdkit, inchikey14_from_smiles
from src.data import dataset_from_structure_index, fitting_subset_indices, load_test, load_train_slice
from src.infer import (
    build_library_from_frame,
    build_library_from_train,
    featurize_test_molecules,
    predict_fingerprint_ensemble,
    predict_fingerprints,
    predict_test,
    rank_molecules,
)
from src.metrics import mrr_at_k
from src.retrieval import load_external_structure_index
from src.tpu_trainer import get_device, load_model, train_spec2fp
from src import validation as V
from src.rerank import FEATURE_NAMES, RERANKER_PARAMS, fit_reranker, load_reranker, save_reranker

def find_competition_dir() -> Path:
    candidates = [
        Path("/kaggle/input/enveda-casmi26-molecule-id-mass-spectra"),
        Path("/kaggle/input/enveda-CASMI26-molecule-id-mass-spectra"),
        ROOT / "data",
        ROOT / "enveda-CASMI26-molecule-id-mass-spectra",
    ]
    input_root = Path("/kaggle/input")
    if input_root.exists():
        for parquet in input_root.rglob("train.parquet"):
            parent = parquet.parent
            if (parent / "test.parquet").exists():
                candidates.insert(0, parent)
    for cand in candidates:
        if cand.exists() and (cand / "train.parquet").exists():
            return cand
    raise FileNotFoundError("Could not find train.parquet under /kaggle/input or ./data")

data_dir = find_competition_dir()
print("competition data_dir", data_dir)

# Same 0.139/0.150-sized net on GPU and CPU. Do not enlarge the model on CUDA.
cfg = get_config(
    batch_size=BATCH_SIZE,
    num_epochs=NUM_EPOCHS,
    max_train_spectra=MAX_TRAIN_SPECTRA,
    use_representative_metadata=bool(USE_REPRESENTATIVE_METADATA),
    restore_best_checkpoint=bool(RESTORE_BEST_CHECKPOINT),
    train_time_limit_s=TRAIN_TIME_LIMIT_S,
    top_n_peaks=TOP_N_PEAKS,
    n_mz_bins=N_MZ_BINS,
    fp_bits=FP_BITS,
    log_every=50,
    d_model=128,
    n_heads=4,
    n_transformer_layers=2,
    n_conv_channels=64,
    smiles_d_model=128,
    smiles_nhead=4,
    smiles_num_layers=2,
    smiles_dim_feedforward=512,
    smiles_num_epochs=1,
    smiles_batch_size=64,
    smiles_num_samples=48,
    smiles_gen_max_len=80,
    smiles_mass_ppm=50.0,
    num_workers=0,
    max_mass_candidates=8192,
    use_neighbor_fp=bool(USE_NEIGHBOR_FP),
    neighbor_blend=float(NEIGHBOR_BLEND),
    ensemble_seeds=int(ENSEMBLE_SEEDS),
    rerank_window=int(RERANK_WINDOW),
)
cfg.data_dir = data_dir
cfg.artifact_dir = Path("/tmp/casmi_artifacts") if IS_KAGGLE else (ROOT / "artifacts")
cfg.artifact_dir.mkdir(parents=True, exist_ok=True)
cfg.external_candidates_path = CLASS2_PATH
print("class2", cfg.external_candidates_path, "exists", CLASS2_PATH is not None and Path(CLASS2_PATH).exists())
print("train", cfg.train_path.exists(), cfg.train_path)
print("test", cfg.test_path.exists(), cfg.test_path)
assert cfg.train_path.exists() and cfg.test_path.exists(), "Attach the competition dataset."
CONFIG_HASH = V.model_config_hash(cfg)
print("model_config_hash", CONFIG_HASH)
INPUT_FINGERPRINTS = {
    "train.parquet": V.file_fingerprint(cfg.train_path),
    "test.parquet": V.file_fingerprint(cfg.test_path),
    "sample_submission.csv": V.file_fingerprint(cfg.sample_submission_path),
    "class2_candidates": V.file_fingerprint(CLASS2_PATH) if CLASS2_PATH is not None else {"exists": False},
    "rdkit_wheel": V.file_fingerprint(RDKIT_WHEEL_PATH) if RDKIT_WHEEL_PATH else {"exists": False},
}
print(json.dumps(INPUT_FINGERPRINTS, indent=1))
mark("config")
""",
    )

    cell_md(nb, "## 3. Build the candidate library from train (mass + fingerprints + representative MS2)")
    cell_code(
        nb,
        """
t_lib = time.time()
if FAST_DEV_RUN:
    print("FAST_DEV_RUN: library from first 8,000 train rows")
    slice_df = load_train_slice(cfg, 8_000)
    index = build_library_from_frame(slice_df, cfg)
else:
    index = build_library_from_train(cfg, progress=True)

print(
    f"library n={len(index.smiles)} peaks={None if index.peak_mz is None else index.peak_mz.shape} "
    f"fps={index.fingerprints.shape} in {time.time()-t_lib:.1f}s"
)
FIT_SEED = int(cfg.seed)  # the dataset below is drawn once, before any per-seed cfg.seed change
n_fit = 4_000 if FAST_DEV_RUN else int(cfg.max_train_spectra)
fit_take = fitting_subset_indices(len(index.smiles), n_fit, seed=FIT_SEED)
FIT_HASH = V.fit_subset_hash(index, fit_take)
print(f"fit subset n={fit_take.size} of {len(index.smiles)} seed={FIT_SEED} fit_hash={FIT_HASH}")
mark("library")
""",
    )

    cell_md(nb, "## 4. Train (or verify and reuse) the 3 Spec2FP seeds — every seed must finish 4 epochs")
    cell_code(
        nb,
        """
device, use_xla = get_device(prefer_xla=bool(PREFER_XLA))
print("train device", device, "xla", use_xla, "d_model", cfg.d_model, "batch", cfg.batch_size)

if FAST_DEV_RUN:
    cfg.num_epochs = 1
    cfg.train_time_limit_s = 180.0
    cfg.max_train_spectra = 4_000
    cfg.log_every = 20
    CONFIG_HASH = V.model_config_hash(cfg)

t_data = time.time()
print(f"building Spec2FP dataset from library n={len(index.smiles)} cap={n_fit} neighbor_fp={cfg.use_neighbor_fp} blend={cfg.neighbor_blend}")
ds = dataset_from_structure_index(index, cfg, max_n=n_fit, seed=FIT_SEED)
assert np.array_equal(np.asarray(ds.library_indices), np.asarray(fit_take)), "fit subset drifted from fitting_subset_indices"
probe = ds[0]
for k, expected in {
    "peak_mz": (cfg.top_n_peaks,),
    "peak_intensity": (cfg.top_n_peaks,),
    "peak_nl": (cfg.top_n_peaks,),
    "peak_mask": (cfg.top_n_peaks,),
    "binned": (cfg.n_mz_bins,),
    "fingerprint": (cfg.fp_bits,),
}.items():
    assert tuple(probe[k].shape) == expected, (k, probe[k].shape, expected)
print(f"dataset ready in {time.time()-t_data:.1f}s  n={len(ds)}")

def find_reusable_checkpoint(seed):
    if not REUSE_CHECKPOINTS or not Path("/kaggle/input").exists():
        return None, "reuse disabled or no inputs"
    reasons = []
    for p in Path("/kaggle/input").rglob(f"spec2fp_s{seed}.pt"):
        try:
            extra = V.checkpoint_extra(p)
        except Exception as exc:
            reasons.append(f"{p}: unreadable ({exc})")
            continue
        ok, why = V.checkpoint_is_complete(extra, cfg, fit_hash=FIT_HASH, config_hash=CONFIG_HASH, seed=seed)
        if ok:
            return p, "verified"
        reasons.append(f"{p}: {why}")
    return None, "; ".join(reasons) if reasons else "no candidate file"

t_tr = time.time()
models = []
TRAINING_RECORDS = []
n_seeds = 1 if FAST_DEV_RUN else int(ENSEMBLE_SEEDS)
seed_list = [42, 7, 123, 99, 2024][:n_seeds]
for si, seed in enumerate(seed_list):
    cfg.seed = int(seed)
    cfg.ckpt_name = f"spec2fp_s{seed}.pt"
    cfg.train_time_limit_s = float(TRAIN_TIME_LIMIT_S if not FAST_DEV_RUN else 180.0)
    reuse_path, reuse_note = find_reusable_checkpoint(seed)
    record = {"seed": seed, "reused": False, "reuse_note": reuse_note}
    if reuse_path is not None:
        print(f"seed={seed}: reusing verified checkpoint {reuse_path}")
        m = load_model(reuse_path, cfg)
        shutil.copy2(reuse_path, RUN_DIR / "checkpoints" / cfg.ckpt_name)
        record.update(reused=True, path=str(RUN_DIR / "checkpoints" / cfg.ckpt_name), extra=V.checkpoint_extra(reuse_path))
    else:
        print(f"seed={seed}: training (reuse: {reuse_note}) time_limit_s={cfg.train_time_limit_s:.0f} epochs={cfg.num_epochs}")
        t_seed = time.time()
        m = train_spec2fp(
            ds,
            cfg,
            device=device,
            seed=int(seed),
            ckpt_name=cfg.ckpt_name,
            time_limit_s=cfg.train_time_limit_s,
            extra_meta={
                "fit_subset_hash": FIT_HASH,
                "model_config_hash": CONFIG_HASH,
                "library_n": int(len(index.smiles)),
                "run_id": RUN_ID,
                "source_commit": SOURCE_COMMIT,
            },
        )
        extra = V.checkpoint_extra(cfg.checkpoint_path)
        ok, why = V.checkpoint_is_complete(extra, cfg, fit_hash=FIT_HASH, config_hash=CONFIG_HASH, seed=seed)
        if not ok:
            raise RuntimeError(
                f"seed={seed} did not complete training ({why}; extra={extra}). "
                "Refusing to rank with a partial ensemble."
            )
        shutil.copy2(cfg.checkpoint_path, RUN_DIR / "checkpoints" / cfg.ckpt_name)
        record.update(path=str(RUN_DIR / "checkpoints" / cfg.ckpt_name), extra=extra, seconds=round(time.time() - t_seed, 1))
    models.append(m.to("cpu").eval())
    TRAINING_RECORDS.append(record)
    print(f"seed={seed} done; elapsed_s={time.time()-T_START:.0f}")

if len(models) != n_seeds:
    raise RuntimeError(f"expected {n_seeds} models, got {len(models)}")
cfg.seed = FIT_SEED
model = models[0]
print("ensemble models", len(models), "train wall", round(time.time() - t_tr, 1), "s")
del ds
mark("training")
""",
    )

    cell_md(nb, "## 5. Validation — partitions outside the fitting subset, two regimes, one candidate pool per query")
    cell_code(
        nb,
        """
print("PROMOTION_RULE (declared before any audit number exists):")
print(json.dumps(V.PROMOTION_RULE, indent=1))

plan = None
queries = []
results = {}
reranker = None
fit_report = {"status": "not attempted"}
dev_eval = audit_eval = panel_eval = None
training_counts = {}
coverage_natural = {}
inserted_stats = {}
skipped_queries = {}
panel_instruments = {}
validation_notes = []

remaining = SESSION_BUDGET_S - (time.time() - T_START)
val_budget = min(float(VALIDATION_BUDGET_S), remaining - float(RANK_RESERVE_S))
min_val_budget = 20 * 60.0
if FAST_DEV_RUN:
    val_budget = min(val_budget, 900.0)
    min_val_budget = 120.0
RUN_VALIDATION = bool(RERANKER_EXPERIMENT) and CLASS2_PATH is not None and val_budget > min_val_budget
print(f"remaining_s={remaining:.0f} validation_budget_s={val_budget:.0f} run_validation={RUN_VALIDATION}")
if not RUN_VALIDATION:
    validation_notes.append(f"validation skipped: experiment={RERANKER_EXPERIMENT} class2={CLASS2_PATH is not None} budget_s={val_budget:.0f}")

if RUN_VALIDATION:
    t_v = time.time()
    sizes = {"RR_TRAIN": int(N_RR_TRAIN), "DEV": int(N_DEV), "AUDIT": int(N_AUDIT), "PANEL": int(N_PANEL)}
    if FAST_DEV_RUN:
        sizes = {"RR_TRAIN": 60, "DEV": 20, "AUDIT": 40, "PANEL": 10}
    panel_keys, panel_instruments, panel_rows_n = V.panel_metadata(cfg, V.PANEL_LIB)
    key_to_row = {str(k): i for i, k in enumerate(index.inchikey14)}
    panel_rows = [key_to_row[k] for k in panel_keys if k in key_to_row]
    print(f"panel skeletons in library: {len(panel_rows)} (rows in lib {panel_rows_n})")
    plan = V.plan_holdouts(index, fit_take, sizes=sizes, panel_rows=panel_rows, seed=int(VALIDATION_SEED))
    V.check_plan_disjoint(plan, fit_take)
    wanted = {str(index.inchikey14[r]) for rows in plan.partitions.values() for r in rows}
    acq = V.collect_acquisitions(
        cfg,
        wanted,
        panel_lib=V.PANEL_LIB,
        panel_raw_keys={str(index.inchikey14[r]) for r in plan.partitions["PANEL"]},
        max_row_groups=3 if FAST_DEV_RUN else None,
    )
    queries, skipped_queries = V.build_queries(plan, acq, index, cfg)
    del acq
    V.write_json(
        RUN_DIR / "splits.json",
        {
            **plan.to_jsonable(),
            "sizes_requested": sizes,
            "panel_instruments": panel_instruments,
            "queries": [
                {"qid": q.qid, "partition": q.partition, "regime": q.regime, "row": q.row,
                 "truth_raw_key": q.truth.raw_key, "truth_canonical": q.truth.canonical, **q.meta}
                for q in queries
            ],
            "skipped": skipped_queries,
        },
    )
    mark("validation_partitions")
""",
    )

    cell_code(
        nb,
        """
ext_prod = load_external_structure_index(None, cfg) if CLASS2_PATH is not None else None
if ext_prod is None:
    print("[class2] no external pool loaded")

if RUN_VALIDATION:
    from src.neighbors import SpectralNeighborIndex, blend_predicted_fingerprints

    u_queries = [q for q in queries if q.regime == "U"]
    k_queries = [q for q in queries if q.regime == "K"]
    truths = [q.truth for q in u_queries]
    coverage_natural = V.natural_coverage(ext_prod, truths)
    print("natural coverage of holdout truths in the production pool:", coverage_natural)
    ext_val, inserted_stats = V.external_with_truths(ext_prod, truths, cfg)
    masked = V.masked_library(index, plan.masked_rows())
    assert len(masked.smiles) == len(index.smiles) - len(plan.masked_rows())
    masked_keys = set(masked.inchikey14.tolist())
    for q in u_queries[:50]:
        assert q.truth.raw_key not in masked_keys, q.qid

    pred_fps = {}
    t_fp = time.time()
    if cfg.use_neighbor_fp:
        nbr_masked = SpectralNeighborIndex.from_structure_index(masked, cfg)
        nbr_full = SpectralNeighborIndex.from_structure_index(index, cfg)
    else:
        nbr_masked = nbr_full = None
    for regime, qs, nbr in (("U", u_queries, nbr_masked), ("K", k_queries, nbr_full)):
        if not qs:
            continue
        feats = [q.feat for q in qs]
        fps = predict_fingerprint_ensemble(models, feats, cfg)
        if nbr is not None:
            fps = blend_predicted_fingerprints(fps, feats, nbr, k=int(cfg.neighbor_k), alpha=float(cfg.neighbor_blend))
        for q, fp in zip(qs, fps):
            pred_fps[q.qid] = fp
        print(f"regime {regime}: fingerprints for {len(qs)} queries")
    del nbr_masked
    print(f"query fingerprints in {time.time()-t_fp:.0f}s")
    mark("validation_fingerprints")
""",
    )

    cell_code(
        nb,
        """
if RUN_VALIDATION:
    priority = {"RR_TRAIN": 0, "AUDIT": 1, "PANEL": 2, "DEV": 3}
    ordered = sorted(queries, key=lambda q: (priority[q.partition], q.regime, q.row))
    spent = time.time() - T_START
    budget_left = max(val_budget - (time.time() - t_v), 600.0)
    budget_u = 0.55 * budget_left
    budget_k = 0.45 * budget_left
    print(f"candidate generation budget: U={budget_u:.0f}s K={budget_k:.0f}s workers={RANK_WORKERS}")
    results_u = V.rank_queries(
        [q for q in ordered if q.regime == "U"], pred_fps, masked, ext_val, cfg,
        workers=int(RANK_WORKERS), time_budget_s=budget_u, label="U",
    )
    results_k = V.rank_queries(
        [q for q in ordered if q.regime == "K"], pred_fps, index, ext_prod, cfg,
        workers=int(RANK_WORKERS), time_budget_s=budget_k, label="K",
    )
    results = {**results_u, **results_k}
    by_part = {}
    for q in queries:
        if q.qid in results:
            key = f"{q.partition}_{q.regime}"
            by_part[key] = by_part.get(key, 0) + 1
    print("ranked queries by partition/regime:", by_part)
    del masked
    mark("validation_candidates")
""",
    )

    cell_md(nb, "## 5b. Fit the reranker on RR_TRAIN; DEV is a sanity check only")
    cell_code(
        nb,
        """
if RUN_VALIDATION:
    rr_train = [q for q in queries if q.partition == "RR_TRAIN"]
    X, y, qid, training_counts = V.training_rows(rr_train, results)
    print("training rows", X.shape, "positives", int(y.sum()), "queries", training_counts)
    assert X.shape[1] == len(FEATURE_NAMES)
    reranker, fit_report = fit_reranker(X, y, qid)
    print("fit report:", json.dumps({k: v for k, v in fit_report.items() if k not in ("feature_names", "monotonic")}, default=str))
    if reranker is not None:
        save_reranker(reranker, RUN_DIR / "reranker.pkl")
        reranker = load_reranker(RUN_DIR / "reranker.pkl")  # round-trip the artifact that ships
        dev_q = [q for q in queries if q.partition == "DEV"]
        dev_eval = V.evaluate_arms(dev_q, results, reranker, cfg)
        print("DEV (sanity, not used for selection):")
        for regime, block in dev_eval["regimes"].items():
            print(f"  {regime}: baseline={block['baseline']} rerank={block['rerank']} delta={block['paired_delta']}")
        print("  rankings changed:", dev_eval["pooled"]["n_rankings_changed"], "coverage:", dev_eval["coverage"])
        if dev_eval["pooled"]["n_rankings_changed"] == 0:
            validation_notes.append("reranker never changed a DEV ordering (inert model)")
        try:
            from sklearn.inspection import permutation_importance
            Xd, yd, qd, _ = V.training_rows(dev_q, results)
            if Xd.shape[0] > 0 and yd.sum() > 0:
                imp = permutation_importance(reranker.model, Xd, yd, scoring="roc_auc", n_repeats=3, random_state=0)
                order = np.argsort(-imp.importances_mean)
                fit_report["dev_permutation_importance"] = {FEATURE_NAMES[i]: float(imp.importances_mean[i]) for i in order}
                print("  top features by DEV permutation importance (roc_auc):", [(FEATURE_NAMES[i], round(float(imp.importances_mean[i]), 4)) for i in order[:8]])
        except Exception as exc:
            print("permutation importance skipped:", exc)
    else:
        validation_notes.append(f"reranker not fitted: {fit_report.get('status')}")
    mark("validation_fit")
""",
    )

    cell_md(nb, "## 5c. Audit both arms on identical pools and apply the pre-declared promotion rule")
    cell_code(
        nb,
        """
decision = {"selected_arm": "baseline", "promoted": False, "checks": [], "rule": dict(V.PROMOTION_RULE)}
if RUN_VALIDATION:
    audit_q = [q for q in queries if q.partition == "AUDIT"]
    panel_q = [q for q in queries if q.partition == "PANEL"]
    audit_eval = V.evaluate_arms(audit_q, results, reranker, cfg)
    panel_eval = V.evaluate_arms(panel_q, results, reranker, cfg) if panel_q else None
    for name, ev in (("AUDIT", audit_eval), ("PANEL", panel_eval)):
        if ev is None:
            print(f"{name}: no queries")
            continue
        print(f"{name}: coverage={ev['coverage']}")
        for regime, block in ev["regimes"].items():
            print(f"  {regime}: baseline={block['baseline']}")
            print(f"     rerank={block['rerank']}")
            print(f"     paired_delta={block['paired_delta']}")
            if "truth_in_window_only" in block:
                print(f"     truth_in_window_only={block['truth_in_window_only']}")
        print(f"  pooled: baseline_mrr={ev['pooled']['baseline']['mrr']:.4f} rerank_mrr={ev['pooled']['rerank']['mrr']:.4f} delta={ev['pooled']['paired_delta']} changed={ev['pooled']['n_rankings_changed']}")
    decision = V.decide_promotion(audit_eval, panel_eval, reranker_fitted=reranker is not None)
print("DECISION:", decision["selected_arm"])
for c in decision["checks"]:
    print(f"  [{'ok' if c['passed'] else 'FAIL'}] {c['check']}: {c['detail']}")

def _strip_rows(ev):
    if ev is None:
        return None
    return {k: v for k, v in ev.items() if k != "rows"}

validation_results = {
    "run_id": RUN_ID,
    "promotion_rule": V.PROMOTION_RULE,
    "decision": decision,
    "fit_report": {k: v for k, v in fit_report.items() if k != "monotonic"},
    "training_counts": training_counts,
    "natural_coverage": coverage_natural,
    "inserted_truths": inserted_stats,
    "skipped_queries": skipped_queries,
    "panel_instruments": panel_instruments,
    "dev": _strip_rows(dev_eval),
    "audit": _strip_rows(audit_eval),
    "panel": _strip_rows(panel_eval),
    "notes": validation_notes,
    "n_queries_built": len(queries),
    "n_queries_ranked": len(results),
}
V.write_json(RUN_DIR / "validation_results.json", validation_results)
if RUN_VALIDATION:
    rows = []
    for ev in (dev_eval, audit_eval, panel_eval):
        if ev is not None:
            rows.extend(ev["rows"])
    pd.DataFrame(rows).to_parquet(RUN_DIR / "diagnostics" / "per_query.parquet", index=False)
    trace_df = V.trace_table([q for q in queries if q.partition in ("AUDIT", "PANEL")], results, reranker, cfg, top_n=50)
    trace_df.to_parquet(RUN_DIR / "candidate_traces.parquet", index=False)
    print("saved per-query rows", len(rows), "candidate trace rows", len(trace_df))
mark("validation_decision")
""",
    )

    cell_md(nb, "## 6. Rank the test molecules with the selected arm (Class 1/2/3, InChIKey14-deduped top 25)")
    cell_code(
        nb,
        """
from src.ranker import validate_submission, write_submission_csv

out_path = (KAGGLE_WORKING if IS_KAGGLE else cfg.artifact_dir) / "submission.csv"
selected = reranker if decision.get("promoted") else None
print("selected arm:", decision["selected_arm"], "reranker object:", type(selected).__name__ if selected is not None else None)
test_traces = {}
sub = predict_test(
    cfg,
    index=index,
    model=model,
    models=models,
    decoder=None,
    tokenizer=None,
    external_index=ext_prod,
    output_path=out_path,
    reranker=selected,
    trace_out=test_traces,
)
validate_submission(out_path, cfg.sample_submission_path)
print(sub.head())
print("rows", len(sub), "empty smiles", int((sub["smiles"].astype(str).str.strip() == "").sum()))
print("max guesses", sub["smiles"].str.split(";").map(lambda x: len([t for t in x if t])).max())
assert list(sub.columns) == ["molecule_id", "smiles"]
if not FAST_DEV_RUN:
    assert len(sub) == 400, len(sub)
    assert not (sub["smiles"].astype(str).str.strip() == "").any()

# The other arm on the same test pools, for diagnostics only (no labels on test).
test_arms = {}
n_differ = 0
for mid, tr in test_traces.items():
    shipped = list(tr["final_smiles"])
    if selected is not None:
        other = V.baseline_from_trace(tr, cfg, final=shipped)
    elif reranker is not None:
        other = V.apply_reranker_to_trace(tr, reranker, cfg, baseline_final=shipped)
    else:
        other = None
    test_arms[mid] = {"shipped": shipped, "other_arm": other, "n_locked": len(tr.get("locked", {}).get("smiles", [])),
                      "pool_size": int(tr.get("pool_size", 0))}
    if other is not None and other != shipped:
        n_differ += 1
print(f"test molecules where the two arms differ: {n_differ}/{len(test_arms)}")
V.write_json(RUN_DIR / "diagnostics" / "test_arms.json", test_arms)
print("wrote", out_path, "bytes", out_path.stat().st_size)
mark("production_ranking")
""",
    )

    cell_md(nb, "## 7. Artifacts, manifest, strict submission check, inline summary (no cleanup)")
    cell_code(
        nb,
        """
final_path = (KAGGLE_WORKING if IS_KAGGLE else Path(".")) / "submission.csv"
if Path(out_path).resolve() != final_path.resolve():
    final_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(out_path, final_path)
submission_report = V.strict_submission_check(final_path, cfg.sample_submission_path, top_k=cfg.top_k)
print("submission check", submission_report)

manifest = {
    "run_id": RUN_ID,
    "source_commit": SOURCE_COMMIT,
    "source_dirty_at_generation": SOURCE_DIRTY,
    "src_hashes": SRC_HASHES,
    "versions": VERSIONS,
    "inputs": INPUT_FINGERPRINTS,
    "config": V.config_to_jsonable(cfg),
    "fit_subset": {"n": int(fit_take.size), "seed": FIT_SEED, "hash": FIT_HASH},
    "model_config_hash": CONFIG_HASH,
    "training": TRAINING_RECORDS,
    "reranker": {
        "experiment_enabled": bool(RERANKER_EXPERIMENT),
        "window": int(cfg.rerank_window),
        "params": RERANKER_PARAMS,
        "feature_names": list(FEATURE_NAMES),
        "fitted": reranker is not None,
        "artifact": str(RUN_DIR / "reranker.pkl") if reranker is not None else None,
    },
    "decision": decision,
    "selected_arm": decision["selected_arm"],
    "validation_summary": {
        "n_queries_built": len(queries),
        "n_queries_ranked": len(results),
        "audit_pooled": None if audit_eval is None else {k: audit_eval["pooled"][k] for k in ("baseline", "rerank", "paired_delta", "n_rankings_changed")},
        "panel_pooled": None if panel_eval is None else {k: panel_eval["pooled"][k] for k in ("baseline", "rerank", "paired_delta", "n_rankings_changed")},
        "notes": validation_notes,
    },
    "submission": {"path": str(final_path), "bytes": int(final_path.stat().st_size), **submission_report},
    "timings_s": TIMINGS,
    "elapsed_s": round(time.time() - T_START, 1),
}
V.write_json(RUN_DIR / "run_manifest.json", manifest)
zip_path = RUN_DIR / "casmi_diagnostics.zip"
V.zip_directory(RUN_DIR, zip_path, exclude={"casmi_diagnostics.zip"})
print("artifacts:")
for p in sorted(RUN_DIR.rglob("*")):
    if p.is_file():
        print(f"  {p.relative_to(RUN_DIR)}  {p.stat().st_size} bytes")

if IS_KAGGLE:
    root_csvs = sorted(p.name for p in KAGGLE_WORKING.glob("*.csv"))
    print("csv files at /kaggle/working root:", root_csvs)
    assert root_csvs == ["submission.csv"], root_csvs

summary = {
    "run_id": RUN_ID,
    "selected_arm": decision["selected_arm"],
    "promoted": bool(decision.get("promoted")),
    "checks": {c["check"]: c["passed"] for c in decision["checks"]},
    "audit_baseline_mrr": None if audit_eval is None else round(audit_eval["pooled"]["baseline"]["mrr"], 4),
    "audit_rerank_mrr": None if audit_eval is None else round(audit_eval["pooled"]["rerank"]["mrr"], 4),
    "audit_delta_ci": None if audit_eval is None else [round(audit_eval["pooled"]["paired_delta"]["ci_lo"], 4), round(audit_eval["pooled"]["paired_delta"]["ci_hi"], 4)],
    "panel_n": None if panel_eval is None else panel_eval["pooled"]["baseline"]["n"],
    "n_queries_ranked": len(results),
    "reranker_status": fit_report.get("status"),
    "versions": {k: VERSIONS[k] for k in ("rdkit", "sklearn", "torch")},
    "elapsed_s": round(time.time() - T_START, 1),
    "submission_bytes": int(final_path.stat().st_size),
    "artifacts_dir": str(RUN_DIR),
}
print("SUMMARY_JSON " + json.dumps(summary, default=str))
print("done")
""",
    )

    for name in NOTEBOOK_NAMES:
        out = ROOT / name
        nbf.write(nb, out)
        nbytes = out.stat().st_size
        print("wrote", out, "cells", len(nb.cells), "bytes", nbytes)
        if nbytes >= 1_000_000:
            raise SystemExit(
                f"{out.name} is {nbytes} bytes; Kaggle kernel source must be < 1 MB. "
                "Do not embed Class 2 inside the notebook."
            )

    parquet = ROOT / "data" / "class2_candidates.parquet"
    ds = ROOT / "kaggle_dataset"
    ds.mkdir(parents=True, exist_ok=True)
    if parquet.exists():
        dest = ds / "class2_candidates.parquet"
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        try:
            dest.hardlink_to(parquet)
        except OSError:
            shutil.copy2(parquet, dest)
        print("dataset file", dest, "bytes", dest.stat().st_size)
    meta = {
        "title": "CASMI Class 2 COCONUT LOTUS candidates",
        "subtitle": "COCONUT 2.0 ∪ LOTUS natural-product structures for Enveda CASMI 2026 (not filtered on public test masses)",
        "licenses": [{"name": "CC0-1.0"}],
        "keywords": ["chemistry", "mass spectrometry", "natural products"],
    }
    (ds / "dataset-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
