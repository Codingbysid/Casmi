#!/usr/bin/env python3
"""Generate the standalone Kaggle CPU/GPU notebook from the src/ package."""

from __future__ import annotations

import json
import shutil
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
    "src/formula_head.py",
    "src/metrics.py",
    "src/tpu_trainer.py",
    "src/infer.py",
    "src/train_tpu.py",
]


def cell_md(nb, text: str) -> None:
    nb.cells.append(nbf.v4.new_markdown_cell(text))


def cell_code(nb, text: str) -> None:
    nb.cells.append(nbf.v4.new_code_cell(text.strip() + "\n"))


def main() -> None:
    files = {}
    for rel in SRC_FILES:
        files[rel] = (ROOT / rel).read_text()

    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    # CPU session by default (TPU queue is often full). GPU is used if the session has one.
    nb.metadata["accelerator"] = "none"
    nb.metadata["kaggle"] = {
        "accelerator": "none",
        "isInternetEnabled": False,
        "language": "python",
        "sourceType": "notebook",
    }

    cell_md(
        nb,
        """# Enveda CASMI 2026 — Molecule ID from Mass Spectra (CPU / GPU)

Configured for **Kaggle Run All** with **internet off**. No TPU required.

Attach **three** inputs (Kaggle kernels must stay under **1 MB**, so the NP pool and RDKit wheels cannot live inside the notebook):

1. Competition data: `train.parquet`, `test.parquet`, `sample_submission.csv`
2. Class 2 COCONUT ∪ LOTUS table: attach the dataset that contains `class2_candidates.parquet` (~61 MB, ~453k structures). Kaggle may name the dataset folder `class2_candidates.parquet2`.
3. **RDKit wheel (required, ~35 MB):** do **not** use the current 20 KB stub named `rdkit-2025-9-5`. Upload `kaggle_rdkit_wheels/rdkit-2025.9.5-cp312-cp312-manylinux_2_28_x86_64.whl` as a new dataset, then Add Input.

Internet stays **OFF**. The first code cell installs RDKit from that wheelhouse. You must see `rdkit 2025...` printed, not `WARNING: RDKit missing`.

That parquet is the **full** NP set (train skeletons removed, generic 50–2000 Da). It is **not** filtered on the public `test.parquet` masses.

Pipeline:
1. Build a train-structure library (InChIKey14, exact mass, Morgan fingerprints, representative MS2).
2. Train up to **3 Spec2FP seeds** on 120k unique train skeletons (4 epochs, ~1.5 h each) and average pre-sigmoid logits. Blend with train spectral neighbors (k=20, a=0.5).
3. Rank: lock Class 1 if modified cosine ≥ 0.75. Remaining slots merge weak train hits, Class 2 COCONUT, and mass-shifted analogs by `0.42*spec + 0.46*tani + 0.12*mass`.
4. Fill leftover slots from attached COCONUT ∪ LOTUS (`class2_candidates.parquet`, mass filter at query time). Analog parents are retrieved at ±sugar/CH2/O/acetyl mass shifts.
5. Write **exactly 25** unique InChIKey14 guesses.

Output: `/kaggle/working/submission.csv`.
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
        """
# ===== Fully configured for Run All on Kaggle CPU (9 h) =====
# Session: CPU (or GPU). Do not start a TPU session.
# Internet: OFF.
# Add Data:
#   1) Enveda CASMI 2026 competition
#   2) class2_candidates.parquet (folder may be class2_candidates.parquet2)
#   3) rdkit-cp312-wheel (~35 MB .whl, not the 20 KB stub)
REQUIRE_CLASS2 = True         # fail fast if class2_candidates.parquet is not attached
FAST_DEV_RUN = False          # True = 8k-row debug pass; keep False for a real submit
PREFER_XLA = False            # TPU off: CPU/GPU path
TRAIN_DECODER = False         # Class 3 analog shifts run without the SMILES decoder
ENSEMBLE_SEEDS = 3            # average pre-sigmoid Spec2FP logits
MAX_TRAIN_SPECTRA = 120_000   # V5/V6 0.139; 280k/6ep scored 0.126
NUM_EPOCHS = 4
BATCH_SIZE = 64
TRAIN_TIME_LIMIT_S = 4.5 * 3600.0
USE_NEIGHBOR_FP = True
NEIGHBOR_BLEND = 0.5
RANK_RESERVE_S = 2.0 * 3600.0
PER_SEED_MAX_S = 1.5 * 3600.0
DECODER_TIME_LIMIT_S = 0.6 * 3600.0
DECODER_MAX_SMILES = 50_000
TOP_N_PEAKS = 128
N_MZ_BINS = 512
FP_BITS = 2048
SESSION_BUDGET_S = 8.5 * 3600.0
""",
    )

    cell_code(
        nb,
        """
import os, sys, json, time, math, traceback
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

print("ROOT", ROOT, "kaggle", IS_KAGGLE)
print("python", sys.version)
print("cpu_count", os.cpu_count())
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
for rel, content in FILES.items():
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
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
    print("rdkit", Chem.rdBase.rdkitVersion)
    HAS_RDKIT = True
    _probe = Chem.MolFromSmiles("CCO")
    assert _probe is not None, "RDKit imported but MolFromSmiles('CCO') returned None"
    assert HAS_RDKIT is True
except Exception as e:
    print("WARNING: RDKit missing — fingerprint training disabled, spectral+mass only.", e)
    HAS_RDKIT = False
    raise ImportError(
        "HAS_RDKIT is False. The first cell must pip-install the attached "
        "~35 MB rdkit .whl. Attaching a 20 KB stub does not install the package."
    ) from e

try:
    import torch_xla.core.xla_model as xm  # noqa: F401
    HAS_XLA = True
    print("torch_xla present (not used unless PREFER_XLA=True)")
except Exception:
    HAS_XLA = False
    print("torch_xla not available (expected on CPU/GPU sessions)")
""",
    )

    cell_md(nb, "## 2. Config, library, train, infer")
    cell_code(
        nb,
        """
from src.config import get_config
from src.chem import has_rdkit, inchikey14_from_smiles
from src.data import dataset_from_dataframe, dataset_from_structure_index, load_test, load_train_slice
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
from src.tpu_trainer import get_device, train_spec2fp

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

# CPU-sized model; GPU sessions get a slightly larger batch automatically after device pick.
cfg = get_config(
    batch_size=BATCH_SIZE,
    num_epochs=NUM_EPOCHS,
    max_train_spectra=MAX_TRAIN_SPECTRA,
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
    smiles_train_time_limit_s=DECODER_TIME_LIMIT_S,
    smiles_num_samples=48,
    smiles_gen_max_len=80,
    smiles_mass_ppm=50.0,
    num_workers=0,
    max_mass_candidates=8192,
    use_neighbor_fp=bool(USE_NEIGHBOR_FP),
    neighbor_blend=float(NEIGHBOR_BLEND),
    ensemble_seeds=int(ENSEMBLE_SEEDS),
)
cfg.data_dir = data_dir
cfg.artifact_dir = Path("/tmp/casmi_artifacts") if IS_KAGGLE else (ROOT / "artifacts")
cfg.artifact_dir.mkdir(parents=True, exist_ok=True)
cfg.external_candidates_path = CLASS2_PATH
print("class2", cfg.external_candidates_path, "exists", CLASS2_PATH is not None and Path(CLASS2_PATH).exists())

print("train", cfg.train_path.exists(), cfg.train_path)
print("test", cfg.test_path.exists(), cfg.test_path)
assert cfg.train_path.exists() and cfg.test_path.exists(), "Attach the competition dataset."
""",
    )

    cell_md(nb, "## 3. Build candidate library from train (mass + fingerprints + representative MS2)")
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
print("elapsed_s", round(time.time() - T_START, 1))
""",
    )

    cell_md(nb, "## 4. Train Spec2FP on CPU/GPU from the library")
    cell_code(
        nb,
        """
device, use_xla = get_device(prefer_xla=bool(PREFER_XLA))
if str(device).startswith("cuda"):
    cfg.batch_size = max(cfg.batch_size, 128)
    cfg.d_model = 256
    cfg.n_heads = 8
    cfg.n_transformer_layers = 3
print("train device", device, "xla", use_xla)

if FAST_DEV_RUN:
    cfg.num_epochs = 1
    cfg.train_time_limit_s = 180.0
    cfg.max_train_spectra = 4_000
    cfg.log_every = 20

t_data = time.time()
n_fit = 4_000 if FAST_DEV_RUN else int(cfg.max_train_spectra)
print(
    f"building Spec2FP dataset from library n={len(index.smiles)} cap={n_fit} "
    f"neighbor_fp={cfg.use_neighbor_fp} blend={cfg.neighbor_blend}"
)
ds = dataset_from_structure_index(index, cfg, max_n=n_fit, seed=cfg.seed)
probe = ds[0]
print("example tensor shapes:", {k: tuple(v.shape) for k, v in probe.items()})
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

t_tr = time.time()
models = []
n_seeds = 1 if FAST_DEV_RUN else int(ENSEMBLE_SEEDS)
seed_list = [42, 7, 123, 99, 2024][:n_seeds]
if HAS_RDKIT:
    for si, seed in enumerate(seed_list):
        seeds_left = n_seeds - si
        left = SESSION_BUDGET_S - (time.time() - T_START)
        budget = min(float(PER_SEED_MAX_S), max(0.0, (left - float(RANK_RESERVE_S)) / max(seeds_left, 1)))
        budget = min(budget, float(TRAIN_TIME_LIMIT_S))
        if FAST_DEV_RUN:
            budget = min(budget, 180.0)
        if budget < 8 * 60 and si > 0:
            print(f"skipping remaining seeds; budget_s={budget:.0f}")
            break
        cfg.seed = int(seed)
        cfg.ckpt_name = f"spec2fp_s{seed}.pt"
        cfg.train_time_limit_s = float(budget)
        print(f"ensemble seed={seed} time_limit_s={budget:.0f} epochs={cfg.num_epochs} ckpt={cfg.ckpt_name}")
        try:
            m = train_spec2fp(ds, cfg, device=device, seed=int(seed), ckpt_name=cfg.ckpt_name, time_limit_s=budget)
            models.append(m.to("cpu").eval())
        except Exception:
            traceback.print_exc()
            print(f"Training failed for seed={seed}")
    if not models:
        print("Training failed — continuing with spectral + mass ranking.")
else:
    print("Skipping neural training (no RDKit).")
model = models[0] if models else None
print("ensemble models", len(models))
print(f"train wall {time.time()-t_tr:.1f}s")
del ds
""",
    )

    cell_md(nb, "## 4b. Train prefix-conditioned SMILES decoder (Class 3)")
    cell_code(
        nb,
        """
from src.smiles_tokenizer import SmilesTokenizer
from src.smiles_train import dataset_from_structures, train_smiles_decoder
from src.models.smiles_decoder import count_parameters, decoder_from_config

decoder = None
smiles_tokenizer = None
remaining = SESSION_BUDGET_S - (time.time() - T_START)
do_decoder = bool(TRAIN_DECODER) and HAS_RDKIT and remaining > 15 * 60
print("decoder remaining_s", round(remaining, 1), "do_decoder", do_decoder)
if do_decoder:
    try:
        smiles_tokenizer = SmilesTokenizer.default()
        smiles_tokenizer.save(cfg.smiles_vocab_path)
        if FAST_DEV_RUN:
            cfg.smiles_num_epochs = 1
            cfg.smiles_train_time_limit_s = 120.0
        cap = 2_000 if FAST_DEV_RUN else int(DECODER_MAX_SMILES)
        n_smi = min(len(index.smiles), cap)
        rng = np.random.default_rng(cfg.seed)
        pick = (
            np.sort(rng.choice(len(index.smiles), size=n_smi, replace=False))
            if len(index.smiles) > n_smi
            else np.arange(len(index.smiles))
        )
        smi_ds = dataset_from_structures(
            index.smiles[pick].tolist(),
            index.exact_mass[pick],
            index.fingerprints[pick].astype(np.float32),
            smiles_tokenizer,
            cfg,
        )
        probe = smi_ds[0]
        assert tuple(probe["tgt_tokens"].shape) == (cfg.smiles_max_len,), probe["tgt_tokens"].shape
        n_params = count_parameters(decoder_from_config(smiles_tokenizer, cfg))
        print("smiles vocab", smiles_tokenizer.vocab_size, "n", len(smi_ds), "params", n_params)
        print("decoder train n", len(smi_ds), "device", "cpu")
        decoder = train_smiles_decoder(smi_ds, smiles_tokenizer, cfg, device=torch.device("cpu"))
        decoder = decoder.to("cpu").eval()
        del smi_ds
        print("decoder ckpt", cfg.smiles_checkpoint_path.exists(), cfg.smiles_checkpoint_path)
    except Exception:
        traceback.print_exc()
        print("SMILES decoder training failed or skipped — Class 3 fallback disabled.")
        decoder = None
        smiles_tokenizer = smiles_tokenizer
else:
    print("Skipping SMILES decoder (flag/time/RDKit).")
""",
    )
    cell_md(nb, "## 5. Rank test molecules (Class 1/2/3, InChIKey14-deduped top 25)")
    cell_code(
        nb,
        """
from src.ranker import validate_submission, write_submission_csv

out_path = (KAGGLE_WORKING if IS_KAGGLE else cfg.artifact_dir) / "submission.csv"
# If the session is almost out of time, skip expensive Class 3 sampling.
remaining = SESSION_BUDGET_S - (time.time() - T_START)
if remaining < 20 * 60:
    print("low time remaining — ranking without Class 3 decoder")
    decoder = None
sub = predict_test(
    cfg,
    index=index,
    model=model,
    models=models,
    decoder=decoder,
    tokenizer=smiles_tokenizer,
    output_path=out_path,
)
validate_submission(out_path, cfg.sample_submission_path)
print(sub.head())
print("rows", len(sub), "empty smiles", int((sub["smiles"].astype(str).str.strip() == "").sum()))
print("max guesses", sub["smiles"].str.split(";").map(lambda x: len([t for t in x if t])).max())
assert list(sub.columns) == ["molecule_id", "smiles"]
if not FAST_DEV_RUN:
    assert len(sub) == 400, len(sub)
    assert not (sub["smiles"].astype(str).str.strip() == "").any()
print("wrote", out_path, "bytes", out_path.stat().st_size)
print("elapsed_s", round(time.time() - T_START, 1))
""",
    )

    cell_md(nb, "## 6. Optional: MRR@25 on a labeled train holdout (does not affect submission)")
    cell_code(
        nb,
        """
try:
    hold = load_train_slice(cfg, 2_000 if not FAST_DEV_RUN else 400)
    q = hold.iloc[-80:].reset_index(drop=True)
    from src.preprocessing import featurize_spectrum

    q_feats = [
        featurize_spectrum(
            q.iloc[i]["ms2_mzs"],
            q.iloc[i]["ms2_normalized_intensities"],
            float(q.iloc[i]["precursor_mz"]),
            q.iloc[i]["adduct"],
            cfg=cfg,
            collision_energy=q.iloc[i]["collision_energy_ev"],
            ionization_mode=q.iloc[i]["ionization_mode"],
        )
        for i in range(len(q))
    ]
    q_fps = predict_fingerprints(model, q_feats, cfg)
    mol_feats = {f"h{i}": q_feats[i] for i in range(len(q_feats))}
    pred_map = {f"h{i}": q_fps[i] for i in range(len(q_feats))}
    ranked = rank_molecules(mol_feats, pred_map, index, cfg)
    pred_lists = [ranked[f"h{i}"] for i in range(len(q_feats))]
    score = mrr_at_k(pred_lists, true_keys=q["inchikey14"].astype(str).tolist(), k=25)
    print(f"holdout MRR@25 (train skeletons in library): {score:.4f}")
except Exception:
    traceback.print_exc()
    print("holdout MRR skipped (does not affect submission.csv)")
print("done  elapsed_s", round(time.time() - T_START, 1))
""",
    )

    cell_md(nb, "## 7. Leave only `submission.csv` in `/kaggle/working`")
    cell_code(
        nb,
        """
import shutil

final_path = (KAGGLE_WORKING if IS_KAGGLE else Path(".")) / "submission.csv"
if Path(out_path).resolve() != final_path.resolve():
    final_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(out_path, final_path)
write_submission_csv(sub, final_path)
validate_submission(final_path, cfg.sample_submission_path)

if IS_KAGGLE:
    keep = {"submission.csv"}
    for p in Path("/kaggle/working").iterdir():
        if p.name in keep or p.name.startswith("."):
            continue
        try:
            if p.is_dir():
                shutil.rmtree(p, ignore_errors=True)
            else:
                p.unlink()
        except Exception as exc:
            print("could not remove", p, exc)
    leftover = sorted(x.name for x in Path("/kaggle/working").iterdir())
    csvs = [x.name for x in Path("/kaggle/working").glob("*.csv")]
    print("kaggle/working leftover", leftover, "csvs", csvs)
    # Extra CSVs are what confuse the scorer; other Kaggle system files may remain.
    assert csvs == ["submission.csv"], csvs

print("final submission", final_path, "bytes", final_path.stat().st_size)
print("done")
""",
    )

    for name in ("kaggle_submission.ipynb", "kaggle_submission_tpu.ipynb"):
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
