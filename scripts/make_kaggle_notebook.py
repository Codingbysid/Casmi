#!/usr/bin/env python3
"""Generate the Kaggle notebooks from src/.

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
    "src/pipeline.py",
    "src/formula_head.py",
    "src/metrics.py",
    "src/tpu_trainer.py",
    "src/infer.py",
    "src/train_tpu.py",
]
TRAIN_NOTEBOOKS = ("kaggle_submission.ipynb", "kaggle_gpu_submission.ipynb", "kaggle_submission_tpu.ipynb")
PREDICT_NOTEBOOK = "kaggle_predict_only.ipynb"
RDKIT_CELL = '# Offline RDKit install — MUST be the first code cell (internet OFF).\n# Need a ~35 MB .whl, not the 20 KB stub dataset named rdkit-2025-9-5.\nimport os, sys, shutil, subprocess\nfrom importlib import invalidate_caches\nfrom pathlib import Path\n\ndef _rdkit_ok():\n    try:\n        from rdkit import Chem\n        ver = getattr(getattr(Chem, "rdBase", Chem), "rdkitVersion", None) or "unknown"\n        mol = Chem.MolFromSmiles("CCO")\n        return bool(mol is not None), str(ver)\n    except Exception:\n        return False, ""\n\ndef _looks_like_wheel(path: Path) -> bool:\n    name = path.name.lower()\n    if name.endswith(".whl"):\n        return True\n    if "rdkit" in name and any(x in name for x in ("cp31", "manylinux", "win", "macosx")):\n        return True\n    try:\n        with path.open("rb") as fh:\n            magic = fh.read(4)\n        # ZIP / wheel local file header\n        if magic[:2] == b"PK":\n            return "rdkit" in name or path.stat().st_size > 1_000_000\n    except Exception:\n        return False\n    return False\n\ndef find_rdkit_wheels() -> list[Path]:\n    inp = Path("/kaggle/input")\n    tag = f"cp{sys.version_info.major}{sys.version_info.minor}"\n    print("listing /kaggle/input ...")\n    if inp.exists():\n        for p in sorted(inp.rglob("*")):\n            kind = "dir" if p.is_dir() else "file"\n            size = p.stat().st_size if p.is_file() else 0\n            print(f"  {kind} {p} bytes={size}")\n    small: list[tuple[Path, int]] = []\n    hits: list[Path] = []\n    if not inp.exists():\n        return hits\n    for p in inp.rglob("*"):\n        if not p.is_file():\n            continue\n        parent = str(p.parent).lower()\n        if "rdkit" not in parent and "rdkit" not in p.name.lower():\n            continue\n        size = p.stat().st_size\n        if size < 5_000_000:\n            small.append((p, size))\n            continue\n        if _looks_like_wheel(p) or "rdkit" in p.name.lower() or p.suffix.lower() == ".whl":\n            hits.append(p)\n    if small:\n        print("rdkit-named files too small to be a wheel (need ~35 MB):")\n        for p, size in small:\n            print(f"  {p} bytes={size}")\n    uniq: list[Path] = []\n    seen: set[str] = set()\n    for p in hits:\n        key = str(p.resolve())\n        if key in seen:\n            continue\n        seen.add(key)\n        uniq.append(p)\n    uniq.sort(\n        key=lambda p: (\n            0 if p.suffix.lower() == ".whl" else 1,\n            0 if tag in p.name else 1,\n            0 if "rdkit" in p.name.lower() else 1,\n            -p.stat().st_size,\n        )\n    )\n    return uniq\n\nRDKIT_WHEEL_PATH = None\nok, ver = _rdkit_ok()\nif ok:\n    print("rdkit already installed", ver)\nelse:\n    print("python", sys.version)\n    wheels = find_rdkit_wheels()\n    print("rdkit wheels found", [str(w) for w in wheels])\n    if not wheels:\n        raise FileNotFoundError(\n            "No ~35 MB RDKit wheel under /kaggle/input. "\n            "The attached \'rdkit-2025-9-5\' file is only ~20 KB — that is not the wheel. "\n            "On your laptop, upload this file as a NEW Kaggle dataset (do not reuse the stub): "\n            "kaggle_rdkit_wheels/rdkit-2025.9.5-cp312-cp312-manylinux_2_28_x86_64.whl "\n            "(~35 MB). Then Add Input that dataset. Keep internet OFF."\n        )\n    src = wheels[0]\n    RDKIT_WHEEL_PATH = str(src)\n    work = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("/tmp")\n    dest_dir = work / "_rdkit_wheels"\n    dest_dir.mkdir(parents=True, exist_ok=True)\n    py = f"cp{sys.version_info.major}{sys.version_info.minor}"\n    dest = dest_dir / f"rdkit-2025.9.5-{py}-{py}-manylinux_2_28_x86_64.whl"\n    print("copying", src, "->", dest, "src_size", src.stat().st_size)\n    shutil.copy2(src, dest)\n    cmd = [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", str(dest)]\n    print("install", " ".join(cmd))\n    subprocess.check_call(cmd)\n    invalidate_caches()\n    ok, ver = _rdkit_ok()\n    if not ok:\n        subprocess.check_call(\n            [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",\n             f"--find-links={dest_dir}", "rdkit"]\n        )\n        invalidate_caches()\n        ok, ver = _rdkit_ok()\n    if not ok:\n        raise ImportError(f"pip installed {src} but `from rdkit import Chem` still fails")\n    print("rdkit", ver)\nassert _rdkit_ok()[0], "HAS_RDKIT must be True after the offline wheel install"\nprint("HAS_RDKIT", True, "rdkit", _rdkit_ok()[1])\n'



def cell_md(nb, text: str) -> None:
    nb.cells.append(nbf.v4.new_markdown_cell(text))


def cell_code(nb, text: str) -> None:
    nb.cells.append(nbf.v4.new_code_cell(text.strip() + "\n"))


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def _common(nb, files, source_commit, source_dirty) -> None:
    cell_code(nb, RDKIT_CELL)
    cell_code(
        nb,
        f"""
# GPU session, internet OFF, Python 3.12.
# The frozen recipe is src/pipeline.py RECIPE. This notebook does not shrink it.
REQUIRE_CLASS2 = True
REQUIRE_CUDA = True
SOURCE_COMMIT = {source_commit!r}
SOURCE_DIRTY = {source_dirty!r}
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
""",
    )
    payload = repr(json.dumps(files))
    cell_code(
        nb,
        f"""
import hashlib, json, sys
from pathlib import Path
ROOT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".").resolve()
(ROOT / "src" / "models").mkdir(parents=True, exist_ok=True)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
FILES = json.loads({payload})
SRC_HASHES = {{}}
for rel, content in FILES.items():
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    SRC_HASHES[rel] = hashlib.sha256(content.encode("utf-8")).hexdigest()
    print("wrote", rel, "bytes", len(content))
""",
    )


def _blank():
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    nb.metadata["language_info"] = {"name": "python", "pygments_lexer": "ipython3"}
    nb.metadata["accelerator"] = "GPU"
    nb.metadata["kaggle"] = {
        "accelerator": "gpu",
        "isInternetEnabled": False,
        "isGpuEnabled": True,
        "language": "python",
        "sourceType": "notebook",
    }
    return nb


def main() -> None:
    source_commit = _git("rev-parse", "--short", "HEAD")
    source_dirty = bool(_git("status", "--porcelain", "--", "src", "scripts"))
    files = {rel: (ROOT / rel).read_text() for rel in SRC_FILES}

    train = _blank()
    cell_md(
        train,
        """# Enveda CASMI 2026 — required reranker (train and predict)

**Kaggle GPU, internet off, Run All.** Strategy `required-reranker-v1`.

Attach:

1. Competition data (`train.parquet`, `test.parquet`, `sample_submission.csv`)
2. `class2_candidates.parquet` — the 452,608-row COCONUT ∪ LOTUS pool, not filtered on public test masses
3. The ~35 MB RDKit wheel (`rdkit-2025.9.5-cp312`), not the 20 KB stub

Optional: a previous `casmi_run/` with complete checkpoints, library cache, neighbor cache, and reranker. Reuse happens only when fit-subset hash, model config, source hash, dependency versions, seed, and four completed epochs all match.

This run always fits the current 22-feature HistGradientBoosting reranker on RR_TRAIN (regimes U and K only) and ranks the test set with it. It does not score DEV, AUDIT, or PANEL, and it does not fall back to the 0.150 linear order. A failed fit or an incomplete seed raises.

`submission.csv` is moved aside at the start and published atomically after checks. Artifacts stay in `/kaggle/working/casmi_run/`. A deleted Kaggle session deletes that directory; download it and re-upload it as a dataset to use the predict-only notebook.

Inactive experiments (not applied): `precursor-mz-shift-v1`, `official-inchikey14-v14`.

Keep the 0.150 submission selected until this public score is strictly higher. Do not treat an unmeasured number as 0.350.
""",
    )
    _common(train, files, source_commit, source_dirty)
    cell_code(
        train,
        """
from pathlib import Path
from src.pipeline import STRATEGY_ID, prepare_kaggle_config, run_train_and_predict
print("strategy", STRATEGY_ID)
cfg = prepare_kaggle_config(require_class2=REQUIRE_CLASS2)
run_dir = Path("/kaggle/working/casmi_run") if Path("/kaggle/working").exists() else Path("artifacts/casmi_run")
run_train_and_predict(cfg, run_dir=run_dir, source_commit=SOURCE_COMMIT, require_cuda=REQUIRE_CUDA)
""",
    )

    predict = _blank()
    cell_md(
        predict,
        """# Enveda CASMI 2026 — predict only

**Kaggle GPU, internet off, Run All.** Loads a verified `required-reranker-v1` bundle and writes `submission.csv`. It does not train, and it will not start training if the bundle is missing or incompatible.

Attach:

1. Competition data (test molecules and `sample_submission.csv`; `train.parquet` is checked against the library-cache hash)
2. The same 452,608-row Class 2 pool
3. The ~35 MB RDKit wheel
4. The `casmi_run/` directory from a finished train-and-predict output, uploaded as its own Kaggle dataset

`casmi_run/` must contain `run_manifest.json`, `checkpoints/spec2fp_s{42,7,123}.pt`, `reranker.pkl`, `library.npz` + `library.json`, and `neighbor.npz` + `neighbor.json`.

How to attach it: on the finished train run, open Output and download `casmi_run` (or create a dataset from that output). Add that dataset as input. If several manifests are attached, set `BUNDLE_DIR` below to the directory that contains `run_manifest.json`.

Files left in `/kaggle/working` disappear when the session is deleted. A laptop path is not available on Kaggle. Only the dataset you attach is durable.

`BUNDLE_DIR = ""` searches `/kaggle/input` for exactly one manifest.
""",
    )
    _common(predict, files, source_commit, source_dirty)
    cell_code(
        predict,
        """
BUNDLE_DIR = ""  # directory containing run_manifest.json; empty searches /kaggle/input
from src.pipeline import prepare_kaggle_config, run_predict_only
cfg = prepare_kaggle_config(require_class2=REQUIRE_CLASS2)
run_predict_only(cfg, bundle_dir=BUNDLE_DIR or None)
""",
    )

    for name in list(TRAIN_NOTEBOOKS) + [PREDICT_NOTEBOOK]:
        nb = train if name in TRAIN_NOTEBOOKS else predict
        path = ROOT / name
        nbf.write(nb, path)
        nbytes = path.stat().st_size
        print("wrote", path.name, "cells", len(nb.cells), "bytes", nbytes)
        if nbytes >= 1_000_000:
            raise SystemExit(f"{path.name} is {nbytes} bytes; Kaggle kernel source must be < 1 MB")

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
    meta = {
        "title": "CASMI Class 2 COCONUT LOTUS candidates",
        "subtitle": "COCONUT 2.0 ∪ LOTUS natural-product structures for Enveda CASMI 2026 (not filtered on public test masses)",
        "licenses": [{"name": "CC0-1.0"}],
        "keywords": ["chemistry", "mass spectrometry", "natural products"],
    }
    (ds / "dataset-metadata.json").write_text(json.dumps(meta, indent=2) + "\n")


if __name__ == "__main__":
    main()
