#!/usr/bin/env python3
"""Build a Class 2 NP library from COCONUT 2.0 ∪ LOTUS.

CRITICAL: do not read test.parquet. The public test file is a train placeholder;
Kaggle replaces it on re-run. Filtering the pool to those precursor masses would
yield ~0 Class 2 hits on the real 400 unknowns.

Keep every unique skeleton (RDKit InChIKey14) that is not already in train and
whose monoisotopic mass sits in a generic LC-MS window (50–2000 Da).
"""

from __future__ import annotations

import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.chem import (  # noqa: E402
    exact_mass_from_smiles,
    formula_to_mass,
    inchikey14_from_smiles,
    morgan_fingerprint,
    pack_fingerprints,
    parse_formula,
)
from src.config import get_config  # noqa: E402

MW_MIN = 50.0
MW_MAX = 2000.0

COCONUT_URLS = (
    "https://coconut.s3.uni-jena.de/prod/downloads/2026-08/coconut_csv-08-2026.zip",
    "https://coconut.s3.uni-jena.de/prod/downloads/2026-09/coconut-lite.csv.zip",
    "https://coconut.s3.uni-jena.de/prod/downloads/2025-03/coconut_csv-03-2025.zip",
    "https://zenodo.org/records/13382751/files/coconut-08-2024.csv.zip?download=1",
)


def _inchi_formula(inchi: str) -> str:
    if not isinstance(inchi, str) or not inchi.startswith("InChI"):
        return ""
    parts = inchi.split("/")
    return parts[1] if len(parts) > 1 else ""


def _ikey14(value: object) -> str:
    s = str(value or "").strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return ""
    return s.split("-")[0][:14]


def _train_keys(cfg) -> set[str]:
    if not cfg.train_path.exists():
        return set()
    keys = pd.read_parquet(cfg.train_path, columns=["inchikey14"])["inchikey14"]
    out = set(keys.dropna().astype(str).str.slice(0, 14).unique())
    print(f"[class2] train unique InChIKey14={len(out)}")
    return out


def _download_coconut(dest_zip: Path) -> Path:
    if dest_zip.exists() and dest_zip.stat().st_size > 1_000_000:
        print(f"[class2] using existing {dest_zip} ({dest_zip.stat().st_size} bytes)")
        return dest_zip
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    import urllib.request

    last_err = None
    for url in COCONUT_URLS:
        print(f"[class2] downloading {url}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "casmi-v6/1.0"})
            with urllib.request.urlopen(req, timeout=120) as resp, dest_zip.open("wb") as fh:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    fh.write(chunk)
            print(f"[class2] saved {dest_zip} ({dest_zip.stat().st_size} bytes)")
            return dest_zip
        except Exception as exc:
            last_err = exc
            print(f"[class2] download failed: {exc}")
            if dest_zip.exists():
                dest_zip.unlink()
    raise RuntimeError(f"could not download COCONUT CSV: {last_err}")


def _extract_csv(zpath: Path, out_dir: Path) -> Path:
    with zipfile.ZipFile(zpath) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv") and not n.endswith("/")]
        if not names:
            names = zf.namelist()
        # Prefer lite / coconut csv over fragments.
        names.sort(key=lambda n: (0 if "lite" in n.lower() or "coconut" in n.lower() else 1, len(n)))
        member = names[0]
        target = out_dir / Path(member).name
        if not target.exists() or target.stat().st_size < 1_000_000:
            print(f"[class2] extracting {member} -> {target}")
            with zf.open(member) as src, target.open("wb") as dst:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)
        return target


def _read_coconut(path: Path) -> pd.DataFrame:
    peek = pd.read_csv(path, nrows=2, low_memory=False)
    cols = {c.lower(): c for c in peek.columns}
    smi = cols.get("canonical_smiles") or cols.get("canonicalsmiles") or cols.get("smiles")
    key = cols.get("standard_inchi_key") or cols.get("standard_inchikey") or cols.get("inchikey")
    formula = cols.get("molecular_formula") or cols.get("formula") or cols.get("molecularformula")
    mass_col = cols.get("exact_molecular_weight") or cols.get("molecular_weight")
    usecols = [c for c in (smi, key, formula, mass_col) if c]
    print(f"[class2] COCONUT columns smiles={smi} key={key} formula={formula} mass={mass_col}")
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    rename = {smi: "smiles"}
    if key:
        rename[key] = "inchikey"
    if formula:
        rename[formula] = "formula"
    if mass_col:
        rename[mass_col] = "reported_mass"
    df = df.rename(columns=rename)
    if "formula" not in df.columns:
        df["formula"] = ""
    if "inchikey" not in df.columns:
        df["inchikey"] = ""
    if "reported_mass" not in df.columns:
        df["reported_mass"] = np.nan
    df["smiles"] = df["smiles"].astype(str)
    df["inchikey14"] = df["inchikey"].map(_ikey14)
    df["formula"] = df["formula"].astype(str)
    df["source"] = "coconut"
    return df[["smiles", "inchikey14", "formula", "reported_mass", "source"]]


def _read_lotus(path: Path) -> pd.DataFrame:
    usecols = ["canonicalSmiles", "isomericSmiles", "inchi", "inchiKey"]
    lotus = pd.read_csv(path, sep="\t", usecols=usecols, low_memory=False)
    print(f"[class2] LOTUS rows={len(lotus)}")
    rows = []
    seen: set[str] = set()
    for rec in lotus.itertuples(index=False):
        key = _ikey14(getattr(rec, "inchiKey"))
        if not key or key in seen:
            continue
        iso = str(getattr(rec, "isomericSmiles") or "")
        can = str(getattr(rec, "canonicalSmiles") or "")
        smi = iso if iso and iso not in {"nan", "None"} else can
        if not smi or smi in {"nan", "None"}:
            continue
        seen.add(key)
        rows.append(
            {
                "smiles": smi,
                "inchikey14": key,
                "formula": _inchi_formula(str(getattr(rec, "inchi") or "")),
                "reported_mass": np.nan,
                "source": "lotus",
            }
        )
    return pd.DataFrame(rows)


def _keep_row(smi: str, key: str, formula: str, train_keys: set[str]) -> tuple[str, str, str, float] | None:
    if key in train_keys:
        return None
    rd_key = inchikey14_from_smiles(smi)
    if not rd_key or rd_key in train_keys:
        return None
    rd_mass = exact_mass_from_smiles(smi)
    if rd_mass < MW_MIN or rd_mass > MW_MAX:
        return None
    if formula and not parse_formula(formula):
        formula = ""
    return smi, rd_key, formula, float(rd_mass)


def main() -> None:
    cfg = get_config()
    train_keys = _train_keys(cfg)
    work = Path(os.environ.get("CASMI_CLASS2_TMP", "/tmp/casmi_coconut"))
    work.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    coconut_zip = work / "coconut.csv.zip"
    coconut_csv = None
    try:
        zpath = _download_coconut(coconut_zip)
        coconut_csv = _extract_csv(zpath, work)
        frames.append(_read_coconut(coconut_csv))
        print(f"[class2] COCONUT rows={len(frames[-1])}")
    except Exception as exc:
        print(f"[class2] COCONUT unavailable ({exc}); continuing with LOTUS only")

    lotus_path = Path("/tmp/casmi_class2/compounds.tsv")
    if lotus_path.exists():
        frames.append(_read_lotus(lotus_path))
    else:
        print(f"[class2] LOTUS tsv missing at {lotus_path}")

    if not frames:
        raise SystemExit("no COCONUT or LOTUS input")

    union = pd.concat(frames, ignore_index=True)
    print(f"[class2] union raw rows={len(union)} MW filter={MW_MIN:.0f}-{MW_MAX:.0f} Da (no test.parquet)")

    best: dict[str, tuple[str, float, str]] = {}
    n_skip_train = 0
    n_skip_mass = 0
    n_skip_rdkit = 0
    for rec in union.itertuples(index=False):
        smi = str(rec.smiles)
        key = str(rec.inchikey14 or "")
        formula = str(rec.formula or "")
        if not smi or smi in {"nan", "None"}:
            continue
        if key and key in train_keys:
            n_skip_train += 1
            continue
        reported = getattr(rec, "reported_mass", np.nan)
        try:
            reported_f = float(reported)
        except Exception:
            reported_f = float("nan")
        if np.isfinite(reported_f) and (reported_f < MW_MIN - 5 or reported_f > MW_MAX + 5):
            n_skip_mass += 1
            continue
        kept = _keep_row(smi, key, formula, train_keys)
        if kept is None:
            # Distinguish mass vs rdkit failure cheaply.
            rd_key = inchikey14_from_smiles(smi)
            if not rd_key or rd_key in train_keys:
                n_skip_rdkit += 1
                continue
            n_skip_mass += 1
            continue
        smi, rd_key, formula, mass = kept
        if rd_key not in best:
            best[rd_key] = (smi, mass, formula)

    print(
        f"[class2] unique novel={len(best)} skipped_trainish={n_skip_train} "
        f"skipped_rdkit={n_skip_rdkit} skipped_outside_MW={n_skip_mass}"
    )
    keys = list(best.keys())
    smiles = [best[k][0] for k in keys]
    masses = np.array([best[k][1] for k in keys], dtype=np.float64)
    formulas = [best[k][2] for k in keys]
    order = np.argsort(masses, kind="mergesort")
    keys = [keys[i] for i in order]
    smiles = [smiles[i] for i in order]
    formulas = [formulas[i] for i in order]
    masses = masses[order]

    print(f"[class2] computing packed Morgan fps for {len(smiles)} structures ...")
    fps = np.stack([morgan_fingerprint(s, n_bits=cfg.fp_bits, radius=cfg.morgan_radius) for s in smiles])
    packed = pack_fingerprints(fps)
    out = pd.DataFrame(
        {
            "smiles": smiles,
            "inchikey14": keys,
            "exact_mass": masses,
            "formula": formulas,
            "fp_packed": [row.tobytes() for row in packed],
        }
    )
    out_dir = ROOT / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "class2_candidates.parquet"
    out.to_parquet(parquet_path, index=False)
    print(f"[class2] wrote {len(out)} structures  mass {masses.min():.1f}–{masses.max():.1f} Da")
    print(f"  {parquet_path} {parquet_path.stat().st_size} bytes (with packed fps)")
    print("[class2] filter: train InChIKey14 + generic 50-2000 Da. NOT test.parquet.")
    if len(out) < 250_000:
        print("[class2] WARNING: pool < 250k — COCONUT may be missing; expected ~400k.")


if __name__ == "__main__":
    main()
