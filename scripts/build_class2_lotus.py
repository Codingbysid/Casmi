#!/usr/bin/env python3
"""Build a Class 2 NP library from LOTUS that is safe for the hidden Kaggle test set.

CRITICAL: do not read test.parquet. The public test file is a train placeholder;
Kaggle replaces it on re-run. Filtering LOTUS to those precursor masses would
yield ~0 Class 2 hits on the real 400 unknowns.

Keep every unique LOTUS skeleton that is not already in train and whose
monoisotopic mass sits in a generic LC-MS window (50–2000 Da).
"""

from __future__ import annotations

import sys
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


def _inchi_formula(inchi: str) -> str:
    if not isinstance(inchi, str) or not inchi.startswith("InChI"):
        return ""
    parts = inchi.split("/")
    return parts[1] if len(parts) > 1 else ""


def main() -> None:
    lotus_path = Path("/tmp/casmi_class2/compounds.tsv")
    if not lotus_path.exists():
        raise FileNotFoundError(lotus_path)

    cfg = get_config()
    train_keys: set[str] = set()
    if cfg.train_path.exists():
        keys = pd.read_parquet(cfg.train_path, columns=["inchikey14"])["inchikey14"]
        train_keys = set(keys.dropna().astype(str).str.slice(0, 14).unique())
        print(f"[class2] train unique InChIKey14={len(train_keys)}")

    usecols = ["canonicalSmiles", "isomericSmiles", "inchi", "inchiKey"]
    lotus = pd.read_csv(lotus_path, sep="\t", usecols=usecols, low_memory=False)
    print(f"[class2] LOTUS rows={len(lotus)}  MW filter={MW_MIN:.0f}-{MW_MAX:.0f} Da (no test.parquet)")

    keep_smi: list[str] = []
    keep_key: list[str] = []
    keep_formula: list[str] = []
    seen: set[str] = set()
    n_skip_train = 0
    n_skip_mass = 0
    for rec in lotus.itertuples(index=False):
        full_key = str(getattr(rec, "inchiKey") or "")
        key = full_key.split("-")[0][:14] if full_key and full_key != "nan" else ""
        if not key or key in seen:
            continue
        if key in train_keys:
            n_skip_train += 1
            continue
        formula = _inchi_formula(str(getattr(rec, "inchi") or ""))
        mass = formula_to_mass(formula) if formula else 0.0
        if mass < MW_MIN or mass > MW_MAX:
            n_skip_mass += 1
            continue
        iso = str(getattr(rec, "isomericSmiles") or "")
        can = str(getattr(rec, "canonicalSmiles") or "")
        smi = iso if iso and iso not in {"nan", "None"} else can
        if not smi or smi in {"nan", "None"}:
            continue
        seen.add(key)
        keep_smi.append(smi)
        keep_key.append(key)
        keep_formula.append(formula)
    print(
        f"[class2] MW-window novel={len(keep_smi)} skipped_train={n_skip_train} "
        f"skipped_outside_MW={n_skip_mass}"
    )

    smiles_out: list[str] = []
    keys_out: list[str] = []
    formulas_out: list[str] = []
    masses_out: list[float] = []
    for smi, key, formula in zip(keep_smi, keep_key, keep_formula):
        rd_key = inchikey14_from_smiles(smi)
        if not rd_key or rd_key in train_keys:
            continue
        rd_mass = exact_mass_from_smiles(smi)
        if rd_mass < MW_MIN or rd_mass > MW_MAX:
            continue
        counts = parse_formula(formula)
        if not counts:
            formula = ""
        smiles_out.append(smi)
        keys_out.append(rd_key)
        formulas_out.append(formula)
        masses_out.append(float(rd_mass))

    best: dict[str, tuple[str, float, str]] = {}
    for smi, key, mass, formula in zip(smiles_out, keys_out, masses_out, formulas_out):
        if key not in best:
            best[key] = (smi, mass, formula)

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
    slim = out[["smiles", "inchikey14", "exact_mass", "formula"]]
    out_dir = ROOT / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = out_dir / "class2_candidates.parquet"
    gz_path = out_dir / "class2_candidates.csv.gz"
    out.to_parquet(parquet_path, index=False)
    slim.to_csv(gz_path, index=False, compression="gzip")
    print(f"[class2] wrote {len(out)} structures  mass {masses.min():.1f}–{masses.max():.1f} Da")
    print(f"  {parquet_path} {parquet_path.stat().st_size} bytes (with packed fps)")
    print(f"  {gz_path} {gz_path.stat().st_size} bytes (embeddable, no fps)")
    print("[class2] filter: train InChIKey14 + generic 50-2000 Da. NOT test.parquet.")


if __name__ == "__main__":
    main()
