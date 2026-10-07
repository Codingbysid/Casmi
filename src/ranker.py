"""Ensemble ranking, InChIKey14 skeleton dedup, and submission formatting."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src.chem import canonical_smiles, has_rdkit, inchikey14_from_smiles, smiles_to_mol
from src.config import Config
from src.retrieval import fingerprint_cosine, tanimoto

FALLBACK_SMILES = "CCO"


def mass_gaussian_score(dm: np.ndarray, mass: np.ndarray, ppm_scale: float) -> np.ndarray:
    ppm = np.abs(dm) / np.clip(mass, 1e-6, None) * 1e6
    return np.exp(-0.5 * (ppm / max(ppm_scale, 1e-3)) ** 2).astype(np.float32)


def combine_scores(
    spec_sim: np.ndarray,
    fp_tanimoto: np.ndarray,
    mass_score: np.ndarray,
    cfg: Config,
    fp_cosine: np.ndarray | None = None,
) -> np.ndarray:
    spec = np.nan_to_num(spec_sim, nan=0.0).astype(np.float32)
    fp = np.nan_to_num(fp_tanimoto, nan=0.0).astype(np.float32)
    if fp_cosine is not None:
        fp = 0.6 * fp + 0.4 * np.nan_to_num(fp_cosine, nan=0.0).astype(np.float32)
    mass = np.nan_to_num(mass_score, nan=0.0).astype(np.float32)
    w = float(cfg.w_spectral + cfg.w_fingerprint + cfg.w_mass)
    if w <= 0:
        w = 1.0
    return (cfg.w_spectral * spec + cfg.w_fingerprint * fp + cfg.w_mass * mass) / w


def dedup_inchikey14(
    smiles: Sequence[str],
    scores: np.ndarray,
    inchikey14: Sequence[str] | None = None,
    top_k: int = 25,
) -> list[str]:
    """Keep the highest-scoring SMILES per tautomer-canonical InChIKey14.

    Stereoisomers and tautomers (keto/enol, amide/imidic) share one skeleton
    under the competition metric, so a second form would waste an MRR@25 slot.
    The key is recomputed from SMILES. A stored key is only a fallback when
    RDKit cannot parse the SMILES.
    """
    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    seen: set[str] = set()
    picked: list[str] = []
    for idx in order:
        smi = str(smiles[int(idx)])
        if not smi or smi.lower() in {"nan", "none"}:
            continue
        key = inchikey14_from_smiles(smi)
        if not key and inchikey14 is not None:
            key = str(inchikey14[int(idx)] or "")
        if not key:
            key = f"RAW::{smi}"
        if key in seen:
            continue
        seen.add(key)
        picked.append(smi)
        if len(picked) >= top_k:
            break
    return picked


def merge_unique_smiles(
    primary: Sequence[str],
    extra: Sequence[str],
    *,
    top_k: int = 25,
) -> list[str]:
    """Append ``extra`` SMILES, dropping InChIKey14 collisions, up to ``top_k``."""
    out: list[str] = []
    seen: set[str] = set()
    for smi in list(primary) + list(extra):
        if not smi or str(smi).lower() in {"nan", "none"}:
            continue
        key = inchikey14_from_smiles(smi) or str(smi)
        if key in seen:
            continue
        seen.add(key)
        out.append(str(smi))
        if len(out) >= top_k:
            break
    return out


def format_prediction_row(molecule_id: str, smiles_list: Sequence[str]) -> str:
    guesses = sanitize_guesses(smiles_list, top_k=25)
    if not guesses:
        guesses = [FALLBACK_SMILES]
    return ";".join(guesses)


def sanitize_guesses(smiles_list: Sequence[str] | None, top_k: int = 25) -> list[str]:
    """Keep at most ``top_k`` parseable unique-skeleton SMILES, no empties/nulls."""
    out: list[str] = []
    seen: set[str] = set()
    if not smiles_list:
        return out
    for raw in smiles_list:
        if raw is None:
            continue
        if isinstance(raw, float) and not np.isfinite(raw):
            continue
        text = str(raw).replace("\r", " ").replace("\n", " ").strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            continue
        for part in text.split(";"):
            part = part.strip().strip('"').strip("'")
            if not part or part.lower() in {"nan", "none", "null"}:
                continue
            mol = smiles_to_mol(part)
            if mol is not None:
                canon = canonical_smiles(part) or part
            elif has_rdkit():
                continue
            else:
                canon = part
            if not canon or ";" in canon:
                continue
            key = inchikey14_from_smiles(canon) or canon
            if key in seen:
                continue
            seen.add(key)
            out.append(canon)
            if len(out) >= int(top_k):
                return out
    return out


def official_molecule_ids(cfg: Config, test_df: pd.DataFrame | None = None) -> list[str]:
    sample_path = cfg.sample_submission_path
    if sample_path.exists():
        ids = pd.read_csv(sample_path, dtype=str)["molecule_id"].astype(str).tolist()
        if ids:
            return ids
    if test_df is not None and "molecule_id" in test_df.columns:
        return list(dict.fromkeys(test_df["molecule_id"].astype(str).tolist()))
    raise FileNotFoundError("Need sample_submission.csv or a test frame with molecule_id")


def predictions_to_submission(
    molecule_ids: Sequence[str],
    ranked_smiles: Sequence[Sequence[str]],
) -> pd.DataFrame:
    rows = []
    for mid, smiles in zip(molecule_ids, ranked_smiles):
        rows.append({"molecule_id": str(mid), "smiles": format_prediction_row(str(mid), smiles)})
    return pd.DataFrame(rows, columns=["molecule_id", "smiles"])


def build_submission_frame(
    molecule_ids: Sequence[str],
    ranked_map: dict[str, Sequence[str]],
    *,
    top_k: int = 25,
    fallback: str = FALLBACK_SMILES,
) -> pd.DataFrame:
    rows = []
    for mid in molecule_ids:
        guesses = sanitize_guesses(ranked_map.get(str(mid), []), top_k=top_k)
        if not guesses:
            guesses = sanitize_guesses([fallback], top_k=top_k) or [fallback]
        rows.append({"molecule_id": str(mid), "smiles": ";".join(guesses[:top_k])})
    return pd.DataFrame(rows, columns=["molecule_id", "smiles"])


def write_submission_csv(df: pd.DataFrame, path: Path | str) -> Path:
    """Write a competition CSV with exactly two columns and no empty fields."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df[["molecule_id", "smiles"]].copy()
    out["molecule_id"] = out["molecule_id"].astype(str)
    out["smiles"] = out["smiles"].astype(str)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["molecule_id", "smiles"])
        for mid, smi in out.itertuples(index=False, name=None):
            if not str(mid).strip() or not str(smi).strip():
                raise ValueError(f"empty submission field for {mid!r}")
            writer.writerow([str(mid), str(smi)])
    validate_submission(path)
    return path


def validate_submission(path: Path | str, sample_path: Path | str | None = None) -> None:
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError("submission.csv is empty")
    header = [h.strip() for h in rows[0]]
    if header != ["molecule_id", "smiles"]:
        raise ValueError(f"bad header {header}")
    body = rows[1:]
    if not body:
        raise ValueError("submission.csv has no data rows")
    ids: list[str] = []
    for i, row in enumerate(body, start=2):
        if len(row) != 2:
            raise ValueError(f"line {i} has {len(row)} columns")
        mid, smi = row[0].strip(), row[1].strip()
        if not mid or not smi:
            raise ValueError(f"line {i} has an empty molecule_id or smiles")
        parts = [p for p in smi.split(";") if p]
        if not parts:
            raise ValueError(f"line {i} has no SMILES guesses")
        if len(parts) > 25:
            raise ValueError(f"line {i} has {len(parts)} guesses")
        ids.append(mid)
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate molecule_id in submission.csv")
    if sample_path is not None and Path(sample_path).exists():
        expected = pd.read_csv(sample_path, dtype=str)["molecule_id"].astype(str).tolist()
        if ids != expected:
            raise ValueError(
                f"molecule_id mismatch vs sample_submission "
                f"(got {len(ids)} expected {len(expected)})"
            )


def soft_tanimoto(pred_prob: np.ndarray, cand_fp: np.ndarray) -> np.ndarray:
    """Probability-weighted Tanimoto between a predicted fp and candidate bits."""
    q = np.asarray(pred_prob, dtype=np.float32).ravel()
    c = np.asarray(cand_fp, dtype=np.float32)
    if c.ndim == 1:
        c = c.reshape(1, -1)
    inter = c @ q
    union = c.sum(axis=1) + float(q.sum()) - inter
    return (inter / np.clip(union, 1e-6, None)).astype(np.float32)


def fingerprint_scores(
    pred_prob: np.ndarray,
    cand_fp: np.ndarray,
    threshold: float,
    *,
    soft_mix: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    bits = (pred_prob >= threshold).astype(np.float32)
    if bits.sum() == 0:
        # Avoid empty fingerprints: take top 64 predicted bits.
        k = min(64, pred_prob.size)
        top = np.argpartition(pred_prob, -k)[-k:]
        bits = np.zeros_like(pred_prob, dtype=np.float32)
        bits[top] = 1.0
    tani = tanimoto(bits, cand_fp)
    mix = float(np.clip(soft_mix, 0.0, 1.0))
    if mix > 0:
        tani_s = soft_tanimoto(pred_prob, cand_fp)
        tani = (1.0 - mix) * tani + mix * tani_s
    cos = fingerprint_cosine(pred_prob, cand_fp)
    return tani, cos
