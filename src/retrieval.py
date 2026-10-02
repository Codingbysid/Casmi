"""Candidate retrieval: precursor-mass filter + vectorized spectral similarity."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.chem import precursor_to_neutral_mass
from src.config import Config
from src.preprocessing import bin_spectrum


@dataclass
class StructureIndex:
    """Mass-sorted unique 2D structures (one row per InChIKey14)."""

    smiles: np.ndarray  # object
    inchikey14: np.ndarray  # object / U14
    exact_mass: np.ndarray  # float64
    fingerprints: np.ndarray  # uint8 (N, fp_bits) or (N, 0) if packed-only
    order: np.ndarray  # argsort of exact_mass
    sorted_mass: np.ndarray
    peak_mz: np.ndarray | None = None  # (N, top_n)
    peak_intensity: np.ndarray | None = None
    peak_mask: np.ndarray | None = None
    formulas: np.ndarray | None = None  # object / str
    fp_packed: np.ndarray | None = None  # uint8 (N, fp_bits/8)
    ionization_mode: np.ndarray | None = None  # object / str
    # Original featurize_spectrum outputs for the same representative as peak_mz.
    # Optional for structure-only candidate libraries; required by corrected training.
    peak_nl: np.ndarray | None = None  # (N, top_n)
    precursor_feat: np.ndarray | None = None  # (N, precursor_feat_dim)
    adduct_id: np.ndarray | None = None  # (N,)

    @classmethod
    def build(
        cls,
        smiles: list[str] | np.ndarray,
        inchikey14: list[str] | np.ndarray,
        exact_mass: np.ndarray,
        fingerprints: np.ndarray,
        peak_mz: np.ndarray | None = None,
        peak_intensity: np.ndarray | None = None,
        peak_mask: np.ndarray | None = None,
        formulas: list[str] | np.ndarray | None = None,
        fp_packed: np.ndarray | None = None,
        ionization_mode: list[str] | np.ndarray | None = None,
        peak_nl: np.ndarray | None = None,
        precursor_feat: np.ndarray | None = None,
        adduct_id: np.ndarray | None = None,
    ) -> "StructureIndex":
        mass = np.asarray(exact_mass, dtype=np.float64)
        order = np.argsort(mass, kind="mergesort")
        packed = None if fp_packed is None else np.asarray(fp_packed, dtype=np.uint8)
        return cls(
            smiles=np.asarray(smiles, dtype=object),
            inchikey14=np.asarray(inchikey14, dtype=object),
            exact_mass=mass,
            fingerprints=np.asarray(fingerprints),
            order=order,
            sorted_mass=mass[order],
            peak_mz=None if peak_mz is None else np.asarray(peak_mz, dtype=np.float32),
            peak_intensity=None
            if peak_intensity is None
            else np.asarray(peak_intensity, dtype=np.float32),
            peak_mask=None if peak_mask is None else np.asarray(peak_mask, dtype=np.float32),
            formulas=None if formulas is None else np.asarray(formulas, dtype=object),
            fp_packed=packed,
            ionization_mode=None
            if ionization_mode is None
            else np.asarray(ionization_mode, dtype=object),
            peak_nl=None if peak_nl is None else np.asarray(peak_nl, dtype=np.float32),
            precursor_feat=None
            if precursor_feat is None
            else np.asarray(precursor_feat, dtype=np.float32),
            adduct_id=None if adduct_id is None else np.asarray(adduct_id, dtype=np.int64),
        )

    def mass_window(
        self,
        query_mass: float,
        *,
        ppm: float,
        abs_da: float,
    ) -> np.ndarray:
        """Return *unsorted* structure indices whose exact mass is within the window."""
        if not np.isfinite(query_mass) or query_mass <= 0:
            return np.zeros((0,), dtype=np.int64)
        tol = max(float(abs_da), float(ppm) * query_mass * 1e-6)
        lo = query_mass - tol
        hi = query_mass + tol
        left = int(np.searchsorted(self.sorted_mass, lo, side="left"))
        right = int(np.searchsorted(self.sorted_mass, hi, side="right"))
        if right <= left:
            return np.zeros((0,), dtype=np.int64)
        return self.order[left:right].astype(np.int64, copy=False)

    def query_masses(
        self,
        masses: np.ndarray | list[float],
        cfg: Config,
        *,
        use_fallback: bool = False,
        ppm: float | None = None,
        abs_da: float | None = None,
        max_candidates: int | None = None,
    ) -> np.ndarray:
        """Union of candidates for several plausible query masses (multi-adduct)."""
        ppm_v = float(cfg.mass_ppm if ppm is None else ppm)
        abs_v = float(cfg.mass_abs_da if abs_da is None else abs_da)
        cap = int(cfg.max_mass_candidates if max_candidates is None else max_candidates)
        idx_sets: list[np.ndarray] = []
        mass_arr = np.asarray(masses, dtype=np.float64).ravel()
        for m in mass_arr:
            hit = self.mass_window(float(m), ppm=ppm_v, abs_da=abs_v)
            if hit.size:
                idx_sets.append(hit)
        if not idx_sets and use_fallback:
            for extra in cfg.mass_fallback_da:
                for m in mass_arr:
                    hit = self.mass_window(float(m), ppm=0.0, abs_da=float(extra))
                    if hit.size:
                        idx_sets.append(hit)
                if idx_sets:
                    break
        if not idx_sets:
            return np.zeros((0,), dtype=np.int64)
        uniq = np.unique(np.concatenate(idx_sets))
        if uniq.size > cap:
            primary = float(mass_arr[0]) if mass_arr.size else 0.0
            err = np.abs(self.exact_mass[uniq] - primary)
            keep = np.argpartition(err, cap - 1)[:cap]
            uniq = uniq[keep]
        return uniq.astype(np.int64)

    def query_masses_progressive(
        self,
        masses: np.ndarray | list[float],
        cfg: Config,
        *,
        min_hits: int = 25,
        stages: list[tuple[float, float]] | None = None,
    ) -> np.ndarray:
        """Widen ppm / Da until at least ``min_hits`` unique skeletons (or the last stage)."""
        seen: list[np.ndarray] = []
        have: set[int] = set()
        if stages is None:
            stages = [
                (float(cfg.mass_ppm), float(cfg.mass_abs_da)),
                (25.0, 0.03),
                (50.0, 0.05),
            ]
        for ppm, abs_da in stages:
            hit = self.query_masses(
                masses,
                cfg,
                use_fallback=False,
                ppm=ppm,
                abs_da=abs_da,
                max_candidates=max(int(cfg.max_mass_candidates), 8192),
            )
            if hit.size:
                new = [int(i) for i in hit.tolist() if int(i) not in have]
                if new:
                    have.update(new)
                    seen.append(np.asarray(new, dtype=np.int64))
            if len(have) >= int(min_hits):
                break
        if not seen:
            return np.zeros((0,), dtype=np.int64)
        return np.concatenate(seen)


def binned_cosine(
    query_mz: np.ndarray,
    query_int: np.ndarray,
    query_mask: np.ndarray,
    cand_mz: np.ndarray,
    cand_int: np.ndarray,
    cand_mask: np.ndarray,
    cfg: Config,
) -> np.ndarray:
    """Cosine similarity of fixed-bin spectra. ``cand_*`` has shape ``(N, top_n)``."""
    q = bin_spectrum(query_mz, query_int, query_mask, cfg=cfg)
    qn = float(np.linalg.norm(q))
    if qn <= 0 or cand_mz.size == 0:
        return np.zeros((cand_mz.shape[0],), dtype=np.float32)
    n = cand_mz.shape[0]
    scores = np.zeros(n, dtype=np.float32)
    # Small N after mass filter: loop is fine and keeps memory bounded.
    for i in range(n):
        c = bin_spectrum(cand_mz[i], cand_int[i], cand_mask[i] if cand_mask is not None else None, cfg=cfg)
        cn = float(np.linalg.norm(c))
        if cn <= 0:
            continue
        scores[i] = float(np.dot(q, c) / (qn * cn))
    return scores


def peak_cosine_matrix(
    query_mz: np.ndarray,
    query_int: np.ndarray,
    query_mask: np.ndarray,
    cand_mz: np.ndarray,
    cand_int: np.ndarray,
    cand_mask: np.ndarray,
    tol: float,
) -> np.ndarray:
    """Greedy peak-to-peak cosine, vectorized match search with a numpy greedy pass.

    Shapes: query ``(P,)``, candidates ``(N, P)``.
    """
    q_m = query_mask > 0
    q_mz = query_mz[q_m]
    q_i = query_int[q_m]
    n = cand_mz.shape[0]
    scores = np.zeros(n, dtype=np.float32)
    if q_mz.size == 0:
        return scores
    q_norm = float(np.sqrt(np.dot(q_i, q_i)))
    if q_norm <= 0:
        return scores
    for i in range(n):
        c_m = cand_mask[i] > 0
        c_mz = cand_mz[i, c_m]
        c_i = cand_int[i, c_m]
        if c_mz.size == 0:
            continue
        c_norm = float(np.sqrt(np.dot(c_i, c_i)))
        if c_norm <= 0:
            continue
        # Pairwise |Δm/z|
        delta = np.abs(c_mz[:, None] - q_mz[None, :])
        match = delta <= tol
        if not match.any():
            continue
        # Greedy: highest intensity product first.
        prod = (c_i[:, None] * q_i[None, :]) * match
        used_c = np.zeros(c_mz.size, dtype=bool)
        used_q = np.zeros(q_mz.size, dtype=bool)
        dot = 0.0
        flat = prod.ravel()
        order = np.argsort(flat)[::-1]
        n_c = c_mz.size
        n_q = q_mz.size
        for idx in order:
            val = float(flat[idx])
            if val <= 0:
                break
            ci = int(idx) // n_q
            qi = int(idx) % n_q
            if used_c[ci] or used_q[qi]:
                continue
            used_c[ci] = True
            used_q[qi] = True
            dot += val
        scores[i] = float(dot / (q_norm * c_norm))
    return scores


def modified_cosine(
    query_mz: np.ndarray,
    query_int: np.ndarray,
    query_mask: np.ndarray,
    query_prec: float,
    cand_mz: np.ndarray,
    cand_int: np.ndarray,
    cand_mask: np.ndarray,
    cand_prec: np.ndarray,
    tol: float,
    *,
    return_n_match: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """GNPS-style modified cosine: match fragments *or* precursor-shifted fragments."""
    q_m = query_mask > 0
    q_mz = query_mz[q_m]
    q_i = query_int[q_m]
    n = cand_mz.shape[0]
    scores = np.zeros(n, dtype=np.float32)
    n_match = np.zeros(n, dtype=np.float32)
    if q_mz.size == 0:
        return (scores, n_match) if return_n_match else scores
    q_norm = float(np.sqrt(np.dot(q_i, q_i)))
    if q_norm <= 0:
        return (scores, n_match) if return_n_match else scores
    for i in range(n):
        c_m = cand_mask[i] > 0
        c_mz = cand_mz[i, c_m]
        c_i = cand_int[i, c_m]
        if c_mz.size == 0:
            continue
        c_norm = float(np.sqrt(np.dot(c_i, c_i)))
        if c_norm <= 0:
            continue
        shift = float(query_prec) - float(cand_prec[i])
        delta = np.abs(c_mz[:, None] - q_mz[None, :])
        delta_shift = np.abs(c_mz[:, None] - (q_mz[None, :] - shift))
        match = (delta <= tol) | (delta_shift <= tol)
        if not match.any():
            continue
        prod = (c_i[:, None] * q_i[None, :]) * match
        used_c = np.zeros(c_mz.size, dtype=bool)
        used_q = np.zeros(q_mz.size, dtype=bool)
        dot = 0.0
        n_q = q_mz.size
        flat = prod.ravel()
        order = np.argsort(flat)[::-1]
        n_hits = 0
        for idx in order:
            val = float(flat[idx])
            if val <= 0:
                break
            ci = int(idx) // n_q
            qi = int(idx) % n_q
            if used_c[ci] or used_q[qi]:
                continue
            used_c[ci] = True
            used_q[qi] = True
            dot += val
            n_hits += 1
        scores[i] = float(dot / (q_norm * c_norm))
        n_match[i] = float(n_hits)
    if return_n_match:
        return scores, n_match
    return scores


def tanimoto(query_fp: np.ndarray, cand_fp: np.ndarray) -> np.ndarray:
    """Tanimoto between a 1-D query fingerprint and ``(N, bits)`` candidates."""
    q = query_fp.astype(np.float32, copy=False).ravel()
    c = cand_fp.astype(np.float32, copy=False)
    if c.ndim == 1:
        c = c[None, :]
    inter = (c * q[None, :]).sum(axis=1)
    union = c.sum(axis=1) + q.sum() - inter
    return (inter / np.clip(union, 1e-6, None)).astype(np.float32)


def fingerprint_knn(
    query_fp: np.ndarray,
    cand_fp: np.ndarray,
    *,
    k: int = 25,
    threshold: float = 0.35,
    chunk: int = 8192,
    exclude: set[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-``k`` Tanimoto neighbors. ``cand_fp`` may be uint8 of shape ``(N, bits)``."""
    n = int(cand_fp.shape[0])
    if n == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    q = np.asarray(query_fp, dtype=np.float32).ravel()
    bits = (q >= float(threshold)).astype(np.uint8)
    if int(bits.sum()) == 0:
        kk = min(64, q.size)
        topb = np.argpartition(q, -kk)[-kk:]
        bits = np.zeros_like(bits)
        bits[topb] = 1
    qsum = int(bits.sum())
    scores = np.empty(n, dtype=np.float32)
    for start in range(0, n, int(chunk)):
        sl = slice(start, min(start + int(chunk), n))
        c = np.asarray(cand_fp[sl])
        if c.dtype != np.uint8:
            c = (c >= 0.5).astype(np.uint8)
        inter = np.bitwise_and(c, bits).sum(axis=1).astype(np.float32)
        csum = c.sum(axis=1).astype(np.float32)
        scores[sl] = inter / np.clip(csum + qsum - inter, 1e-6, None)
    if exclude:
        for i in exclude:
            if 0 <= int(i) < n:
                scores[int(i)] = -1.0
    k = min(int(k), n)
    pick = np.argpartition(-scores, k - 1)[:k]
    pick = pick[np.argsort(-scores[pick])]
    pick = pick[scores[pick] >= 0]
    return pick.astype(np.int64), scores[pick]


def fingerprint_cosine(query_fp: np.ndarray, cand_fp: np.ndarray) -> np.ndarray:
    q = query_fp.astype(np.float32, copy=False).ravel()
    c = cand_fp.astype(np.float32, copy=False)
    if c.ndim == 1:
        c = c[None, :]
    qn = float(np.linalg.norm(q))
    cn = np.linalg.norm(c, axis=1)
    if qn <= 0:
        return np.zeros(c.shape[0], dtype=np.float32)
    return (c @ q / (np.clip(cn, 1e-6, None) * qn)).astype(np.float32)


def query_neutral_masses(precursor_mz: np.ndarray, adducts: np.ndarray | list[str]) -> np.ndarray:
    masses = []
    for mz, ad in zip(np.asarray(precursor_mz).ravel(), list(adducts)):
        masses.append(precursor_to_neutral_mass(float(mz), str(ad)))
    return np.asarray(masses, dtype=np.float64)


def _is_structure_table(path: Path) -> bool:
    name = path.name.lower()
    return name.endswith(".parquet") or name.endswith(".csv") or name.endswith(".csv.gz")


def load_external_structure_index(path: str | None, cfg: Config) -> StructureIndex | None:
    """Class 2: optional LOTUS / PubChem / COCONUT table.

    Expected columns: ``smiles`` or ``normalized_smiles``, optional
    ``inchikey14``, ``exact_mass`` / ``molecular_formula``.
    Packed Morgan bits stay packed (~100 MB for 400k rows). Mass-window hits
    are unpacked at query time in ``_fingerprints_for``.
    """
    from pathlib import Path

    import pandas as pd

    from src.chem import formula_to_mass, inchikey14_from_smiles, morgan_fingerprint

    keywords = ("coconut", "pubchem", "chembl", "lotus", "external", "class2", "candidate")
    skip_names = {"train.parquet", "test.parquet", "sample_submission.csv"}
    here = Path(__file__).resolve().parent.parent
    candidates: list[Path] = []
    if path:
        candidates.append(Path(path))
    if cfg.external_candidates_path:
        candidates.append(Path(cfg.external_candidates_path))
    for name in (
        "class2_candidates.parquet",
        "class2_candidates.csv.gz",
        "class2_candidates.csv",
        "lotus.parquet",
        "pubchem_candidates.parquet",
        "coconut.parquet",
        "external_candidates.parquet",
        "pubchem_candidates.csv",
        "coconut.csv",
    ):
        candidates.append(cfg.data_dir / name)
        candidates.append(cfg.artifact_dir / name)
        candidates.append(here / "data" / name)
        candidates.append(Path("/kaggle/working") / name)
        candidates.append(Path("/kaggle/working") / "data" / name)
        candidates.append(Path("/kaggle/input") / name)
        candidates.append(Path("/kaggle/input/class2_candidates.parquet2") / name)
        candidates.append(Path("/kaggle/input/class2-candidates-parquet2") / name)
        candidates.append(Path("/kaggle/input/class2_candidates.parquet") / name)
    search_roots = [
        Path("/kaggle/input"),
        Path("/kaggle/working"),
        Path(cfg.data_dir),
        Path(cfg.artifact_dir),
        here / "data",
    ]
    for root in search_roots:
        if not root.exists():
            continue
        try:
            for p in root.rglob("*"):
                if not p.is_file() or p.name in skip_names:
                    continue
                if not _is_structure_table(p):
                    continue
                low = p.name.lower()
                if any(k in low for k in keywords):
                    candidates.append(p)
        except Exception:
            continue
    seen: set[str] = set()
    hit = None
    for p in candidates:
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        if p.exists() and p.is_file():
            hit = p
            break
    if hit is None:
        print("[class2] no external structure table found")
        return None
    name = hit.name.lower()
    try:
        if name.endswith(".parquet"):
            df = pd.read_parquet(hit)
        else:
            df = pd.read_csv(hit)
    except Exception as exc:
        print(f"[class2] failed to read {hit}: {exc}")
        return None
    smi_col = "normalized_smiles" if "normalized_smiles" in df.columns else "smiles"
    if smi_col not in df.columns:
        print(f"[class2] {hit} has no smiles column, skipping")
        return None
    smiles = df[smi_col].astype(str).tolist()
    if "inchikey14" in df.columns:
        keys = df["inchikey14"].astype(str).tolist()
    else:
        keys = [inchikey14_from_smiles(s) or s for s in smiles]
    if "exact_mass" in df.columns:
        masses = df["exact_mass"].to_numpy(dtype=np.float64)
    elif "molecular_formula" in df.columns:
        masses = np.array([formula_to_mass(f) for f in df["molecular_formula"].astype(str)], dtype=np.float64)
    else:
        from src.chem import exact_mass_from_smiles

        masses = np.array([exact_mass_from_smiles(s) for s in smiles], dtype=np.float64)
    n = len(smiles)
    packed = None
    if "fp_packed" in df.columns:
        packed_rows = []
        for x in df["fp_packed"].tolist():
            if isinstance(x, (bytes, bytearray, memoryview)):
                packed_rows.append(np.frombuffer(x, dtype=np.uint8))
            else:
                packed_rows.append(np.asarray(x, dtype=np.uint8).ravel())
        packed = np.stack(packed_rows).astype(np.uint8, copy=False)
        fps = np.zeros((n, 0), dtype=np.uint8)
        print(f"[class2] keeping packed Morgan fingerprints {packed.shape} from {hit} (lazy unpack)")
    elif "fingerprint" in df.columns:
        fps = np.stack(df["fingerprint"].to_numpy())
    elif n <= 250_000:
        print(f"[class2] computing Morgan fingerprints for {n} structures from {hit} ...")
        fps = np.stack([morgan_fingerprint(s, n_bits=cfg.fp_bits, radius=cfg.morgan_radius) for s in smiles])
    else:
        print(f"[class2] skipping full-table fingerprints for {n} structures (computed on mass hits)")
        fps = np.zeros((n, 0), dtype=np.uint8)
    formulas = None
    if "formula" in df.columns:
        formulas = df["formula"].astype(str).tolist()
    elif "molecular_formula" in df.columns:
        formulas = df["molecular_formula"].astype(str).tolist()
    print(f"[class2] loaded {n} structures from {hit} mass {float(np.nanmin(masses)):.1f}–{float(np.nanmax(masses)):.1f} Da")
    return StructureIndex.build(smiles, keys, masses, fps, formulas=formulas, fp_packed=packed)
