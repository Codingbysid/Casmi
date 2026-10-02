"""Mass-aware analog expansion of retrieved SMILES (Class 2 / pad-to-25).

Applies a small set of NP-like biotransformations with RDKit. New skeletons
that land inside the query mass window are ranked first; remaining analogs
fill leftover MRR@25 slots.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from src.chem import exact_mass_from_smiles, inchikey14_from_smiles, smiles_to_mol

# Neutral-mass deltas for parent retrieval (hexose, deoxyhexose, pentose, ...).
MASS_SHIFTS: tuple[tuple[str, float], ...] = (
    ("hexose", 162.052823),
    ("deoxyhexose", 146.057909),
    ("pentose", 132.042259),
    ("glucuronide", 176.032088),
    ("methyl", 14.015650),
    ("oxygen", 15.994915),
    ("acetyl", 42.010565),
    ("water", 18.010565),
)

# SMARTS: reactant >> product. Keep this list tiny — it runs per test molecule.
_RXN_SMARTS: tuple[str, ...] = (
    "[cH:1]>>[c:1]O",           # aromatic hydroxylation
    "[CH2:1]>>[CH:1]O",         # aliphatic hydroxylation
    "[OH:1]>>[O:1]C",           # O-methylation
    "[NH:1]>>[N:1]C",           # N-methylation
    "[O:1][CH3]>>[OH:1]",       # O-demethylation
    "[N:1][CH3]>>[NH:1]",       # N-demethylation
    "[c:1][CH3]>>[cH:1]",       # aromatic demethylation
    "[cH:1]>>[c:1]C",           # aromatic methylation
    "[cH:1]>>[c:1]Cl",          # aromatic chlorination
    "[OH:1]>>[O:1]C(=O)C",      # O-acetylation
    "[C:1](=O)[OH]>>[C:1](=O)OC",  # methyl ester
    "[C:1](=O)O[CH3]>>[C:1](=O)O",  # ester hydrolysis
    # O-glycosylation / glycosidic cleavage (hexose-like ring).
    "[OH:1]>>[O:1][C@H]1O[C@H](CO)[C@@H](O)[C@H](O)[C@H]1O",
    "[#8:1]-[#6]1-[#8]-[#6]-[#6]-[#6]-[#6]-1>>[#8H:1]",
    "[OH:1]>>[O:1]C(=O)C",  # acetyl (aliphatic) already above; keep glycosyl pair
)


def _reactions():
    if not hasattr(_reactions, "_cache"):
        try:
            from rdkit.Chem import AllChem

            rxns = []
            for smarts in _RXN_SMARTS:
                try:
                    rxn = AllChem.ReactionFromSmarts(smarts)
                    if rxn is not None:
                        rxns.append(rxn)
                except Exception:
                    continue
            _reactions._cache = rxns  # type: ignore[attr-defined]
        except Exception:
            _reactions._cache = []  # type: ignore[attr-defined]
    return _reactions._cache  # type: ignore[attr-defined]


def shifted_parent_masses(query_masses: Sequence[float]) -> list[float]:
    """Query masses of putative glycosylated / analog parents."""
    out: list[float] = []
    seen: set[float] = set()
    for raw in query_masses:
        m = float(raw)
        if m <= 0 or not np.isfinite(m):
            continue
        for _, delta in MASS_SHIFTS:
            for parent in (m + delta, m - delta):
                key = round(parent, 4)
                if key in seen or parent <= 0:
                    continue
                seen.add(key)
                out.append(parent)
    return out


def expand_smiles(
    smiles_list: Sequence[str],
    *,
    query_mass: float,
    mass_ppm: float = 50.0,
    max_per_parent: int = 12,
    max_total: int = 80,
    in_window_only: bool = False,
) -> list[str]:
    """Return unique analog SMILES, mass-matching ones first."""
    rxns = _reactions()
    if not rxns:
        return []
    in_window: list[str] = []
    other: list[str] = []
    seen: set[str] = set()
    for parent in smiles_list:
        mol = smiles_to_mol(parent)
        if mol is None:
            continue
        parent_key = inchikey14_from_smiles(parent)
        if parent_key:
            seen.add(parent_key)
        n_from_parent = 0
        for rxn in rxns:
            if n_from_parent >= int(max_per_parent):
                break
            try:
                products = rxn.RunReactants((mol,))
            except Exception:
                continue
            for tup in products:
                if not tup:
                    continue
                prod = tup[0]
                try:
                    from rdkit import Chem

                    Chem.SanitizeMol(prod)
                    smi = Chem.MolToSmiles(prod, canonical=True)
                except Exception:
                    continue
                key = inchikey14_from_smiles(smi)
                if not key or key in seen:
                    continue
                seen.add(key)
                n_from_parent += 1
                mass = exact_mass_from_smiles(smi)
                ppm = (
                    abs(mass - query_mass) / query_mass * 1e6
                    if query_mass > 0 and mass > 0
                    else 1e9
                )
                if ppm <= float(mass_ppm):
                    in_window.append(smi)
                elif not in_window_only:
                    other.append(smi)
                if len(in_window) + len(other) >= int(max_total):
                    return in_window + other
    return in_window if in_window_only else in_window + other


def expand_unique(smiles_iter: Iterable[str], **kwargs) -> list[str]:
    return expand_smiles(list(smiles_iter), **kwargs)
