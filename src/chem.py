"""Chemistry utilities: formula mass, adduct parsing, RDKit fingerprints, InChIKey14."""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

import numpy as np

from src.config import ELEMENT_MASSES, ELECTRON_MASS, Config

try:
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")
    _HAS_RDKIT = True
    _MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
except Exception:  # pragma: no cover - environment without RDKit
    Chem = None  # type: ignore
    rdFingerprintGenerator = None  # type: ignore
    _HAS_RDKIT = False
    _MORGAN_GEN = None

# Competition InChIKey14 is computed after RDKit tautomer canonicalization.
# The enumerator is not thread-safe; Canonicalize runs under _TAUTOMER_LOCK.
_TAUTOMER_ENUMERATOR = None
_TAUTOMER_LOCK = threading.Lock()
if _HAS_RDKIT:
    try:
        from rdkit.Chem.MolStandardize import rdMolStandardize

        _TAUTOMER_ENUMERATOR = rdMolStandardize.TautomerEnumerator()
        _TAUTOMER_ENUMERATOR.SetMaxTransforms(1000)
    except Exception:
        _TAUTOMER_ENUMERATOR = None


_FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")
_ADDUCT_RE = re.compile(
    r"^\[(\d*)M(?:([A-Za-z0-9]*)?)?(.*?)\](\d*)([+-])$"
)
_EXTRA_TOKEN = re.compile(r"([+-])([A-Za-z0-9]+)")


@dataclass(frozen=True, slots=True)
class AdductInfo:
    """Parsed adduct used to convert precursor m/z into neutral monoisotopic mass."""

    raw: str
    n_molecules: int
    charge: int  # signed
    extra_neutral_mass: float
    delta: float  # ion-mass offset: extra_neutral - signed_charge * m_e

    @property
    def abs_charge(self) -> int:
        return max(abs(self.charge), 1)


def parse_formula(formula: str | None) -> dict[str, int]:
    """Parse a Hill-system formula such as ``C21H30N7O17P3`` into element counts."""
    if not formula:
        return {}
    s = str(formula).strip().replace(" ", "")
    if not s or s.lower() in {"nan", "none"}:
        return {}
    counts: dict[str, int] = {}
    idx = 0
    while idx < len(s):
        m = _FORMULA_TOKEN.match(s, idx)
        if m is None:
            # Skip leftover punctuation rather than aborting the whole formula.
            idx += 1
            continue
        el, num = m.group(1), m.group(2)
        counts[el] = counts.get(el, 0) + (int(num) if num else 1)
        idx = m.end()
    return counts


FORMULA_ELEMENTS: tuple[str, ...] = ("C", "H", "N", "O", "P", "S", "Cl", "Br")


def formula_count_vector(formula: str | None) -> np.ndarray:
    """CHNOPS + Cl/Br counts as float32 length 8."""
    counts = parse_formula(formula)
    return np.array([float(counts.get(el, 0)) for el in FORMULA_ELEMENTS], dtype=np.float32)


def formula_match_score(pred_counts: np.ndarray, formula: str | None) -> float:
    """Soft agreement in (0, 1] between predicted element counts and a formula."""
    if not formula:
        return 1.0
    cand = formula_count_vector(formula)
    pred = np.asarray(pred_counts, dtype=np.float32).ravel()
    if pred.size < 4:
        return 1.0
    dc = abs(float(pred[0]) - float(cand[0]))
    dh = abs(float(pred[1]) - float(cand[1])) if pred.size > 1 else 0.0
    dn = abs(float(pred[2]) - float(cand[2])) if pred.size > 2 else 0.0
    do = abs(float(pred[3]) - float(cand[3])) if pred.size > 3 else 0.0
    return float(np.exp(-0.5 * ((dc / 3.0) ** 2 + (dh / 8.0) ** 2 + (dn / 1.5) ** 2 + (do / 2.0) ** 2)))


def formula_to_mass(formula: str | None) -> float:
    """Monoisotopic mass from a molecular formula. Unknown elements contribute 0."""
    counts = parse_formula(formula)
    mass = 0.0
    for el, n in counts.items():
        mass += ELEMENT_MASSES.get(el, 0.0) * n
    return float(mass)


def _formula_mass_with_leading_count(token: str) -> float:
    """Parse ``2H``, ``H2O``, ``Na``, ``CH2O2`` into a monoisotopic mass."""
    if not token:
        return 0.0
    m = re.match(r"^(\d+)([A-Z].*)$", token)
    if m:
        return int(m.group(1)) * _formula_mass_with_leading_count(m.group(2))
    return formula_to_mass(token)


@lru_cache(maxsize=512)
def parse_adduct(adduct: str | None) -> AdductInfo:
    """Parse strings like ``[M+H]+``, ``[2M+Na]+``, ``[M+CH2O2-H]-``, ``[M+2H]2+``."""
    raw = "" if adduct is None else str(adduct).strip()
    if not raw:
        return AdductInfo(raw=raw, n_molecules=1, charge=1, extra_neutral_mass=0.0, delta=0.0)

    m = _ADDUCT_RE.match(raw)
    if m is None:
        # Best-effort: assume [M+H]+ so downstream math stays finite.
        proton = ELEMENT_MASSES["H"] - ELECTRON_MASS
        return AdductInfo(
            raw=raw, n_molecules=1, charge=1, extra_neutral_mass=ELEMENT_MASSES["H"], delta=proton
        )

    n_str, _bridge, extras, z_str, sign = m.groups()
    n_molecules = int(n_str) if n_str else 1
    abs_z = int(z_str) if z_str else 1
    signed_z = abs_z if sign == "+" else -abs_z

    extra_neutral = 0.0
    extra_str = extras or ""
    if extra_str:
        tokens = _EXTRA_TOKEN.findall(extra_str)
        if not tokens and extra_str:
            # e.g. unexpected leftover; try as a leading-sign-less formula.
            extra_neutral += _formula_mass_with_leading_count(extra_str)
        for sgn, tok in tokens:
            mass = _formula_mass_with_leading_count(tok)
            extra_neutral += mass if sgn == "+" else -mass

    delta = extra_neutral - signed_z * ELECTRON_MASS
    return AdductInfo(
        raw=raw,
        n_molecules=max(n_molecules, 1),
        charge=signed_z,
        extra_neutral_mass=float(extra_neutral),
        delta=float(delta),
    )


def precursor_to_neutral_mass(precursor_mz: float, adduct: str | None) -> float:
    """Neutral monoisotopic mass from precursor m/z and adduct annotation.

    ``mz = (n * M + extra_neutral - z * m_e) / |z|``
    """
    info = parse_adduct(adduct)
    mz = float(precursor_mz)
    if not np.isfinite(mz) or mz <= 0:
        return 0.0
    mass = (mz * info.abs_charge - info.delta) / float(info.n_molecules)
    if not np.isfinite(mass) or mass <= 0:
        return 0.0
    return float(mass)


# mz → monoisotopic M for |z|=1, n=1:  M = mz + offset.
# Values are (extra_neutral − z·m_e) with opposite sign so they match
# the CASMI / ESI convention to ≥5 decimal places.
ADDUCT_OFFSETS: dict[str, float] = {
    "[M+H]+": -1.007276,
    "[M+NH4]+": -18.033826,
    "[M+Na]+": -22.989218,
    "[M+K]+": -38.963158,
    "[M-H]-": +1.007276,
    "[M+CH2O2-H]-": -44.998201,
    "[M+Cl]-": -34.969402,
    "[M+C2H4O2-H]-": -59.013851,
    "[M+Br]-": -78.918885,
}


# Tight Class 1 set. Extra adducts open extra ±15 ppm windows and let
# wrong-mass isomers outrank the true skeleton on noisy cosine.
POS_ADDUCTS: tuple[str, ...] = (
    "[M+H]+",
    "[M+Na]+",
)
NEG_ADDUCTS: tuple[str, ...] = (
    "[M-H]-",
    "[M+CH2O2-H]-",
)
# Class 2 has no MS2 library, so probe the common ESI alternatives.
CLASS2_POS_ADDUCTS: tuple[str, ...] = (
    "[M+H]+",
    "[M+Na]+",
    "[M+NH4]+",
    "[M+K]+",
)
CLASS2_NEG_ADDUCTS: tuple[str, ...] = (
    "[M-H]-",
    "[M+CH2O2-H]-",
    "[M+C2H4O2-H]-",
    "[M+Cl]-",
)
COMMON_ADDUCTS: tuple[str, ...] = CLASS2_POS_ADDUCTS + CLASS2_NEG_ADDUCTS

# 37Cl − 35Cl and 81Br − 79Br (monoisotopic minor vs major halogen adduct).
_CL37_MINUS_CL35: float = 36.96590258 - 34.968852682
_BR81_MINUS_BR79: float = 80.9162897 - 78.9183376


def adduct_offset(adduct: str | None) -> float:
    """Return ``M - mz`` for a singly charged 1M ion (0 if unknown)."""
    raw = "" if adduct is None else str(adduct).strip()
    if raw in ADDUCT_OFFSETS:
        return float(ADDUCT_OFFSETS[raw])
    info = parse_adduct(raw)
    if info.n_molecules != 1 or info.abs_charge != 1:
        return 0.0
    return float(-info.delta)


def plausible_neutral_masses(
    precursor_mz: float,
    adduct: str | None = None,
    *,
    ionization_mode: str | None = None,
    exhaustive: bool = False,
) -> list[float]:
    """Neutral masses under the annotated adduct and ion-mode-matched alternatives.

    ``exhaustive=True`` also probes the opposite polarity and halogen isotopes.
    The tight Class 1 window should use the default (conservative) set.
    """
    seen: set[float] = set()
    out: list[float] = []

    def _add(mass: float) -> None:
        if mass <= 0 or not np.isfinite(mass):
            return
        key = round(mass, 4)
        if key in seen:
            return
        seen.add(key)
        out.append(float(mass))

    mz = float(precursor_mz)
    ion = str(ionization_mode or "").lower()
    if exhaustive:
        pos_set, neg_set = CLASS2_POS_ADDUCTS, CLASS2_NEG_ADDUCTS
    else:
        pos_set, neg_set = POS_ADDUCTS, NEG_ADDUCTS
    if ion.startswith("neg"):
        mode_adducts: tuple[str, ...] = neg_set
    elif ion.startswith("pos"):
        mode_adducts = pos_set
    else:
        mode_adducts = pos_set + neg_set
    probes = (adduct, *mode_adducts)
    if exhaustive:
        probes = (adduct, *CLASS2_POS_ADDUCTS, *CLASS2_NEG_ADDUCTS, "[M+Br]-", "[M-H2O+H]+", "[M-H2O-H]-")
    for ad in probes:
        if not ad:
            continue
        if ad in ADDUCT_OFFSETS:
            _add(mz + ADDUCT_OFFSETS[ad])
        _add(precursor_to_neutral_mass(mz, ad))
    if exhaustive or (adduct and "Cl" in str(adduct)):
        _add(precursor_to_neutral_mass(mz, "[M+Cl]-") - _CL37_MINUS_CL35)
    if exhaustive or (adduct and "Br" in str(adduct)):
        _add(precursor_to_neutral_mass(mz, "[M+Br]-") - _BR81_MINUS_BR79)
    return out


def smiles_to_mol(smiles: str | None):
    if not _HAS_RDKIT or not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(str(smiles))
    except Exception:
        return None
    return mol


def _inchikey14_from_mol(mol) -> str:
    if mol is None or Chem is None:
        return ""
    try:
        key = Chem.MolToInchiKey(mol)
    except Exception:
        return ""
    if not key:
        return ""
    return key.split("-")[0][:14]


def _tautomer_canonical_mol(mol):
    """Return the RDKit canonical tautomer, or ``mol`` if enumeration fails."""
    if mol is None or _TAUTOMER_ENUMERATOR is None:
        return mol
    try:
        with _TAUTOMER_LOCK:
            canon = _TAUTOMER_ENUMERATOR.Canonicalize(mol)
    except Exception:
        return mol
    return mol if canon is None else canon


@lru_cache(maxsize=200_000)
def inchikey14_from_smiles(smiles: str | None) -> str:
    """First InChIKey block after tautomer canonicalization. Empty on failure.

    The competition metric hashes InChIKey14 from
    ``TautomerEnumerator().Canonicalize(mol)``. Keto/enol and amide/imidic
    forms of one skeleton must share this key.
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        return ""
    canon = _tautomer_canonical_mol(mol)
    key = _inchikey14_from_mol(canon)
    if key:
        return key
    if canon is not mol:
        return _inchikey14_from_mol(mol)
    return ""


def canonical_smiles(smiles: str | None) -> str:
    mol = smiles_to_mol(smiles)
    if mol is None:
        return "" if not smiles else str(smiles)
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return str(smiles)


def morgan_fingerprint(smiles: str | None, n_bits: int = 2048, radius: int = 2) -> np.ndarray:
    """Return a uint8 vector of shape ``(n_bits,)``. Zeros if SMILES cannot be parsed."""
    fp = np.zeros(n_bits, dtype=np.uint8)
    if not _HAS_RDKIT or not smiles:
        return fp
    mol = smiles_to_mol(smiles)
    if mol is None:
        return fp
    try:
        if n_bits == 2048 and radius == 2 and _MORGAN_GEN is not None:
            bitvect = _MORGAN_GEN.GetFingerprint(mol)
        else:
            gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
            bitvect = gen.GetFingerprint(mol)
        on_bits = np.asarray(list(bitvect.GetOnBits()), dtype=np.int32)
        on_bits = on_bits[on_bits < n_bits]
        fp[on_bits] = 1
        return fp
    except Exception:
        return fp


def morgan_fingerprint_batch(smiles_list: Iterable[str], cfg: Config | None = None) -> np.ndarray:
    n_bits = 2048 if cfg is None else cfg.fp_bits
    radius = 2 if cfg is None else cfg.morgan_radius
    smiles_list = list(smiles_list)
    out = np.zeros((len(smiles_list), n_bits), dtype=np.uint8)
    for i, smi in enumerate(smiles_list):
        out[i] = morgan_fingerprint(smi, n_bits=n_bits, radius=radius)
    return out


def pack_fingerprints(fp: np.ndarray) -> np.ndarray:
    """Pack bit last-axis to uint8 bytes."""
    return np.packbits(fp.astype(np.uint8, copy=False), axis=-1)


def unpack_fingerprints(packed: np.ndarray, n_bits: int) -> np.ndarray:
    bits = np.unpackbits(packed, axis=-1)
    return bits[..., :n_bits]


def has_rdkit() -> bool:
    return bool(_HAS_RDKIT)


def randomize_smiles(smiles: str | None, attempts: int = 8) -> str:
    """Non-canonical SMILES traversal; falls back to the input string."""
    mol = smiles_to_mol(smiles)
    if mol is None:
        return "" if not smiles else str(smiles)
    for _ in range(max(int(attempts), 1)):
        try:
            out = Chem.MolToSmiles(mol, canonical=False, doRandom=True, isomericSmiles=True)
            if out:
                return out
        except Exception:
            continue
    return canonical_smiles(smiles)


def exact_mass_from_smiles(smiles: str | None) -> float:
    mol = smiles_to_mol(smiles)
    if mol is None:
        return 0.0
    try:
        from rdkit.Chem import Descriptors

        return float(Descriptors.ExactMolWt(mol))
    except Exception:
        return 0.0
