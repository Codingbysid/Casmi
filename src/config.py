"""Centralized hyperparameters, adduct tables, and filesystem paths.

Shapes in this config are the TPU/XLA contract: every spectral tensor is padded
or binned to these fixed dimensions so the XLA graph is compiled once.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# Monoisotopic masses (most abundant isotope) and the electron mass.
ELEMENT_MASSES: dict[str, float] = {
    "H": 1.00782503224,
    "D": 2.01410177811,
    "C": 12.0,
    "N": 14.00307400443,
    "O": 15.99491461957,
    "F": 18.99840316273,
    "Na": 22.9897692820,
    "Mg": 23.985041697,
    "Si": 27.97692653465,
    "P": 30.97376199842,
    "S": 31.9720711744,
    "Cl": 34.968852682,
    "K": 38.9637064864,
    "Ca": 39.962590863,
    "Fe": 55.93493633,
    "Cu": 62.92959772,
    "Zn": 63.92914201,
    "Br": 78.9183376,
    "I": 126.9044719,
    "Li": 7.0160034366,
    "B": 11.009305167,
    "Se": 79.9165218,
    "As": 74.92159457,
    "e": 0.000548579909065,
}

ELECTRON_MASS: float = ELEMENT_MASSES["e"]
PROTON_MASS: float = ELEMENT_MASSES["H"] - ELECTRON_MASS


def _default_data_dir() -> Path:
    here = Path(__file__).resolve().parent.parent
    local = here / "data"
    kaggle_candidates = [
        Path("/kaggle/input/enveda-casmi26-molecule-id-mass-spectra"),
        Path("/kaggle/input/enveda-CASMI26-molecule-id-mass-spectra"),
    ]
    for cand in kaggle_candidates:
        if cand.exists():
            return cand
    if (local / "train.parquet").exists() or (local / "test.parquet").exists():
        return local
    bundled = here / "enveda-CASMI26-molecule-id-mass-spectra"
    if bundled.exists():
        return bundled
    return local


@dataclass
class Config:
    """All pipeline hyperparameters. Mutate a copy rather than editing callers."""

    # Paths
    data_dir: Path = field(default_factory=_default_data_dir)
    artifact_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parent.parent / "artifacts"
    )

    # Fixed spectral shapes (XLA-static)
    top_n_peaks: int = 128
    n_mz_bins: int = 512
    mz_bin_min: float = 0.0
    mz_bin_max: float = 600.0
    fp_bits: int = 2048
    morgan_radius: int = 2

    # Peak cleaning
    precursor_peak_margin_da: float = 2.0
    min_rel_intensity: float = 0.005  # 0.5% of base peak
    min_mz: float = 1.0

    # Mass filter
    mass_ppm: float = 15.0
    mass_abs_da: float = 0.02
    mass_fallback_da: tuple[float, ...] = (0.05, 0.2, 0.5, 2.0)
    max_mass_candidates: int = 4096

    # Spectral similarity
    cosine_mz_tol: float = 0.05
    modified_cosine_mz_tol: float = 0.05

    # Ranker
    w_spectral: float = 0.42
    w_fingerprint: float = 0.46
    w_mass: float = 0.12
    top_k: int = 25
    mass_score_ppm_scale: float = 10.0
    # V6: calibrated OR-merge below the cosine lock; 3-seed logits; neighbor fp.
    use_soft_merge: bool = True
    soft_merge_lock_cosine: float = 0.75
    fp_soft_mix: float = 0.5
    neighbor_k: int = 20
    neighbor_blend: float = 0.5
    use_neighbor_fp: bool = True
    use_mass_shift: bool = True
    ensemble_seeds: int = 3

    # Model
    d_model: int = 256
    n_heads: int = 8
    n_transformer_layers: int = 3
    n_conv_channels: int = 128
    adduct_embed_dim: int = 32
    dropout: float = 0.10
    precursor_feat_dim: int = 8

    # Training
    batch_size: int = 128
    num_epochs: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    max_grad_norm: float = 1.0
    focal_gamma: float = 1.5
    bce_pos_weight: float = 6.0
    num_workers: int = 0
    seed: int = 42
    val_fraction: float = 0.05
    max_train_spectra: int = 400_000
    log_every: int = 50
    ckpt_name: str = "spec2fp.pt"
    train_time_limit_s: float = 6.0 * 3600.0  # leave headroom inside the 9 h Kaggle cap

    # Inference
    fp_threshold: float = 0.35
    min_candidates: int = 25

    # Prefix-conditioned SMILES decoder (Class 3 / de novo)
    smiles_max_len: int = 128
    smiles_prefix_len: int = 8
    smiles_d_model: int = 256
    smiles_nhead: int = 8
    smiles_num_layers: int = 4
    smiles_dim_feedforward: int = 1024
    smiles_dropout: float = 0.1
    smiles_label_smoothing: float = 0.05
    smiles_num_epochs: int = 6
    smiles_batch_size: int = 128
    smiles_lr: float = 1e-3
    smiles_train_time_limit_s: float = 1.0 * 3600.0
    smiles_randomize: bool = True
    smiles_num_samples: int = 100
    smiles_temperature: float = 0.75
    smiles_top_p: float = 0.90
    smiles_gen_max_len: int = 100
    smiles_mass_ppm: float = 20.0
    smiles_ckpt_name: str = "smiles_decoder.pt"
    smiles_vocab_name: str = "smiles_vocab.json"
    # Optional offline PubChem / COCONUT table (parquet/csv) for Class 2.
    external_candidates_path: Path | None = None

    @property
    def train_path(self) -> Path:
        return self.data_dir / "train.parquet"

    @property
    def test_path(self) -> Path:
        return self.data_dir / "test.parquet"

    @property
    def sample_submission_path(self) -> Path:
        return self.data_dir / "sample_submission.csv"

    @property
    def checkpoint_path(self) -> Path:
        return self.artifact_dir / self.ckpt_name

    @property
    def smiles_checkpoint_path(self) -> Path:
        return self.artifact_dir / self.smiles_ckpt_name

    @property
    def smiles_vocab_path(self) -> Path:
        return self.artifact_dir / self.smiles_vocab_name

    @property
    def mz_bin_width(self) -> float:
        return (self.mz_bin_max - self.mz_bin_min) / float(self.n_mz_bins)


# Closed vocabulary of adducts seen in train (121 unique) plus a few extras.
# Unknown adducts are hashed into the last bucket.
ADDUCT_VOCAB: tuple[str, ...] = (
    "[M+H]+",
    "[M-H]-",
    "[M+Na]+",
    "[M+K]+",
    "[M+NH4]+",
    "[M+Cl]-",
    "[M+CH2O2-H]-",
    "[2M+H]+",
    "[2M+Na]+",
    "[2M-H]-",
    "[2M+NH4]+",
    "[2M+K]+",
    "[2M+CH2O2-H]-",
    "[2M+C2H4O2-H]-",
    "[2M+Na-2H]-",
    "[M-H2O+H]+",
    "[M-2H2O+H]+",
    "[M]+",
    "[M]-",
    "[M+C2H4O2-H]-",
    "[M+2H]2+",
    "[M+3H]3+",
    "[M+H3O]+",
    "[M-H2O]+",
    "[M-H2O-H]-",
    "[M-CH3]-",
    "[M-2H]-",
    "[M+2Na-H]+",
    "[M+Ca]2+",
    "[M+Br]-",
    "[M+Li]+",
    "[M+C2F3O2]-",
    "[M+C2H4N]+",
    "[M-H5O3]+",
    "[3M+H]+",
    "[3M-H]-",
    "[3M+Na]+",
    "UNKNOWN",
)

ADDUCT2ID: dict[str, int] = {a: i for i, a in enumerate(ADDUCT_VOCAB)}
NUM_ADDUCTS: int = len(ADDUCT_VOCAB)


def adduct_to_id(adduct: str | None) -> int:
    if adduct is None:
        return ADDUCT2ID["UNKNOWN"]
    return ADDUCT2ID.get(str(adduct).strip(), ADDUCT2ID["UNKNOWN"])


def get_config(**overrides) -> Config:
    cfg = Config()
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise AttributeError(f"Unknown config field: {k}")
        setattr(cfg, k, v)
    cfg.artifact_dir.mkdir(parents=True, exist_ok=True)
    return cfg
