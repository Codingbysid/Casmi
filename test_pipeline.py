#!/usr/bin/env python3
"""Smoke-test the CASMI pipeline on a 500-row train slice.

Checks:
  * adduct -> neutral mass conversion
  * static TPU shapes (top-128 peaks, 512 bins, 2048-bit fingerprints)
  * RDKit InChIKey14 canonicalization / skeleton dedup
  * retrieval + ranker + submission formatting
  * one short Spec2FP training step on CPU
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from src.chem import (
    formula_to_mass,
    has_rdkit,
    inchikey14_from_smiles,
    morgan_fingerprint,
    parse_adduct,
    precursor_to_neutral_mass,
)
from src.config import get_config
from src.data import (
    assert_static_shapes,
    collate_fixed,
    dataset_from_dataframe,
    dataset_from_structure_index,
    load_train_slice,
)
from src.infer import build_library_from_frame, predict_fingerprints, rank_molecules
from src.metrics import mrr_at_k
from src.preprocessing import featurize_aggregated, featurize_spectrum, mean_collision_energy
from src.ranker import dedup_inchikey14, predictions_to_submission
from src.tpu_trainer import get_device, train_spec2fp


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def test_adducts() -> None:
    proton = precursor_to_neutral_mass(101.007276, "[M+H]+")
    _assert(abs(proton - 100.0) < 1e-4, f"[M+H]+ mass {proton}")
    anion = precursor_to_neutral_mass(98.992724, "[M-H]-")
    _assert(abs(anion - 100.0) < 1e-4, f"[M-H]- mass {anion}")
    sodiated = precursor_to_neutral_mass(122.989218, "[M+Na]+")
    _assert(abs(sodiated - 100.0) < 0.002, f"[M+Na]+ mass {sodiated}")
    dimer = precursor_to_neutral_mass(201.007276, "[2M+H]+")
    _assert(abs(dimer - 100.0) < 0.002, f"[2M+H]+ mass {dimer}")
    formate = parse_adduct("[M+CH2O2-H]-")
    _assert(formate.charge == -1, formate)
    _assert(formate.n_molecules == 1, formate)
    formate_m = precursor_to_neutral_mass(144.998201, "[M+CH2O2-H]-")
    _assert(abs(formate_m - 100.0) < 5e-4, formate_m)
    cl_m = precursor_to_neutral_mass(134.969402, "[M+Cl]-")
    _assert(abs(cl_m - 100.0) < 5e-4, cl_m)
    from src.chem import ADDUCT_OFFSETS, plausible_neutral_masses

    for ad, off in ADDUCT_OFFSETS.items():
        mz = 400.0
        got = mz + off
        parse_m = precursor_to_neutral_mass(mz, ad)
        _assert(abs(got - parse_m) < 5e-4, f"{ad} offset {got} parse {parse_m}")
    masses = plausible_neutral_masses(400.0, "[M-H]-", ionization_mode="negative")
    _assert(any(abs(m - 401.007276) < 1e-3 for m in masses), masses[:8])
    pos = plausible_neutral_masses(400.0, "[M+H]+", ionization_mode="positive")
    _assert(any(abs(m - (400.0 - 1.007276)) < 1e-3 for m in pos), pos[:8])
    _assert(not any(abs(m - (400.0 + 1.007276)) < 1e-3 for m in pos), pos)
    from src.chem import POS_ADDUCTS

    _assert("[M+H]+" in POS_ADDUCTS and "[M+Na]+" in POS_ADDUCTS, POS_ADDUCTS)
    _assert(abs(ADDUCT_OFFSETS["[M+H]+"] + 1.007276) < 1e-9, ADDUCT_OFFSETS["[M+H]+"])
    _assert(abs(ADDUCT_OFFSETS["[M-H]-"] - 1.007276) < 1e-9, ADDUCT_OFFSETS["[M-H]-"])
    _assert(abs(ADDUCT_OFFSETS["[M+CH2O2-H]-"] + 44.998201) < 1e-9, ADDUCT_OFFSETS["[M+CH2O2-H]-"])
    _assert(abs(ADDUCT_OFFSETS["[M+Cl]-"] + 34.969402) < 1e-9, ADDUCT_OFFSETS["[M+Cl]-"])
    _assert(abs(ADDUCT_OFFSETS["[M+Na]+"] + 22.989218) < 1e-9, ADDUCT_OFFSETS["[M+Na]+"])
    _assert(abs(ADDUCT_OFFSETS["[M+NH4]+"] + 18.033826) < 1e-9, ADDUCT_OFFSETS["[M+NH4]+"])
    _assert(abs(ADDUCT_OFFSETS["[M+K]+"] + 38.963158) < 1e-9, ADDUCT_OFFSETS["[M+K]+"])
    print("[ok] adduct parsing")


def test_class2_library_hidden_test_safe() -> None:
    for name in ("build_class2_lotus.py", "build_class2_np.py"):
        src = (ROOT / "scripts" / name).read_text()
        _assert("_test_query_masses" not in src, f"{name} must not filter on test masses")
        _assert("cfg.test_path" not in src, src)
        _assert("MW_MIN" in src and "MW_MAX" in src, "generic MW window required")
        _assert("test.parquet" in src, f"{name} should mention the hidden-test trap")
    p = ROOT / "data" / "class2_candidates.parquet"
    _assert(p.exists(), p)
    df = pd.read_parquet(p, columns=["exact_mass"])
    _assert(int(df["exact_mass"].min()) < 80, float(df["exact_mass"].min()))
    _assert(float(df["exact_mass"].max()) > 1500.0, float(df["exact_mass"].max()))
    _assert(len(df) > 250_000, len(df))
    cols = pd.read_parquet(p, columns=["fp_packed"]).head(1)
    _assert("fp_packed" in cols.columns, list(cols.columns))
    from src.retrieval import load_external_structure_index

    ext = load_external_structure_index(str(p), get_config())
    _assert(ext is not None, "failed to load class2 parquet")
    _assert(ext.fp_packed is not None and ext.fp_packed.ndim == 2, getattr(ext, "fp_packed", None))
    _assert(ext.fingerprints.shape[1] == 0, ext.fingerprints.shape)
    _assert(len(ext.smiles) > 250_000, len(ext.smiles))
    from src.chem import formula_count_vector, formula_match_score, pack_fingerprints, unpack_fingerprints

    v = formula_count_vector("C6H6O")
    _assert(int(v[0]) == 6 and int(v[3]) == 1, v)
    _assert(formula_match_score(v, "C6H6O") > 0.95, formula_match_score(v, "C6H6O"))
    bits = np.zeros((2, 2048), dtype=np.uint8)
    bits[0, :8] = 1
    packed = pack_fingerprints(bits)
    back = unpack_fingerprints(packed, 2048)
    _assert(int(back[0, :8].sum()) == 8, back[0, :8])
    from src.model import Spec2FP

    m = Spec2FP(get_config(d_model=64, n_heads=4, n_transformer_layers=1, n_conv_channels=32))
    _assert(m.peak_in.in_features == 6 + 32, m.peak_in.in_features)
    from src.infer import CLASS1_LOCK_COSINE

    _assert(CLASS1_LOCK_COSINE == 0.75, CLASS1_LOCK_COSINE)
    infer_src = (ROOT / "src" / "infer.py").read_text()
    _assert("dynamic mass window lookup" in infer_src, "missing Class 2 searchsorted log")
    print("[ok] Class 2 library is MW-bounded, not test.parquet-filtered")


def test_analogs() -> None:
    from src.analogs import expand_smiles
    from src.chem import exact_mass_from_smiles

    parent = "c1ccccc1O"
    analogs = expand_smiles([parent], query_mass=exact_mass_from_smiles(parent), mass_ppm=50.0)
    _assert(len(analogs) > 0, analogs)
    print("[ok] analog expansion", analogs[:3])


def test_class2_merges_when_class1_is_full() -> None:
    from src.infer import (
        CLASS1_LOCK_COSINE,
        _Hit,
        _hits_to_smiles,
        _merge_hits,
        _merge_tiers,
        rank_molecules,
    )
    from src.chem import exact_mass_from_smiles, inchikey14_from_smiles, morgan_fingerprint
    from src.retrieval import StructureIndex

    c1 = [_Hit("C" * (i + 2), f"TRN{i:011d}", 0.18) for i in range(25)]
    phenol = "Oc1ccccc1"
    key = inchikey14_from_smiles(phenol)
    merged = _merge_hits(c1, [_Hit(phenol, key, 0.48)], top_k=25)
    smiles = [h.smiles for h in merged]
    _assert(phenol in smiles, smiles[:8])
    _assert(len(merged) == 25, len(merged))

    cfg = get_config()
    locked_hit = _Hit("CCO", "LFQSCWFLJHTTHZ", score=0.35, spec=0.91, tani=0.40, mass=1.0)
    noisy_c2 = _Hit(phenol, key, score=0.99, spec=0.0, tani=0.85, mass=1.0)
    _assert(locked_hit.spec >= CLASS1_LOCK_COSINE, locked_hit.spec)
    locked_order = _hits_to_smiles(_merge_tiers([locked_hit], [noisy_c2], [], cfg))
    _assert(locked_order[0] == "CCO", locked_order[:3])
    _assert(phenol in locked_order, locked_order[:5])

    near_lock = _Hit("CCO", "LFQSCWFLJHTTHZ", score=0.40, spec=0.76, tani=0.20, mass=1.0)
    near_order = _hits_to_smiles(_merge_tiers([near_lock], [noisy_c2], [], cfg))
    _assert(near_order[0] == "CCO", near_order[:3])

    # Low-cosine train decoys lose to high-Tanimoto Class 2, but are not dropped.
    weak_decoys = [
        _Hit(f"C{'C' * (i + 1)}", f"TRN{i:011d}XX", score=0.18, spec=0.15, tani=0.10, mass=1.0)
        for i in range(30)
    ]
    competed = _merge_tiers(weak_decoys, [noisy_c2], [], cfg)
    competed_smi = _hits_to_smiles(competed)
    _assert(competed_smi[0] == phenol, competed_smi[:5])
    _assert(phenol in competed_smi[:5], competed_smi[:8])

    analog = _Hit("CCN", "XFNJVJPHKREAPP", score=0.05, spec=0.0, tani=0.05, mass=0.3)
    padded = _merge_tiers([], [], [analog], cfg)
    _assert(padded[0].smiles == "CCN", _hits_to_smiles(padded)[:3])

    analog_good = _Hit("CCN", "XFNJVJPHKREAPP", score=0.40, spec=0.0, tani=0.80, mass=1.0)
    pooled = _hits_to_smiles(_merge_tiers(weak_decoys, [noisy_c2], [analog_good], cfg))
    _assert(pooled[0] == phenol, pooled[:5])
    _assert(pooled[1] == "CCN", pooled[:5])
    locked_beats_analog = _hits_to_smiles(_merge_tiers([locked_hit], [noisy_c2], [analog_good], cfg))
    _assert(locked_beats_analog[0] == "CCO", locked_beats_analog[:3])
    print("[ok] Class 1 cosine lock; weak train decoys compete with Class 2 / analogs")

    train_smi = ["CCO"] + [f"C{'C' * i}O" for i in range(2, 8)]
    train_keys = [inchikey14_from_smiles(s) or s for s in train_smi]
    train_mass = np.array([exact_mass_from_smiles(s) for s in train_smi], dtype=np.float64)
    train_fp = np.stack([morgan_fingerprint(s, n_bits=2048) for s in train_smi])
    train_index = StructureIndex.build(train_smi, train_keys, train_mass, train_fp)
    ext_index = StructureIndex.build(
        [phenol],
        [key],
        np.array([exact_mass_from_smiles(phenol)], dtype=np.float64),
        np.stack([morgan_fingerprint(phenol, n_bits=2048)]),
    )
    feat = {
        "neutral_mass": float(exact_mass_from_smiles(phenol)),
        "precursor_mz": float(exact_mass_from_smiles(phenol) + 1.007276),
        "adduct": "[M+H]+",
        "ionization_mode": "positive",
        "adduct_id": 0,
        "peak_mz": np.zeros(128, dtype=np.float32),
        "peak_intensity": np.zeros(128, dtype=np.float32),
        "peak_mask": np.zeros(128, dtype=np.float32),
    }
    cfg = get_config()
    ranked = rank_molecules(
        {"m0": feat},
        {"m0": morgan_fingerprint(phenol, n_bits=cfg.fp_bits).astype(np.float32)},
        train_index,
        cfg,
        external_index=ext_index,
    )
    _assert(phenol in ranked["m0"], ranked["m0"][:8])
    print("[ok] Class 2 merge into a full Class 1 list")


def test_ragged_collision_energy(cfg) -> None:
    ragged = [np.array([20.0, 40.0, 60.0]), np.array([40.0]), np.array([60.0])]
    mean = mean_collision_energy(ragged)
    _assert(abs(mean - 44.0) < 1e-3, mean)
    _assert(abs(mean_collision_energy("20,40,60") - 40.0) < 1e-3, mean_collision_energy("20,40,60"))
    rows = [
        {
            "ms2_mzs": np.array([100.0, 120.0, 150.0], dtype=np.float64),
            "ms2_normalized_intensities": np.array([0.2, 1.0, 0.4], dtype=np.float64),
            "precursor_mz": 200.1,
            "adduct": "[M+H]+",
            "collision_energy_ev": np.array([20.0, 40.0, 60.0]),
            "ionization_mode": "positive",
        },
        {
            "ms2_mzs": np.array([101.0, 121.0], dtype=np.float64),
            "ms2_normalized_intensities": np.array([0.3, 1.0], dtype=np.float64),
            "precursor_mz": 200.1,
            "adduct": "[M+H]+",
            "collision_energy_ev": np.array([40.0]),
            "ionization_mode": "positive",
        },
        {
            "ms2_mzs": np.array([99.0, 119.0, 149.0], dtype=np.float64),
            "ms2_normalized_intensities": np.array([0.1, 0.8, 1.0], dtype=np.float64),
            "precursor_mz": 200.1,
            "adduct": "[M+H]+",
            "collision_energy_ev": np.array([60.0]),
            "ionization_mode": "positive",
        },
    ]
    feat = featurize_aggregated(rows, cfg=cfg)
    _assert(feat["peak_mz"].shape == (cfg.top_n_peaks,), feat["peak_mz"].shape)
    print("[ok] ragged collision-energy aggregation")


def test_inchikey(smiles: str, expected14: str) -> None:
    _assert(has_rdkit(), "RDKit is required for InChIKey14 canonicalization")
    key = inchikey14_from_smiles(smiles)
    _assert(len(key) == 14, f"InChIKey14 length {key}")
    _assert(key == expected14, f"{key} != {expected14} for {smiles}")
    # Stereo / tautomer-style duplicates must collapse.
    dup = dedup_inchikey14([smiles, smiles, smiles + "C"], np.array([0.9, 0.8, 0.1]), top_k=25)
    _assert(dup[0] == smiles, dup)
    print("[ok] InChIKey14 canonicalization / dedup")


def main() -> None:
    test_adducts()
    test_class2_library_hidden_test_safe()
    test_analogs()
    test_class2_merges_when_class1_is_full()
    cfg = get_config(
        batch_size=16,
        num_epochs=1,
        max_train_spectra=500,
        log_every=10,
        d_model=64,
        n_heads=4,
        n_transformer_layers=1,
        n_conv_channels=32,
        train_time_limit_s=180.0,
        max_mass_candidates=512,
    )
    test_ragged_collision_energy(cfg)
    from src.ranker import official_molecule_ids, build_submission_frame, write_submission_csv, validate_submission
    dummy = build_submission_frame(official_molecule_ids(cfg), {})
    dummy_path = ROOT / "artifacts" / "format_check_submission.csv"
    dummy_path.parent.mkdir(parents=True, exist_ok=True)
    write_submission_csv(dummy, dummy_path)
    validate_submission(dummy_path, cfg.sample_submission_path)
    _assert(len(dummy) == 400, len(dummy))
    _assert((dummy["smiles"] == "CCO").all(), "fallback smiles")
    print("[ok] official 400-row submission format")
    print(f"[info] train parquet: {cfg.train_path}")
    df = load_train_slice(cfg, 500)
    _assert(len(df) == 500, f"expected 500 rows, got {len(df)}")
    print(f"[info] slice columns: {list(df.columns)}")
    print(f"[info] adducts: {df['adduct'].value_counts().head(8).to_dict()}")

    feat0 = featurize_spectrum(
        df.iloc[0]["ms2_mzs"],
        df.iloc[0]["ms2_normalized_intensities"],
        float(df.iloc[0]["precursor_mz"]),
        df.iloc[0]["adduct"],
        cfg=cfg,
        collision_energy=df.iloc[0]["collision_energy_ev"],
        ionization_mode=df.iloc[0]["ionization_mode"],
    )
    _assert(feat0["peak_mz"].shape == (cfg.top_n_peaks,), feat0["peak_mz"].shape)
    _assert(feat0["peak_intensity"].shape == (cfg.top_n_peaks,), feat0["peak_intensity"].shape)
    _assert(feat0["peak_nl"].shape == (cfg.top_n_peaks,), feat0["peak_nl"].shape)
    _assert(feat0["peak_mask"].shape == (cfg.top_n_peaks,), feat0["peak_mask"].shape)
    _assert(feat0["binned"].shape == (cfg.n_mz_bins,), feat0["binned"].shape)
    _assert(feat0["precursor_feat"].shape == (cfg.precursor_feat_dim,), feat0["precursor_feat"].shape)
    _assert(int(feat0["peak_mask"].sum()) > 0, "all peaks padded")
    pred_mass = float(feat0["neutral_mass"])
    formula_mass = formula_to_mass(str(df.iloc[0]["molecular_formula"]))
    print(f"[info] precursor={df.iloc[0]['precursor_mz']} adduct={df.iloc[0]['adduct']}")
    print(f"[info] derived_neutral={pred_mass:.4f} formula_mass={formula_mass:.4f}")
    if formula_mass > 0:
        ppm = abs(pred_mass - formula_mass) / formula_mass * 1e6
        print(f"[info] mass error {ppm:.2f} ppm")

    test_inchikey(str(df.iloc[0]["normalized_smiles"]), str(df.iloc[0]["inchikey14"]))

    ds = dataset_from_dataframe(df, cfg)
    batch = collate_fixed([ds[i] for i in range(min(8, len(ds)))])
    assert_static_shapes(batch, cfg)
    _assert(tuple(batch["peak_mz"].shape) == (min(8, len(ds)), cfg.top_n_peaks), batch["peak_mz"].shape)
    _assert(tuple(batch["fingerprint"].shape) == (min(8, len(ds)), cfg.fp_bits), batch["fingerprint"].shape)
    _assert(tuple(batch["binned"].shape) == (min(8, len(ds)), cfg.n_mz_bins), batch["binned"].shape)
    print("[ok] static shapes")

    # Query structures that are actually in the library (self-retrieval).
    lib_df = df.iloc[:450].reset_index(drop=True)
    query_df = df.iloc[:40].reset_index(drop=True)
    index = build_library_from_frame(lib_df, cfg)
    _assert(len(index.smiles) > 0, "empty library")
    _assert(index.peak_mz.shape[1] == cfg.top_n_peaks, index.peak_mz.shape)
    _assert(index.fingerprints.shape[1] == cfg.fp_bits, index.fingerprints.shape)
    print(f"[ok] library n={len(index.smiles)}")

    lib_ds = dataset_from_structure_index(index, cfg, max_n=32, seed=0)
    _assert(len(lib_ds) == min(32, len(index.smiles)), len(lib_ds))
    assert_static_shapes(lib_ds[0], cfg)
    print("[ok] dataset_from_structure_index", len(lib_ds))

    print("[info] short Spec2FP fit on CPU/GPU")
    device, use_xla = get_device(prefer_xla=False)
    print(f"[info] device={device} xla={use_xla}")
    model = train_spec2fp(ds, cfg, device=device, time_limit_s=180.0)
    model.eval()

    q_feats = []
    for i in range(len(query_df)):
        q_feats.append(
            featurize_spectrum(
                query_df.iloc[i]["ms2_mzs"],
                query_df.iloc[i]["ms2_normalized_intensities"],
                float(query_df.iloc[i]["precursor_mz"]),
                query_df.iloc[i]["adduct"],
                cfg=cfg,
                collision_energy=query_df.iloc[i]["collision_energy_ev"],
                ionization_mode=query_df.iloc[i]["ionization_mode"],
            )
        )
    fps = predict_fingerprints(model, q_feats, cfg, device=device)
    _assert(fps.shape == (len(q_feats), cfg.fp_bits), fps.shape)

    mol_feats = {f"q{i}": q_feats[i] for i in range(len(q_feats))}
    pred_map = {f"q{i}": fps[i] for i in range(len(q_feats))}
    ranked = rank_molecules(mol_feats, pred_map, index, cfg)
    pred_lists = [ranked[f"q{i}"] for i in range(len(q_feats))]
    for smiles_list in pred_lists:
        _assert(len(smiles_list) == 25, len(smiles_list))
        keys = [inchikey14_from_smiles(s) for s in smiles_list]
        _assert(len(keys) == len(set(keys)), "duplicate InChIKey14 in ranked list")

    mrr = mrr_at_k(pred_lists, true_keys=query_df["inchikey14"].astype(str).tolist(), k=25)
    print(f"[ok] ranking + dedup  MRR@25 self-retrieval (n=40, library=450): {mrr:.4f}")
    _assert(mrr >= 0.2, f"self-retrieval MRR@25 too low: {mrr}")

    # --- Prefix-conditioned SMILES decoder (Class 3) ---
    from src.models.smiles_decoder import count_parameters, decoder_from_config, sample_autoregressive
    from src.ranker import merge_unique_smiles
    from src.chem_tokenizer import SMILESTokenizer
    from src.smiles_tokenizer import SmilesTokenizer, tokenize_smiles

    raw = SMILESTokenizer()
    stereo = raw.tokenize("C[C@@H](O)C(=O)O")
    _assert("[C@@H]" in stereo, stereo)
    _assert("[" not in stereo and "]" not in stereo, stereo)
    _assert(raw.tokenize("ClCCBr") == ["Cl", "C", "C", "Br"], raw.tokenize("ClCCBr"))
    _assert("%10" in raw.tokenize("C%10CCCC%10"), raw.tokenize("C%10CCCC%10"))
    _assert(raw.tokenize("[O-]C(=O)[NH3+]") == ["[O-]", "C", "(", "=", "O", ")", "[NH3+]"], raw.tokenize("[O-]C(=O)[NH3+]"))
    print("[ok] NP tokenizer keeps [C@@H], Cl/Br, %10, [O-]/[NH3+] atomic")

    from src.smiles_train import dataset_from_structures, train_smiles_decoder

    smi0 = str(df.iloc[0]["normalized_smiles"])
    toks = tokenize_smiles(smi0)
    _assert(len(toks) > 0, "tokenizer produced no tokens")
    tok = SmilesTokenizer.default()
    ids = tok.encode(smi0, cfg.smiles_max_len)
    _assert(len(ids) == cfg.smiles_max_len, len(ids))
    _assert(ids[0] == tok.bos_id, ids[0])
    _assert(tok.eos_id in ids, "missing EOS")
    roundtrip = tok.decode(ids)
    _assert(len(roundtrip) > 0, roundtrip)
    print(f"[ok] SMILES tokenizer vocab={tok.vocab_size} roundtrip={roundtrip[:80]}")

    dec = decoder_from_config(tok, cfg)
    n_params = count_parameters(dec)
    print(f"[info] decoder params={n_params/1e6:.2f}M")
    _assert(2_000_000 < n_params < 12_000_000, n_params)
    cpu = torch.device("cpu")
    dec = dec.to(cpu)
    bsz = 4
    fp = torch.zeros(bsz, cfg.fp_bits)
    mass = torch.full((bsz, 1), 300.0)
    adduct = torch.zeros(bsz, dtype=torch.long)
    tgt = torch.tensor([tok.encode(smi0, cfg.smiles_max_len) for _ in range(bsz)], dtype=torch.long)
    pad = tgt == tok.pad_id
    logits = dec(fp, mass, adduct, tgt, tgt_mask=pad)
    _assert(tuple(logits.shape) == (bsz, cfg.smiles_max_len, tok.vocab_size), logits.shape)
    print("[ok] decoder static logits", tuple(logits.shape))

    # Shrink for the smoke-test fit; the 256-d/4-layer model is what Kaggle trains.
    cfg.smiles_d_model = 64
    cfg.smiles_num_layers = 2
    cfg.smiles_nhead = 4
    cfg.smiles_dim_feedforward = 256
    cfg.smiles_num_epochs = 1
    cfg.smiles_batch_size = 16
    cfg.log_every = 20

    smi_ds = dataset_from_structures(
        df["normalized_smiles"].astype(str).tolist()[:200],
        np.array([formula_to_mass(f) for f in df["molecular_formula"].astype(str).tolist()[:200]], dtype=np.float32),
        np.stack(
            [
                morgan_fingerprint(s, n_bits=cfg.fp_bits)
                for s in df["normalized_smiles"].astype(str).tolist()[:200]
            ]
        ).astype(np.float32),
        tok,
        cfg,
    )
    _assert(tuple(smi_ds[0]["tgt_tokens"].shape) == (cfg.smiles_max_len,), smi_ds[0]["tgt_tokens"].shape)
    cfg.smiles_num_epochs = 1
    cfg.smiles_batch_size = 16
    cfg.log_every = 20
    # Keep decoder training on CPU: MPS + weight-tied embedding/head is flaky,
    # and submission-time sampling is specified to run on CPU anyway.
    dec = train_smiles_decoder(smi_ds, tok, cfg, device=cpu, time_limit_s=120.0)
    dec = dec.to(cpu).eval()
    sampled = sample_autoregressive(
        dec,
        torch.rand(8, cfg.fp_bits),
        torch.full((8, 1), float(feat0["neutral_mass"])),
        torch.zeros(8, dtype=torch.long),
        tok,
        max_len=40,
        temp=0.8,
        top_p=0.9,
    )
    _assert(len(sampled) == 8, len(sampled))
    decoded = [tok.decode(s) for s in sampled]
    print("[ok] sampled SMILES", decoded[:3])

    # Force Class 3: empty-ish library mass miss + decoder fallback.
    from src.infer import rank_molecules as rank_mols

    cfg.smiles_num_samples = 8
    cfg.smiles_gen_max_len = 32
    fake_feat = dict(q_feats[0])
    fake_feat["neutral_mass"] = 12.0  # no train hit at 15 ppm
    fake_feat["precursor_mz"] = 12.0
    fake_map = {"c3": fps[0]}
    fake_feats = {"c3": fake_feat}
    class3 = rank_mols(
        fake_feats,
        fake_map,
        index,
        cfg,
        decoder=dec,
        tokenizer=tok,
    )
    _assert("c3" in class3, class3)
    merged = merge_unique_smiles(["CCO"], class3["c3"], top_k=25)
    _assert(merged[0] == "CCO", merged)
    print(f"[ok] Class 3 fallback returned {len(class3['c3'])} de novo SMILES")

    sub = predictions_to_submission([f"q{i}" for i in range(len(pred_lists))], pred_lists)
    _assert(list(sub.columns) == ["molecule_id", "smiles"], list(sub.columns))
    _assert(sub["smiles"].str.split(";").map(len).max() <= 25, "more than 25 SMILES")
    _assert(not (sub["smiles"].astype(str).str.strip() == "").any(), "empty smiles")

    from src.ranker import build_submission_frame, sanitize_guesses, validate_submission, write_submission_csv

    _assert(sanitize_guesses(["", "nan", "CCO", "CCO", "not-a-mol", "CCN"])[0] == "CCO", "sanitize first")
    _assert(len(sanitize_guesses(["C"] * 40)) <= 25, "sanitize cap")
    empty_row = build_submission_frame(["m_x"], {"m_x": []})
    _assert(empty_row.iloc[0]["smiles"] == "CCO", empty_row)

    out = ROOT / "artifacts" / "test_pipeline_submission.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_submission_csv(sub, out)
    validate_submission(out)
    print(f"[ok] wrote {out}")
    test_v6_packed_fp_neighbors_merge_shift()
    print("[ok] pipeline smoke test passed")


def test_v6_packed_fp_neighbors_merge_shift() -> None:
    from src.analogs import MASS_SHIFTS, expand_smiles, shifted_parent_masses
    from src.calibration import load_calibration
    from src.chem import exact_mass_from_smiles, pack_fingerprints, unpack_fingerprints
    from src.infer import _fingerprints_for, predict_fingerprints
    from src.neighbors import SpectralNeighborIndex, blend_fingerprints
    from src.retrieval import StructureIndex

    cfg = get_config()
    smiles = ["CCO", "Oc1ccccc1", "CCN"]
    keys = [inchikey14_from_smiles(s) or s for s in smiles]
    masses = np.array([exact_mass_from_smiles(s) for s in smiles], dtype=np.float64)
    fps = np.stack([morgan_fingerprint(s, n_bits=cfg.fp_bits) for s in smiles])
    packed = pack_fingerprints(fps)
    packed_index = StructureIndex.build(
        smiles,
        keys,
        masses,
        np.zeros((len(smiles), 0), dtype=np.uint8),
        fp_packed=packed,
    )
    got = _fingerprints_for(packed_index, np.array([1]), cfg)
    _assert(got.shape == (1, cfg.fp_bits), got.shape)
    _assert(int(got[0, :].sum()) == int(fps[1].sum()), (got[0].sum(), fps[1].sum()))
    back = unpack_fingerprints(packed, cfg.fp_bits)
    _assert(int(back[1].sum()) == int(fps[1].sum()), back[1].sum())

    peak_mz = np.zeros((3, cfg.top_n_peaks), dtype=np.float32)
    peak_int = np.zeros((3, cfg.top_n_peaks), dtype=np.float32)
    peak_mask = np.zeros((3, cfg.top_n_peaks), dtype=np.float32)
    peak_mz[:, 0] = np.array([45.0, 93.0, 58.0], dtype=np.float32)
    peak_int[:, 0] = 1.0
    peak_mask[:, 0] = 1.0
    spec_index = StructureIndex.build(
        smiles, keys, masses, fps, peak_mz=peak_mz, peak_intensity=peak_int, peak_mask=peak_mask
    )
    nbr = SpectralNeighborIndex.from_structure_index(spec_index, cfg)
    nfp = nbr.neighbor_fp(np.ones(cfg.n_mz_bins, dtype=np.float32), "positive", k=2)
    _assert(nfp.shape == (cfg.fp_bits,), nfp.shape)
    blended = blend_fingerprints(np.full(cfg.fp_bits, 0.2, dtype=np.float32), nfp, alpha=0.5)
    _assert(blended.shape == (cfg.fp_bits,), blended.shape)

    logits = predict_fingerprints(None, [], cfg, return_logits=True)
    _assert(logits.shape[0] == 0, logits.shape)
    none_logits = predict_fingerprints(None, [{"binned": np.zeros(cfg.n_mz_bins)}], cfg, return_logits=True)
    _assert(none_logits.shape == (1, cfg.fp_bits), none_logits.shape)

    cal = load_calibration()
    xs = np.linspace(0.0, 1.0, 11)
    ps = cal.p_cosine(xs)
    _assert(np.all(np.diff(ps) >= -1e-6), ps)
    _assert(float(cal.p_cosine(0.9)) > float(cal.p_cosine(0.5)), (cal.p_cosine(0.9), cal.p_cosine(0.5)))
    s78 = float(cal.or_score(0.78, 0.2, 1.0))
    s_c2 = float(cal.or_score(0.0, 0.40, 1.0))
    _assert(s78 > s_c2, (s78, s_c2))

    deltas = {name: delta for name, delta in MASS_SHIFTS}
    _assert(abs(deltas["hexose"] - 162.0528) < 1e-3, deltas["hexose"])
    parents = shifted_parent_masses([200.0])
    _assert(any(abs(p - (200.0 + 162.052823)) < 1e-4 for p in parents), parents[:8])
    phenol = "Oc1ccccc1"
    qmass = exact_mass_from_smiles(phenol) + deltas["hexose"]
    products = expand_smiles([phenol], query_mass=qmass, mass_ppm=15.0, in_window_only=True, max_total=40)
    in_win = [
        s
        for s in products
        if exact_mass_from_smiles(s) > 0
        and abs(exact_mass_from_smiles(s) - qmass) / qmass * 1e6 <= 15.0
    ]
    _assert(len(in_win) > 0, products[:8])

    nb = ROOT / "kaggle_submission.ipynb"
    if nb.exists():
        _assert(nb.stat().st_size < 1_000_000, nb.stat().st_size)
    print("[ok] V6 packed-fp / neighbor / calibrated merge / mass-shift")


if __name__ == "__main__":
    main()
