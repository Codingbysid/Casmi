"""Spectral-lock, Class 2 compete score, and tautomer-dedup gates."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chem import inchikey14_from_smiles  # noqa: E402
from src.config import get_config  # noqa: E402
from src.infer import (  # noqa: E402
    CLASS1_LOCK_MIN_PEAKS,
    _Hit,
    _is_class1_lock,
    _linear_score,
    _merge_tiers,
)
from src.ranker import dedup_inchikey14  # noqa: E402
from src.retrieval import modified_cosine  # noqa: E402


def _cosine(query_mz, query_int, cand_mz, cand_int, prec=400.0):
    q_mz = np.asarray(query_mz, dtype=np.float32)
    q_int = np.asarray(query_int, dtype=np.float32)
    q_mask = np.ones_like(q_mz)
    c_mz = np.asarray(cand_mz, dtype=np.float32).reshape(1, -1)
    c_int = np.asarray(cand_int, dtype=np.float32).reshape(1, -1)
    c_mask = np.ones_like(c_mz)
    scores, n_match, frac = modified_cosine(
        q_mz,
        q_int,
        q_mask,
        prec,
        c_mz,
        c_int,
        c_mask,
        np.asarray([prec], dtype=np.float32),
        0.05,
        return_n_match=True,
        return_intensity_fraction=True,
    )
    return float(scores[0]), float(n_match[0]), float(frac[0])


class SpectralGateTests(unittest.TestCase):
    def test_two_shared_peaks_cannot_score(self):
        mz = [50, 60, 100, 120, 140, 160, 180, 200]
        query_int = [1.0, 1.0, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02]
        score, n_match, frac = _cosine(mz, query_int, [50, 60], [1.0, 1.0])
        self.assertEqual(n_match, 2.0)
        self.assertGreater(frac, 0.15)
        self.assertEqual(score, 0.0)

    def test_low_intensity_fraction_is_zero(self):
        mz = [50, 70, 90, 110, 130, 150, 170, 190]
        query_int = [1.0, 1.0, 1.0, 1.0, 0.01, 0.01, 0.01, 0.01]
        score, n_match, frac = _cosine(mz, query_int, mz[4:], query_int[4:])
        self.assertEqual(n_match, 4.0)
        self.assertLess(frac, 0.15)
        self.assertEqual(score, 0.0)

    def test_coverage_damps_four_peaks_and_full_match_stays_high(self):
        mz = [50, 70, 90, 110, 130, 150, 170, 190]
        ones = [1.0] * 8
        partial, n_partial, frac_partial = _cosine(mz, ones, mz[:4], ones[:4])
        full, n_full, frac_full = _cosine(mz, ones, mz, ones)
        self.assertEqual(n_partial, 4.0)
        self.assertAlmostEqual(frac_partial, 0.5, places=5)
        self.assertGreater(partial, 0.0)
        self.assertLess(partial, 0.75)
        self.assertEqual(n_full, 8.0)
        self.assertAlmostEqual(frac_full, 1.0, places=5)
        self.assertGreater(full, 0.9)
        self.assertGreater(full, partial)


class ScoreCandidateTests(unittest.TestCase):
    def test_infer_gate_drops_two_peak_cosine(self):
        from src.infer import _score_candidates
        from src.retrieval import StructureIndex

        q_mz = np.array([50, 60, 100, 120, 140, 160, 180, 200], dtype=np.float32)
        q_int = np.array([1, 1, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02], dtype=np.float32)
        c_mz = np.zeros((1, 8), dtype=np.float32)
        c_int = np.zeros((1, 8), dtype=np.float32)
        c_mask = np.zeros((1, 8), dtype=np.float32)
        c_mz[0, :2] = [50, 60]
        c_int[0, :2] = [1, 1]
        c_mask[0, :2] = 1
        index = StructureIndex.build(
            ["CCO"],
            ["LFQSCWFLJHTTHZ"],
            np.array([46.04]),
            np.zeros((1, 8), dtype=np.uint8),
            peak_mz=c_mz,
            peak_intensity=c_int,
            peak_mask=c_mask,
        )
        feat = {
            "peak_mz": q_mz,
            "peak_intensity": q_int,
            "peak_mask": np.ones(8, dtype=np.float32),
            "precursor_mz": 400.0,
            "neutral_mass": 46.04,
            "adduct": "[M+H]+",
            "ionization_mode": "positive",
        }
        hits = _score_candidates(
            feat,
            [feat],
            np.zeros(8, dtype=np.float32),
            index,
            np.array([0], dtype=np.int64),
            get_config(fp_bits=8, fp_threshold=0.35),
            use_spectral=True,
        )
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].spec, 0.0)
        self.assertEqual(hits[0].n_match, 2.0)


class MergeGateTests(unittest.TestCase):
    def test_lock_requires_five_peaks(self):
        cfg = get_config()
        supported = _Hit("CCO", "LFQSCWFLJHTTHZ", 0.4, spec=0.91, tani=0.4, mass=1.0, n_match=8)
        sparse = _Hit("CCO", "LFQSCWFLJHTTHZ", 0.4, spec=0.95, tani=0.4, mass=1.0, n_match=2)
        natural = _Hit("Oc1ccccc1", "ISWSIDIOOBJBQZ", 0.2, spec=0.0, tani=0.85, mass=1.0, n_match=0)
        self.assertTrue(_is_class1_lock(supported, 0.75))
        self.assertFalse(_is_class1_lock(sparse, 0.75))
        self.assertGreaterEqual(CLASS1_LOCK_MIN_PEAKS, 5)
        locked = _merge_tiers([supported], [natural], [], cfg)
        self.assertEqual(locked[0].smiles, "CCO")
        unlocked = _merge_tiers([sparse], [natural], [], cfg)
        self.assertEqual(unlocked[0].smiles, "Oc1ccccc1")

    def test_class2_beats_noisy_spectral_decoy(self):
        cfg = get_config()
        decoy = _Hit("CCCC", "IJDNQMDRQITEOD", 0.1, spec=0.30, tani=0.40, mass=1.0, n_match=6)
        natural = _Hit("Oc1ccccc1", "ISWSIDIOOBJBQZ", 0.1, spec=0.0, tani=0.65, mass=1.0, n_match=0)
        self.assertAlmostEqual(_linear_score(decoy, cfg), 0.43, places=6)
        self.assertGreater(_linear_score(natural, cfg), _linear_score(decoy, cfg))
        order = _merge_tiers([decoy], [natural], [], cfg)
        self.assertEqual(order[0].smiles, "Oc1ccccc1")

    def test_strong_spectrum_still_beats_database_hit(self):
        cfg = get_config()
        spectral = _Hit("CCO", "LFQSCWFLJHTTHZ", 0.1, spec=0.90, tani=0.50, mass=1.0, n_match=8)
        natural = _Hit("Oc1ccccc1", "ISWSIDIOOBJBQZ", 0.1, spec=0.0, tani=0.65, mass=1.0, n_match=0)
        self.assertGreater(_linear_score(spectral, cfg), _linear_score(natural, cfg))


class TautomerDedupTests(unittest.TestCase):
    def test_keto_enol_share_one_slot(self):
        keto = "CC(=O)CC(=O)C"
        enol = "CC(=O)C=C(C)O"
        keto_key = inchikey14_from_smiles(keto)
        enol_key = inchikey14_from_smiles(enol)
        self.assertEqual(len(keto_key), 14)
        self.assertEqual(keto_key, enol_key)
        kept = dedup_inchikey14([keto, enol], np.array([0.2, 0.9]), top_k=25)
        self.assertEqual(kept, [enol])

    def test_stereo_isomers_share_the_connectivity_block(self):
        left = "C[C@H](O)CC"
        right = "C[C@@H](O)CC"
        self.assertEqual(inchikey14_from_smiles(left), inchikey14_from_smiles(right))
        kept = dedup_inchikey14(
            [left, right],
            np.array([0.4, 0.8]),
            inchikey14=["DIFFERENTKEY1", "DIFFERENTKEY2"],
            top_k=25,
        )
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0], right)


class TrainingObjectiveTests(unittest.TestCase):
    def test_focal_bce_is_the_only_spec2fp_loss(self):
        trainer = (ROOT / "src" / "tpu_trainer.py").read_text()
        self.assertIn("FocalBCELoss", trainer)
        self.assertNotIn("candidate_loss", trainer.lower())
        self.assertNotIn("soft_tanimoto", trainer.lower())
        self.assertFalse((ROOT / "src" / "gpu_training.py").exists())
        self.assertFalse((ROOT / "src" / "candidate_training.py").exists())


if __name__ == "__main__":
    unittest.main()
