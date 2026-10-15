"""Separate experiments: observed precursor m/z, and the official metric identity.

Neither is enabled in required-reranker-v1.
"""

from __future__ import annotations

import inspect
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chem import inchikey14_from_smiles, official_metric_inchikey14  # noqa: E402
from src.config import get_config  # noqa: E402
from src.infer import _score_candidates  # noqa: E402
from src.pipeline import PRECURSOR_EXPERIMENT_ID, IDENTITY_EXPERIMENT_ID, STRATEGY_ID  # noqa: E402
from src.retrieval import StructureIndex  # noqa: E402


def _index(precursor_mz):
    query = np.array([100, 110, 120, 130], dtype=np.float32)
    cand = np.array([50, 60, 70, 80], dtype=np.float32)
    c_mz = np.zeros((1, 4), np.float32)
    c_int = np.zeros((1, 4), np.float32)
    c_mask = np.ones((1, 4), np.float32)
    c_mz[0] = cand
    c_int[0] = 1
    return StructureIndex.build(
        ["CCO"],
        ["LFQSCWFLJHTTHZ"],
        np.array([450.0]),
        np.zeros((1, 8), np.uint8),
        peak_mz=c_mz,
        peak_intensity=c_int,
        peak_mask=c_mask,
        precursor_mz=np.asarray(precursor_mz, dtype=np.float64),
    )


def _feat():
    mz = np.array([100, 110, 120, 130], dtype=np.float32)
    return {
        "peak_mz": mz,
        "peak_intensity": np.ones(4, np.float32),
        "peak_mask": np.ones(4, np.float32),
        "precursor_mz": 500.0,
        "neutral_mass": 450.0,
        "adduct": "[M+H]+",
        "ionization_mode": "positive",
    }


class PrecursorExperimentTests(unittest.TestCase):
    def test_default_path_still_passes_neutral_mass(self):
        cfg = get_config(fp_bits=8, fp_threshold=0.0)
        self.assertFalse(cfg.use_observed_precursor)
        hits = _score_candidates(_feat(), [_feat()], np.zeros(8, np.float32), _index([500.0]), np.array([0]), cfg, use_spectral=True)
        self.assertGreater(hits[0].spec, 0.0)
        self.assertEqual(PRECURSOR_EXPERIMENT_ID, "precursor-mz-shift-v1")
        self.assertNotEqual(PRECURSOR_EXPERIMENT_ID, STRATEGY_ID)

    def test_opt_in_uses_observed_precursor_and_does_not_invent(self):
        cfg = get_config(fp_bits=8, fp_threshold=0.0, use_observed_precursor=True)
        observed = _score_candidates(
            _feat(), [_feat()], np.zeros(8, np.float32), _index([500.0]), np.array([0]), cfg, use_spectral=True
        )
        self.assertEqual(observed[0].spec, 0.0)
        with self.assertRaises(RuntimeError) as ctx:
            _score_candidates(
                _feat(), [_feat()], np.zeros(8, np.float32), _index([np.nan]), np.array([0]), cfg, use_spectral=True
            )
        self.assertIn("invent", str(ctx.exception))


class OfficialIdentityTests(unittest.TestCase):
    def test_metric_sequence_fixtures(self):
        left, right = "C[C@H](O)CC", "C[C@@H](O)CC"
        keto, enol = "CC(=O)CC(=O)C", "CC(=O)C=C(C)O"
        self.assertEqual(official_metric_inchikey14(left), official_metric_inchikey14(right))
        self.assertEqual(len(official_metric_inchikey14(left)), 14)
        self.assertEqual(official_metric_inchikey14(keto), official_metric_inchikey14(enol))
        self.assertEqual(official_metric_inchikey14("not a smiles"), "")
        self.assertEqual(official_metric_inchikey14(""), "")
        source = inspect.getsource(official_metric_inchikey14)
        self.assertLess(source.index("RemoveStereochemistry"), source.index("Canonicalize"))
        self.assertIn("MolToInchiKey", Path(official_metric_inchikey14.__code__.co_filename).read_text())
        # Production dedup is still the existing helper. A difference here is
        # the separate official-inchikey14-v14 experiment, not this scored run.
        self.assertIn("inchikey14_from_smiles", inspect.getsource(__import__("src.infer", fromlist=["_skeleton_key"])._skeleton_key))
        self.assertNotIn("official_metric_inchikey14", inspect.getsource(__import__("src.infer", fromlist=["_skeleton_key"])._skeleton_key))
        self.assertEqual(IDENTITY_EXPERIMENT_ID, "official-inchikey14-v14")

    def test_isolated_process_matches_in_process(self):
        code = (
            "from src.chem import official_metric_inchikey14\n"
            "from rdkit import Chem\n"
            "print(Chem.rdBase.rdkitVersion)\n"
            "print(official_metric_inchikey14('C[C@H](O)CC'))\n"
            "print(official_metric_inchikey14('CC(=O)C=C(C)O'))\n"
            "print(official_metric_inchikey14('not a smiles'))\n"
        )
        env = dict(**os_environ())
        env["PYTHONPATH"] = str(ROOT)
        proc = subprocess.run(
            [sys.executable, "-c", code], cwd=ROOT, env=env, text=True, capture_output=True, check=True
        )
        version, stereo, tautomer, invalid = proc.stdout.splitlines()
        self.assertTrue(version)
        self.assertEqual(stereo, official_metric_inchikey14("C[C@H](O)CC"))
        self.assertEqual(tautomer, official_metric_inchikey14("CC(=O)C=C(C)O"))
        self.assertEqual(invalid, "")
        # Local RDKit is not silently replaced with the scorer pin.
        self.assertNotIn("pip install", proc.stdout)
        _ = inchikey14_from_smiles  # production helper remains importable


def os_environ():
    import os
    return os.environ.copy()
