"""Gates for the unlocked-pool reranker experiment.

Leakage, locked-prefix preservation, reranker integration, disabled-mode
equivalence with the reference merge, lazy tautomer dedup, checkpoint reuse
guard, strict submission validation, bootstrap determinism, booster
determinism, fit-subset reproducibility, and notebook/source synchronisation.
Everything here runs on synthetic data in seconds; nothing is fitted on
competition data.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import validation as V  # noqa: E402
from src.chem import (  # noqa: E402
    exact_mass_from_smiles,
    heavy_atom_graph_key,
    inchikey14_from_smiles,
    morgan_fingerprint,
    pack_fingerprints,
)
from src.config import get_config  # noqa: E402
from src.data import fitting_subset_indices  # noqa: E402
from src.infer import (  # noqa: E402
    _emit_unique,
    _Hit,
    _locked_hits,
    _merge_hits,
    _merge_tiers,
    _skeleton_key,
    _unlocked_pool,
    rank_molecules,
)
from src.preprocessing import featurize_spectrum  # noqa: E402
from src.rerank import (  # noqa: E402
    FEATURE_NAMES,
    fit_reranker,
    load_reranker,
    save_reranker,
    window_feature_matrix,
)
from src.retrieval import StructureIndex  # noqa: E402

KETO, ENOL = "CC(=O)CC(=O)C", "CC(=O)C=C(C)O"  # one tautomer-canonical skeleton
PYRIDONE, HYDROXYPYRIDINE = "O=C1NC=CC=C1", "Oc1ccccn1"  # another pair

LIBRARY = [
    KETO, ENOL, PYRIDONE, HYDROXYPYRIDINE, "CCCCO", "c1ccccc1O", "CCOC(=O)C", "c1ccncc1",
    "OCC(O)CO", "C1CCOC1", "c1ccc2ccccc2c1", "CCCCCCCC", "CC(=O)OC", "OCCN", "c1ccoc1",
    "CC(=O)N", "Cc1ccccc1", "c1ccccc1C(=O)O", "OC(=O)CC(=O)O", "CCCC(=O)O", "c1ccc(cc1)N",
    "OCc1ccccc1", "CCCCCO", "OC1CCCCC1", "COc1ccccc1", "CC(=O)c1ccccc1", "OCCO", "CCCCCCO",
]
EXTERNAL = [
    "CCCCCCCCCCCO", "c1ccc2[nH]ccc2c1", "OC(=O)c1ccccc1O", "CCCCC(=O)O", "CC(C)=CCO", "CCCCCCCCCCCC",
    # isomers of the three end-to-end queries (keto C5H8O2, butanol C4H10O, glycerol C3H8O3)
    "O=C1CCCCO1", "C=CC(=O)OCC", "CC(=O)C(=O)CC", "OC1CCC(=O)C1", "CCC(=O)OC=C",
    "CC(C)CO", "CCOCC", "CC(C)(C)O", "CCC(C)O",
    "OCCOCO", "CC(O)(O)CO", "COC(O)CO",
]


def _spectrum(rng, mass, n=8):
    prec = mass + 1.007276
    mz = np.sort(rng.uniform(15, max(prec - 3, 20), size=n))
    inten = rng.uniform(0.1, 1.0, size=n)
    return mz, inten, prec


def build_index(cfg, smiles, *, raw_keys=None, seed=0):
    rng = np.random.default_rng(seed)
    n = len(smiles)
    keys = raw_keys or [inchikey14_from_smiles(s) for s in smiles]
    masses = np.array([exact_mass_from_smiles(s) for s in smiles])
    fps = np.stack([morgan_fingerprint(s, n_bits=cfg.fp_bits, radius=cfg.morgan_radius) for s in smiles])
    top_n = cfg.top_n_peaks
    arrays = {k: np.zeros((n, top_n), np.float32) for k in ("peak_mz", "peak_intensity", "peak_mask", "peak_nl")}
    pfeat = np.zeros((n, cfg.precursor_feat_dim), np.float32)
    add_id = np.zeros(n, np.int64)
    feats = []
    for i in range(n):
        mz, inten, prec = _spectrum(rng, masses[i])
        f = featurize_spectrum(mz, inten, prec, "[M+H]+", cfg=cfg, collision_energy=20.0, ionization_mode="positive")
        feats.append(f)
        for k in arrays:
            arrays[k][i] = f[k]
        pfeat[i] = f["precursor_feat"]
        add_id[i] = f["adduct_id"]
    index = StructureIndex.build(
        smiles, keys, masses, fps, formulas=[""] * n, ionization_mode=["positive"] * n,
        precursor_feat=pfeat, adduct_id=add_id, **arrays,
    )
    return index, feats


def build_external(cfg, smiles):
    packed = pack_fingerprints(np.stack([morgan_fingerprint(s, n_bits=cfg.fp_bits, radius=cfg.morgan_radius) for s in smiles]))
    return StructureIndex.build(
        smiles, [inchikey14_from_smiles(s) for s in smiles],
        np.array([exact_mass_from_smiles(s) for s in smiles]), np.zeros((len(smiles), 0), np.uint8), fp_packed=packed,
    )


def random_pool(rng, *, n_class1=6, n_class2=10, n_analog=3, lock_first=False):
    """Random hits with tautomer pairs, raw keys distinct per SMILES string."""
    mols = [KETO, ENOL, PYRIDONE, HYDROXYPYRIDINE, "CCCCO", "c1ccccc1O", "CCOC(=O)C", "c1ccncc1", "OCC(O)CO",
            "C1CCOC1", "CCCCCCCC", "CC(=O)OC", "OCCN", "c1ccoc1", "CC(=O)N"]

    def hit(src):
        smi = mols[int(rng.integers(len(mols)))]
        spec = float(rng.uniform(0.0, 0.7)) if src == 1 else 0.0
        return _Hit(
            smi, f"RAW_{smi}", float(rng.uniform(0, 1)), spec=spec, tani=float(rng.uniform(0.2, 0.95)),
            mass=float(rng.uniform(0.5, 1.0)), n_match=float(rng.integers(0, 9)) if src == 1 else 0.0,
            frac=float(rng.uniform(0, 1)), fcos=float(rng.uniform(0.2, 0.95)), ppm=float(rng.uniform(0, 20)),
            exact_mass=float(exact_mass_from_smiles(smi)), source=src, spec_evaluated=src == 1,
        )

    class1 = [hit(1) for _ in range(n_class1)]
    if lock_first:
        class1[0] = _Hit(KETO, "RAW_LOCK", 0.9, spec=0.93, tani=0.8, mass=1.0, n_match=8.0, source=1, spec_evaluated=True,
                         exact_mass=float(exact_mass_from_smiles(KETO)))
    return class1, [hit(2) for _ in range(n_class2)], [hit(3) for _ in range(n_analog)]


def reference_merge(class1, class2, analogs, cfg):
    """Baseline semantics: locks by cosine, then the reference canonical dedup of the pool."""
    top_k = int(cfg.top_k)
    locked = _locked_hits(class1, 0.75, top_k)
    locked_keys = {_skeleton_key(h) for h in locked}
    pool = _unlocked_pool(class1, class2, analogs, locked, cfg)
    merged = _merge_hits(pool, top_k=len(pool))
    competed = [h for h in merged if h.inchikey14 not in locked_keys][: max(top_k - len(locked), 0)]
    return [h.smiles for h in locked + competed][:top_k]


class ReverseStub:
    """Adversarial reranker: reverses the window order."""

    def predict(self, X):
        return np.arange(len(X), dtype=np.float64)  # higher is better, so the last row ranks first


class FeatureStub:
    def __init__(self, name):
        self.col = FEATURE_NAMES.index(name)

    def predict(self, X):
        return np.asarray(X, dtype=np.float64)[:, self.col]


class LeakageTests(unittest.TestCase):
    def test_feature_contract_is_inference_only(self):
        forbidden = ("inchikey", "smiles", "formula", "label", "truth", "target", "bit_", "identity", "is_truth")
        for name in FEATURE_NAMES:
            for tok in forbidden:
                self.assertNotIn(tok, name.lower(), name)
        params = inspect.signature(window_feature_matrix).parameters
        for p in params:
            self.assertFalse(re.search(r"(^|_)(fp|fingerprint)s?$", p), p)
            for tok in forbidden:
                self.assertNotIn(tok, p.lower(), p)
        self.assertLess(len(FEATURE_NAMES), 64, "a raw fingerprint would not fit this contract")

    def test_masked_library_and_partitions_exclude_identity_and_aliases(self):
        cfg = get_config()
        raw = [inchikey14_from_smiles(s) for s in LIBRARY]
        raw[1] = "RAWALIAS_ENOL"  # enol row carries a distinct raw key but the same canonical skeleton
        raw[3] = "RAWALIAS_PYR"
        index, _ = build_index(cfg, LIBRARY, raw_keys=raw)
        n = len(LIBRARY)
        fit_take = np.array([2, 5, 6, 7, 8], dtype=np.int64)  # pyridone (row 2) is in the fit subset
        plan = V.plan_holdouts(index, fit_take, sizes={"RR_TRAIN": 6, "DEV": 3, "AUDIT": 6, "PANEL": 2},
                               panel_rows=[20, 21], seed=0, progress=False)
        V.check_plan_disjoint(plan, fit_take)
        holdout = {r for rows in plan.partitions.values() for r in rows}
        self.assertFalse(holdout & set(fit_take.tolist()))
        self.assertNotIn(3, holdout, "hydroxypyridine aliases a fit-subset skeleton and must be excluded")
        self.assertGreaterEqual(plan.excluded["alias_in_fit"], 1)
        canon = [plan.canonical[r] for r in holdout]
        self.assertEqual(len(canon), len(set(canon)))
        masked_rows = plan.masked_rows()
        if 0 in holdout or 1 in holdout:
            self.assertTrue({0, 1} <= masked_rows, "both tautomers of a held-out skeleton are masked")
        masked = V.masked_library(index, masked_rows)
        self.assertEqual(len(masked.smiles), n - len(masked_rows))
        masked_keys = set(masked.inchikey14.tolist())
        masked_canon = {inchikey14_from_smiles(s) for s in masked.smiles}
        for r in holdout:
            self.assertNotIn(str(index.inchikey14[r]), masked_keys)
            self.assertNotIn(plan.canonical[r], masked_canon)
        from src.neighbors import SpectralNeighborIndex

        nbr = SpectralNeighborIndex.from_structure_index(masked, cfg)
        self.assertEqual(nbr.binned.shape[0], len(masked.smiles))

    def test_fit_subset_is_reproducible_and_matches_the_rng_call(self):
        a = fitting_subset_indices(1000, 120, seed=42)
        b = fitting_subset_indices(1000, 120, seed=42)
        self.assertTrue(np.array_equal(a, b))
        self.assertTrue(np.array_equal(a, np.sort(a)))
        self.assertEqual(a.size, 120)
        expected = np.sort(np.random.default_rng(42).choice(1000, size=120, replace=False))
        self.assertTrue(np.array_equal(a, expected))
        self.assertTrue(np.array_equal(fitting_subset_indices(50, 120), np.arange(50)))


class MergeTests(unittest.TestCase):
    def setUp(self):
        self.cfg = get_config()
        self.cfg.top_k = 25
        self.cfg.rerank_window = 8

    def test_disabled_mode_equals_reference_merge(self):
        rng = np.random.default_rng(1)
        for trial in range(120):
            class1, class2, analogs = random_pool(rng, lock_first=trial % 3 == 0)
            expected = reference_merge(class1, class2, analogs, self.cfg)
            got = [h.smiles for h in _merge_tiers(class1, class2, analogs, self.cfg)]
            self.assertEqual(got, expected, f"trial {trial}")
            traced: dict = {}
            got_traced = [h.smiles for h in _merge_tiers(class1, class2, analogs, self.cfg, trace=traced)]
            self.assertEqual(got_traced, expected, "tracing must not change the baseline order")
            self.assertEqual(traced["pool_size"], len(class1) + len(class2) + len(analogs) - len(traced["locked"]["smiles"]))

    def test_lazy_dedup_equals_reference_on_tautomer_pairs(self):
        rng = np.random.default_rng(2)
        for _ in range(200):
            class1, class2, analogs = random_pool(rng)
            ordered = sorted(class1 + class2 + analogs, key=lambda h: -h.score)
            k = int(rng.integers(1, 12))
            lazy = [h.smiles for h in _emit_unique(ordered, [], k)]
            ref = [h.smiles for h in _merge_hits(ordered, top_k=k)]
            self.assertEqual(lazy, ref)
            keys = [inchikey14_from_smiles(s) for s in lazy]
            self.assertEqual(len(keys), len(set(keys)))

    def test_locked_prefix_survives_adversarial_reranker(self):
        rng = np.random.default_rng(3)
        for _ in range(40):
            class1, class2, analogs = random_pool(rng, lock_first=True)
            class1.append(_Hit(ENOL, "RAW_ENOL_DECOY", 0.99, tani=0.99, fcos=0.99, mass=1.0, source=1,
                               exact_mass=float(exact_mass_from_smiles(ENOL))))  # alias of the locked keto
            baseline = _merge_tiers(class1, class2, analogs, self.cfg)
            reranked = _merge_tiers(class1, class2, analogs, self.cfg, reranker=ReverseStub())
            locked = _locked_hits(class1, 0.75, self.cfg.top_k)
            self.assertGreaterEqual(len(locked), 1)
            self.assertEqual([h.smiles for h in reranked[: len(locked)]], [h.smiles for h in locked])
            self.assertEqual([h.smiles for h in baseline[: len(locked)]], [h.smiles for h in locked])
            keys = [_skeleton_key(h) for h in reranked]
            self.assertEqual(len(keys), len(set(keys)), "no tautomer alias of a locked hit may be emitted")
            self.assertEqual(sorted(_skeleton_key(h) for h in reranked), sorted(_skeleton_key(h) for h in baseline),
                             "the reranker may only reorder the same set of skeletons")

    def test_reranker_changes_only_the_unlocked_window(self):
        rng = np.random.default_rng(4)
        changed = 0
        for _ in range(30):
            class1, class2, analogs = random_pool(rng, n_class2=20, lock_first=True)
            trace: dict = {}
            baseline = [h.smiles for h in _merge_tiers(class1, class2, analogs, self.cfg)]
            reranked = [h.smiles for h in _merge_tiers(class1, class2, analogs, self.cfg, reranker=ReverseStub(), trace=trace)]
            n_locked = len(trace["locked"]["smiles"])
            self.assertEqual(reranked[:n_locked], baseline[:n_locked])
            self.assertIn("rerank_order", trace)
            self.assertEqual(trace["window"]["features"].shape, (len(trace["window"]["smiles"]), len(FEATURE_NAMES)))
            changed += int(reranked != baseline)
        self.assertGreater(changed, 0, "an order-reversing reranker must change some first unlocked slot")

    def test_offline_arm_reconstruction_matches_production(self):
        rng = np.random.default_rng(5)
        stub = FeatureStub("tanimoto")
        for _ in range(40):
            class1, class2, analogs = random_pool(rng, n_class2=20, lock_first=True)
            trace: dict = {}
            baseline = [h.smiles for h in _merge_tiers(class1, class2, analogs, self.cfg, trace=trace)]
            prod = [h.smiles for h in _merge_tiers(class1, class2, analogs, self.cfg, reranker=stub)]
            offline = V.apply_reranker_to_trace(trace, stub, self.cfg)
            self.assertEqual(offline, prod)
            self.assertEqual(V.baseline_from_trace(trace, self.cfg), baseline)


class EndToEndRankingTests(unittest.TestCase):
    def test_rank_molecules_with_and_without_reranker(self):
        cfg = get_config()
        cfg.top_k = 25
        cfg.rerank_window = 50
        index, feats = build_index(cfg, LIBRARY)
        ext = build_external(cfg, EXTERNAL)
        rng = np.random.default_rng(0)
        mids = ["q0", "q1", "q2"]
        mf = {m: feats[i * 4] for i, m in enumerate(mids)}
        pf = {m: np.clip(index.fingerprints[i * 4].astype(np.float32) * 0.8 + rng.uniform(0, 0.3, cfg.fp_bits).astype(np.float32), 0, 1)
              for i, m in enumerate(mids)}
        for q, isomers in ((KETO, EXTERNAL[6:11]), ("CCCCO", EXTERNAL[11:15]), ("OCC(O)CO", EXTERNAL[15:18])):
            for iso in isomers:
                self.assertAlmostEqual(exact_mass_from_smiles(iso), exact_mass_from_smiles(q), places=4, msg=iso)
        traces: dict = {}
        base = rank_molecules(mf, pf, index, cfg, external_index=ext, trace_out=traces)
        again = rank_molecules(mf, pf, index, cfg, external_index=ext)
        self.assertEqual(base, again, "baseline ranking is deterministic")
        rer = rank_molecules(mf, pf, index, cfg, external_index=ext, reranker=ReverseStub())
        for m in mids:
            tr = traces[m]
            locked = tr["locked"]["smiles"]
            self.assertEqual(base[m][: len(locked)], locked)
            self.assertEqual(rer[m][: len(locked)], locked)
            self.assertEqual(tr["final_smiles"], base[m])
            self.assertEqual(len(base[m]), cfg.top_k)
            keys = [inchikey14_from_smiles(s) for s in base[m]]
            self.assertEqual(len(keys), len(set(keys)))
            offline = V.apply_reranker_to_trace(tr, ReverseStub(), cfg, baseline_final=base[m])
            k = len(tr["merged_smiles"])  # locked + competed; the pad region beyond it is not part of either arm
            self.assertEqual(offline[:k], rer[m][:k])
        self.assertTrue(any(rer[m] != base[m] for m in mids))


class ValidationHarnessTests(unittest.TestCase):
    def test_checkpoint_guard(self):
        cfg = get_config(num_epochs=4)
        good = {"stop_reason": "epochs_complete", "completed_epochs": 4, "fit_subset_hash": "f" * 16,
                "model_config_hash": "c" * 16, "seed": 7}
        ok, why = V.checkpoint_is_complete(good, cfg, fit_hash="f" * 16, config_hash="c" * 16, seed=7)
        self.assertTrue(ok, why)
        for bad, needle in (
            (dict(good, stop_reason="time_limit"), "stop_reason"),
            (dict(good, completed_epochs=3), "completed_epochs"),
            (dict(good, fit_subset_hash="0" * 16), "fit subset"),
            (dict(good, model_config_hash="0" * 16), "model config"),
            (dict(good, seed=42), "seed"),
            ({}, "no extra"),
        ):
            ok, why = V.checkpoint_is_complete(bad, cfg, fit_hash="f" * 16, config_hash="c" * 16, seed=7)
            self.assertFalse(ok)
            self.assertIn(needle, why)

    def test_bootstrap_is_deterministic_and_zero_width_for_identical_arms(self):
        zero = V.paired_bootstrap(np.zeros(50), n_resamples=512, seed=0)
        self.assertEqual((zero["mean"], zero["ci_lo"], zero["ci_hi"]), (0.0, 0.0, 0.0))
        d = np.random.default_rng(9).normal(0.02, 0.1, size=200)
        a = V.paired_bootstrap(d, n_resamples=1024, seed=0)
        b = V.paired_bootstrap(d, n_resamples=1024, seed=0)
        self.assertEqual(a, b)
        self.assertLess(a["ci_lo"], a["mean"])
        self.assertGreater(a["ci_hi"], a["mean"])

    def _audit(self, n, delta, k_delta=0.0, base=0.3):
        rr_b = np.full(n, base)
        rr_r = rr_b + delta
        rows = [{"regime": "U", "truth_in_window": True, "rerank_smiles_differs": True}] * n
        pooled = V.paired_bootstrap(rr_r - rr_b, n_resamples=256, seed=0)
        return {
            "regimes": {
                "U": {"baseline": V.arm_metrics(rr_b), "rerank": V.arm_metrics(rr_r), "paired_delta": pooled},
                "K": {"baseline": V.arm_metrics(rr_b[: n // 2]), "rerank": V.arm_metrics(rr_b[: n // 2] + k_delta),
                      "paired_delta": V.paired_bootstrap(np.full(n // 2, k_delta), n_resamples=256, seed=0)},
            },
            "coverage": {}, "rows": rows,
            "pooled": {"baseline": V.arm_metrics(rr_b), "rerank": V.arm_metrics(rr_r), "paired_delta": pooled, "n_rankings_changed": n},
        }

    def test_promotion_rule(self):
        self.assertFalse(V.PROMOTION_RULE["dev_used_for_selection"])
        passing = V.decide_promotion(self._audit(400, 0.03), None, reranker_fitted=True)
        self.assertTrue(passing["promoted"])
        self.assertEqual(passing["selected_arm"], "rerank")
        for audit, panel, fitted in (
            (self._audit(200, 0.03), None, True),  # too few audit queries
            (self._audit(400, 0.0), None, True),  # no gain, zero-width CI
            (self._audit(400, 0.005), None, True),  # below the minimum delta
            (self._audit(400, 0.03, k_delta=-0.01), None, True),  # loses Class 1
            (self._audit(400, 0.03), self._audit(60, -0.05), True),  # panel regression with n >= 50
            (self._audit(400, 0.03), None, False),  # nothing fitted
            (None, None, True),
        ):
            out = V.decide_promotion(audit, panel, reranker_fitted=fitted)
            self.assertFalse(out["promoted"])
            self.assertEqual(out["selected_arm"], "baseline")
        small_panel = V.decide_promotion(self._audit(400, 0.03), self._audit(20, -0.5), reranker_fitted=True)
        self.assertTrue(small_panel["promoted"], "a panel below panel_min_n does not gate")

    def test_booster_is_deterministic_and_round_trips(self):
        rng = np.random.default_rng(11)
        n_q, n_c = 12, 30
        X = rng.uniform(0, 1, size=(n_q * n_c, len(FEATURE_NAMES))).astype(np.float32)
        qid = np.repeat(np.arange(n_q), n_c)
        y = np.zeros(n_q * n_c, dtype=np.int64)
        for q in range(n_q):
            y[q * n_c + int(np.argmax(X[q * n_c:(q + 1) * n_c, FEATURE_NAMES.index("tanimoto")]))] = 1
        model, report = fit_reranker(X, y, qid, min_queries=5)
        self.assertEqual(report["status"], "fitted", report)
        self.assertTrue(report["deterministic"])
        self.assertLessEqual(report["max_refit_abs_diff"], 1e-6)
        with tempfile.TemporaryDirectory() as tmp:
            path = save_reranker(model, Path(tmp) / "reranker.pkl")
            loaded = load_reranker(path)
        self.assertTrue(np.array_equal(loaded.predict(X[:50]), model.predict(X[:50])))
        _, short = fit_reranker(X, y, qid, min_queries=100)
        self.assertIn("queries with a positive", short["status"])

    def test_strict_submission_check(self):
        cfg = get_config()
        pool = [s for s in LIBRARY if s not in (ENOL, HYDROXYPYRIDINE)]
        self.assertGreaterEqual(len(pool), 25)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            sample = tmp / "sample_submission.csv"
            sample.write_text("molecule_id,smiles\nA,C\nB,C\n")

            def write(rows):
                p = tmp / "submission.csv"
                p.write_text("molecule_id,smiles\n" + "\n".join(f"{m},{';'.join(g)}" for m, g in rows) + "\n")
                return p

            good = write([("A", pool[:25]), ("B", pool[1:26])])
            report = V.strict_submission_check(good, sample, top_k=cfg.top_k)
            self.assertEqual(report["rows"], 2)
            self.assertEqual(report["max_guesses"], 25)
            self.assertEqual(report["rows_with_duplicate_skeleton"], 0)
            with self.assertRaises(ValueError):
                V.strict_submission_check(write([("A", pool[:26]), ("B", pool[:25])]), sample)
            with self.assertRaises(ValueError):
                V.strict_submission_check(write([("A", [KETO, ENOL] + pool[2:25]), ("B", pool[:25])]), sample)
            with self.assertRaises(ValueError):
                V.strict_submission_check(write([("A", pool[:25])]), sample)  # molecule_id mismatch
            with self.assertRaises(ValueError):
                V.strict_submission_check(write([("A", ["C(C"] + pool[1:25]), ("B", pool[:25])]), sample)

    def test_graph_key_is_necessary_for_canonical_equality(self):
        for a, b in ((KETO, ENOL), (PYRIDONE, HYDROXYPYRIDINE), ("C[C@H](O)CC", "C[C@@H](O)CC")):
            self.assertEqual(inchikey14_from_smiles(a), inchikey14_from_smiles(b))
            self.assertEqual(heavy_atom_graph_key(a), heavy_atom_graph_key(b))
        self.assertNotEqual(heavy_atom_graph_key("CCCCO"), heavy_atom_graph_key("CC(C)CO"))


class NotebookSyncTests(unittest.TestCase):
    def test_generated_notebooks_match_src_and_are_upload_ready(self):
        gen = (ROOT / "scripts" / "make_kaggle_notebook.py").read_text()
        src_files = re.search(r"SRC_FILES = \[(.*?)\]", gen, re.S).group(1)
        listed = re.findall(r'"(src/[^"]+)"', src_files)
        self.assertIn("src/validation.py", listed)
        self.assertIn("src/rerank.py", listed)
        for name in ("kaggle_submission.ipynb", "kaggle_gpu_submission.ipynb"):
            path = ROOT / name
            self.assertTrue(path.exists(), name)
            self.assertLess(path.stat().st_size, 1_000_000)
            nb = json.loads(path.read_text())
            self.assertEqual(nb["metadata"]["kaggle"]["accelerator"], "gpu")
            self.assertFalse(nb["metadata"]["kaggle"]["isInternetEnabled"])
            code = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
            for cell in code:
                ast.parse(cell)
            joined = "\n".join(code)
            for bad in ("rmtree", ".unlink(", "os.remove("):
                self.assertNotIn(bad, joined, f"destructive cleanup {bad} in {name}")
            self.assertIn("REQUIRE_CUDA", joined)
            self.assertIn("HistGradientBoostingClassifier", joined)
            self.assertIn("PROMOTION_RULE", joined)
            self.assertIn("strict_submission_check", joined)
            self.assertIn("SUMMARY_JSON", joined)
            m = re.search(r"FILES = json\.loads\((.*?)\)\nSRC_HASHES", joined, re.S)
            files = json.loads(ast.literal_eval(m.group(1)))
            for rel in listed:
                self.assertIn(rel, files, f"{rel} missing from {name}")
                current = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
                embedded = hashlib.sha256(files[rel].encode("utf-8")).hexdigest()
                self.assertEqual(embedded, current, f"{rel} changed after {name} was generated; rerun scripts/make_kaggle_notebook.py")


if __name__ == "__main__":
    unittest.main()
