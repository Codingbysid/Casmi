"""Execution-path gates: required reranker, caches, predict-only, notebook sync."""

from __future__ import annotations

import ast
import hashlib
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import get_config  # noqa: E402
from src.infer import _Hit, _merge_tiers, predict_fingerprint_ensemble  # noqa: E402
from src.neighbors import SpectralNeighborIndex  # noqa: E402
from src.pipeline import (  # noqa: E402
    LIBRARY_SCHEMA,
    STRATEGY_ID,
    PipelineError,
    _query_shard_path,
    _row_order_hash,
    assemble_saved_query_rows,
    atomic_write_text,
    file_sha256,
    fit_required_reranker,
    load_array_cache,
    reranker_provenance_problems,
    run_predict_only,
    save_array_cache,
)
from src.validation import Query, Truth  # noqa: E402
from src.rerank import FEATURE_NAMES, rerank_scores  # noqa: E402


class RequiredRerankerTests(unittest.TestCase):
    def test_unfit_reranker_raises(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(size=(20, len(FEATURE_NAMES))).astype(np.float32)
        y = np.zeros(20, dtype=np.int64)
        y[0] = 1
        qid = np.repeat(np.arange(2), 10)
        with self.assertRaises(PipelineError):
            fit_required_reranker(X, y, qid, min_queries=100)

    def test_missing_reranker_is_not_the_linear_score(self):
        from src.infer import predict_test

        text = Path(predict_test.__code__.co_filename).read_text()
        self.assertIn("required reranker is missing", text)
        self.assertIn("allow_decoder", text)
        self.assertNotIn('decoder is None and cfg.smiles_checkpoint_path.exists()', text.split("allow_decoder", 1)[0])

    def test_uniform_predictions_raise(self):
        cfg = get_config(fp_bits=8)
        with self.assertRaises(RuntimeError):
            predict_fingerprint_ensemble([], [{}], cfg, allow_uniform=False)

    def test_nonfinite_reranker_scores_raise(self):
        class Bad:
            def predict(self, X):
                return np.array([np.nan] * len(X))

        with self.assertRaises(RuntimeError):
            rerank_scores(Bad(), np.zeros((2, len(FEATURE_NAMES)), dtype=np.float32))

    def test_neighbor_width_mismatch_raises(self):
        nbr = SpectralNeighborIndex(binned=np.zeros((1, 4), np.float16), fingerprints=np.zeros((1, 4), np.uint8))
        with self.assertRaises(ValueError):
            nbr.blend_one(np.zeros(8, np.float32), {"binned": np.zeros(4), "ionization_mode": ""})

    def test_locked_prefix_survives_required_reranker(self):
        cfg = get_config()
        cfg.top_k = 5
        cfg.rerank_window = 4
        locked = _Hit("CCO", "LOCK", 0.9, spec=0.95, tani=0.5, mass=1.0, n_match=8, source=1)
        rest = [
            _Hit(smi, f"K{i}", 0.1, tani=0.2 + i, mass=1.0, source=2, exact_mass=40.0 + i)
            for i, smi in enumerate(["CCC", "CCCC", "CCCCC", "CCCCCC"])
        ]

        class Flip:
            def predict(self, X):
                return np.arange(len(X), dtype=np.float64)

        base = [h.smiles for h in _merge_tiers([locked], rest, [], cfg)]
        reranked = [h.smiles for h in _merge_tiers([locked], rest, [], cfg, reranker=Flip())]
        self.assertEqual(base[0], "CCO")
        self.assertEqual(reranked[0], "CCO")
        self.assertNotEqual(reranked[1:], base[1:])


class CacheAndPublishTests(unittest.TestCase):
    def test_interrupted_write_keeps_the_previous_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run_manifest.json"
            path.write_text("previous")
            with patch("src.pipeline.os.replace", side_effect=OSError("disk full")):
                with self.assertRaises(OSError):
                    atomic_write_text(path, "replacement")
            self.assertEqual(path.read_text(), "previous")

    def test_cache_rejects_a_changed_row_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "library.npz"
            keys = np.array(["AAAA"], dtype=object)
            save_array_cache(
                path,
                {"inchikey14": keys, "values": np.zeros(1)},
                {"schema": LIBRARY_SCHEMA, "row_order_sha256": _row_order_hash(keys)},
            )
            side = path.with_suffix(".json")
            meta = json.loads(side.read_text())
            meta["row_order_sha256"] = "not-the-hash"
            side.write_text(json.dumps(meta))
            with self.assertRaises(PipelineError):
                load_array_cache(path, expect_schema=LIBRARY_SCHEMA)

    def test_predict_only_missing_bundle_does_not_train(self):
        calls = []

        def boom(*_a, **_k):
            calls.append("train")
            raise AssertionError("predict-only must not train")

        cfg = get_config(**{key: value for key, value in {
            "num_epochs": 4, "max_train_spectra": 120_000, "batch_size": 64, "d_model": 128,
            "n_heads": 4, "n_transformer_layers": 2, "n_conv_channels": 64, "top_n_peaks": 128,
            "n_mz_bins": 512, "fp_bits": 2048, "neighbor_k": 20, "neighbor_blend": 0.5,
            "use_neighbor_fp": True, "rerank_window": 200, "use_representative_metadata": False,
            "restore_best_checkpoint": False, "use_observed_precursor": False, "ensemble_seeds": 3,
            "max_mass_candidates": 8192,
        }.items()})
        with tempfile.TemporaryDirectory() as tmp:
            with patch("src.tpu_trainer.train_spec2fp", boom):
                with self.assertRaises(PipelineError) as ctx:
                    run_predict_only(cfg, bundle_dir=tmp)
        self.assertIn("does not train", str(ctx.exception))
        self.assertEqual(calls, [])
        self.assertEqual(STRATEGY_ID, "required-reranker-v1")

    def test_query_shards_resume_and_a_new_signature_recomputes(self):
        def query(row: int) -> Query:
            return Query(
                f"RR_TRAIN_U_{row}",
                "RR_TRAIN",
                "U",
                row,
                Truth("RAW", "CANON", "GRAPH", "CCO", 46.0),
                {},
                {},
            )

        queries = [query(1), query(2), query(3)]

        def rows(q: Query):
            x = np.full((2, len(FEATURE_NAMES)), float(q.row), dtype=np.float32)
            y = np.array([1, 0], dtype=np.int64)
            return x, y

        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            calls = []

            def produce(q: Query):
                calls.append(q.qid)
                if q.qid == "RR_TRAIN_U_2":
                    raise RuntimeError("interrupted")
                return rows(q)

            with self.assertRaises(RuntimeError):
                assemble_saved_query_rows(run, "sig-a", queries, produce)
            self.assertEqual(calls, ["RR_TRAIN_U_1", "RR_TRAIN_U_2"])
            first = _query_shard_path(run, "sig-a", "RR_TRAIN_U_1")
            digest = file_sha256(first)
            calls.clear()

            def resume(q: Query):
                calls.append(q.qid)
                return rows(q)

            X, y, qid = assemble_saved_query_rows(run, "sig-a", queries, resume)
            self.assertEqual(calls, ["RR_TRAIN_U_2", "RR_TRAIN_U_3"])
            self.assertEqual(file_sha256(first), digest)
            self.assertEqual(X.shape, (6, len(FEATURE_NAMES)))
            self.assertEqual(y.tolist(), [1, 0, 1, 0, 1, 0])
            self.assertEqual(qid.tolist(), [0, 0, 1, 1, 2, 2])
            calls.clear()
            assemble_saved_query_rows(run, "sig-b", queries, resume)
            self.assertEqual(calls, ["RR_TRAIN_U_1", "RR_TRAIN_U_2", "RR_TRAIN_U_3"])

    def test_predict_only_checks_reranker_and_class2_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "reranker.pkl"
            path.write_bytes(b"reranker-bytes")
            digest = file_sha256(path)
            (Path(tmp) / "reranker.json").write_text(json.dumps({"signature": "pair-sig"}))
            manifest = {
                "pairs_source_hash": "not-the-source",
                "pairs_signature": "pair-sig",
                "class2_sha256": "class2-a",
                "reranker": {"signature": "pair-sig", "sha256": digest},
            }
            problems = reranker_provenance_problems(manifest, path, "class2-b")
            self.assertIn("pairs source hash does not match the manifest", problems)
            self.assertIn("class 2 content hash does not match the manifest", problems)
            manifest["pairs_source_hash"] = __import__(
                "src.pipeline", fromlist=["source_hash"]
            ).source_hash(__import__("src.pipeline", fromlist=["PAIRS_SOURCE"]).PAIRS_SOURCE)
            manifest["class2_sha256"] = "class2-b"
            self.assertEqual(reranker_provenance_problems(manifest, path, "class2-b"), [])
            manifest["reranker"]["sha256"] = "0" * 64
            self.assertIn(
                "reranker artifact hash does not match the manifest",
                reranker_provenance_problems(manifest, path, "class2-b"),
            )
            manifest["reranker"]["sha256"] = digest
            manifest["reranker"]["signature"] = "other"
            self.assertIn(
                "reranker signature does not match the manifest",
                reranker_provenance_problems(manifest, path, "class2-b"),
            )


class NotebookWorkflowTests(unittest.TestCase):
    def test_generated_notebooks_call_the_pipeline(self):
        train = json.loads((ROOT / "kaggle_gpu_submission.ipynb").read_text())
        predict = json.loads((ROOT / "kaggle_predict_only.ipynb").read_text())
        for nb in (train, predict):
            self.assertEqual(nb["metadata"]["kaggle"]["accelerator"], "gpu")
            self.assertFalse(nb["metadata"]["kaggle"]["isInternetEnabled"])
            self.assertLess((ROOT / "kaggle_gpu_submission.ipynb").stat().st_size, 1_000_000)
        train_cells = ["".join(c["source"]) for c in train["cells"] if c["cell_type"] == "code"]
        predict_cells = ["".join(c["source"]) for c in predict["cells"] if c["cell_type"] == "code"]
        for cell in train_cells + predict_cells:
            ast.parse(cell)
        self.assertIn("run_train_and_predict", train_cells[-1])
        self.assertNotIn("run_predict_only", train_cells[-1])
        self.assertIn("run_predict_only", predict_cells[-1])
        self.assertNotIn("run_train_and_predict", predict_cells[-1])
        self.assertNotIn("decision.get(\"promoted\")", "\n".join(train_cells + predict_cells))
        material = next(cell for cell in train_cells if "FILES = json.loads" in cell)
        payload = re.search(r"FILES = json\.loads\((.*?)\)\n", material, re.S).group(1)
        files = json.loads(ast.literal_eval(payload))
        for rel in ("src/pipeline.py", "src/rerank.py", "src/infer.py"):
            current = hashlib.sha256((ROOT / rel).read_bytes()).hexdigest()
            self.assertEqual(hashlib.sha256(files[rel].encode()).hexdigest(), current)
        self.assertNotIn('decision.get("promoted")', files["src/pipeline.py"])
