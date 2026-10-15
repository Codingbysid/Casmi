# Enveda CASMI 2026 — model brief

This file is the source of truth for the current pipeline. Read it before proposing changes.
Notebooks are generated from `src/` by `scripts/make_kaggle_notebook.py`. Edit `src/` and the generator, then regenerate. Do not hand-edit the notebooks. `tests/test_reranker_experiment.py` fails when the embedded `src/` no longer matches the working tree.

The next upload is the required-reranker train-and-predict notebook (`kaggle_submission.ipynb`, `kaggle_gpu_submission.ipynb`, and `kaggle_submission_tpu.ipynb` are the same notebook). `kaggle_predict_only.ipynb` loads a finished bundle and does not train. A notebook uploaded before this change still contains the promotion gate and is not this run.

## Task

Kaggle: **Enveda CASMI 2026 — Molecule ID from Mass Spectra**.

For each unknown molecule, rank up to **25 SMILES**. The metric is **MRR@25** on the first 14 characters of the RDKit InChIKey (InChIKey14, the connectivity skeleton). Stereochemistry and the second InChIKey block do not count. Exactly one row per `molecule_id`. Guesses are semicolon-separated SMILES. Empty rows are invalid.

RDKit versions matter for the key. Production dedup is `inchikey14_from_smiles`: the module `TautomerEnumerator` with `SetMaxTransforms(1000)`, then InChIKey block 1. It does not call `RemoveStereochemistry` first. The Kaggle metric page (version 14, RDKit 2026.03.3) is a different sequence, implemented as `official_metric_inchikey14` and left unused. See experiment `official-inchikey14-v14` below. The notebook installs the 2025.09.5 wheel; this laptop has 2026.03.6. Do not upgrade the training environment to chase the scorer pin.

Measured earlier on the Class 2 pool: 10% of raw `inchikey14` values change under the production tautomer canonicalization. Canonicalization costs ~43 ms per SMILES versus ~0.4 ms for the heavy-atom graph key (`heavy_atom_graph_key`, a verified necessary condition for canonical-key equality: 0 mismatches over 487 tautomer-changed structures). Production dedup canonicalizes only when two candidates share a graph key; the output is identical to canonicalizing everything.

Approximate hidden-test mix (organizer estimate, not a label we have):

- **~16% Class 1** — the true structure is in `train.parquet` (same skeleton, spectrum may be from another instrument).
- **~45% Class 2** — not in train, but a known natural product that can be retrieved from an external structure library if the fingerprint is good.
- **~39% Class 3** — not in train and not in the current external library. These score 0 unless we generate the right skeleton.

Public `test.parquet` is a **placeholder drawn from train**. Kaggle replaces it at score time. Never filter candidate libraries on public test precursor masses.

## Data (on Kaggle, not on the laptop)

Training is Kaggle-only. Local `data/` and the competition download were deleted.

| Input | What |
|---|---|
| Competition | `train.parquet` ~2.5M spectra, ~277k SMILES, ~276k InChIKey14. `test.parquet` ~1,213 spectra / 400 molecules. |
| Class 2 dataset `class2_candidates.parquet2` | `class2_candidates.parquet`, **64 MB, 452,608** unique structures. COCONUT 2.0 (Aug 2026) ∪ LOTUS. Train InChIKey14s removed. Mass 50–2000 Da. Columns: `smiles`, `inchikey14`, `exact_mass`, `formula`, `fp_packed` (Morgan r=2, 2048 bits, packed to 256 uint8 bytes). **Not** filtered on test masses. |
| RDKit wheel | `rdkit-2025.9.5-cp312-cp312-manylinux_2_28_x86_64.whl`, **~35 MB**. Kaggle Python is **3.12**. Internet is **OFF**, so the first notebook cell must `pip install --no-index --no-deps` this wheel. A 20 KB file named `rdkit-2025-9-5` is a stub and is not a wheel. |

Adduct neutral-mass offsets used in `src/chem.py`:

- `[M+H]+` −1.007276
- `[M-H]-` +1.007276
- `[M+NH4]+` −18.033826
- `[M+Na]+` −22.989218
- `[M+K]+` −38.963158
- `[M+CH2O2-H]-` −44.998201
- `[M+Cl]-` −34.969402

## What actually scores

Public leaderboard (same kernel family, `notebook06a99a3696`):

| Run | Change vs best | Public MRR@25 |
|---|---|---|
| V1–V3 | RDKit never installed (no fingerprints) | 0.074 |
| V4 | Class 2 filled all 25 slots, Class 1 discarded | 0.107 then 0.071 depending on merge |
| V5 and V6 | Lock cosine ≥ 0.75, then weak Class 1 competes with Class 2. Neighbor blend on. Spec2FP 120k structures, 4 epochs, 3 seeds, ~1.5 h/seed. | 0.139 |
| V7 | Weak Class 1 dropped; remaining slots Class 2 only | 0.119 |
| V8 | Same as 0.139 but neighbor blend off | 0.133 |
| V9 | Spec2FP on all ~276k skeletons, 6 epochs | 0.126 |
| **Peak-support ranking** | **Best, reported 2026-10-07 (Kaggle label V5).** Cosine lock also requires ≥5 matched peaks. Fewer than 4 peaks, or under 15% of query intensity, scores 0. Class 2 uses `0.92 * (0.78 * tanimoto + 0.22 * mass)`. InChIKey14 dedup after tautomer canonicalization. Focal BCE, 3 seeds, neighbor blend on. | **0.150** |

**0.150 is the score to keep selected.** It is +0.011 over 0.139. Do not revert the peak gate, the Class 2 rescore, or tautomer dedup. Do not drop weak Class 1, turn neighbor blend off, or train Spec2FP longer. Those earlier experiments lost to 0.139, and 0.139 itself is no longer the best.

The 0.150 run bundled **four** ranking changes (peak-support gate, five-peak lock, Class 2 rescore, tautomer dedup) in one submit, so the +0.011 cannot be attributed to any one of them. Two further runs were measured and lost: representative-spectrum metadata in training (0.134) and a candidate-ranking training loss (failed the local gate; its control scored 0.137).

0.150 is still well short of 0.25. The public number does not say how many Class 1 locks or Class 2 slots the run used. The next change needs the log lines `locked Class 1 (cosine>=0.75, n_match>=5)` and `class2_slots` before another ranking tweak.

## Pipeline (inference)

1. Install RDKit from the attached wheel. Fail if `from rdkit import Chem` does not work.
2. Build a train **StructureIndex**: one representative MS2 per InChIKey14 (the row with the most peaks), exact mass from formula, Morgan fingerprint, top-128 peaks.
3. Train **3 Spec2FP seeds** (42, 7, 123) on the fixed 120k fitting subset (`fitting_subset_indices`, seed 42). Average **pre-sigmoid logits**, then sigmoid. Last epoch is kept (`restore_best_checkpoint` is false). Every seed must reach `completed_epochs == 4`; a time limit raises instead of ranking with a partial ensemble. Each finished seed is written atomically under `/kaggle/working/casmi_run/checkpoints/` and the manifest is updated immediately. A checkpoint is reused only when the fit-subset identity hash, model/training config hash, source/preprocessing hash, dependency versions, seed, and four completed epochs all match.
4. Blend the predicted fingerprint with a spectral-neighbor fingerprint: binned-spectrum cosine kNN, **k=20**, against train spectra, **alpha=0.5** (`src/neighbors.py`).
5. `rank_molecules` in `src/infer.py`:
   - **Tier 1 lock.** Class 1 hits stay at the top only when modified cosine is **≥ 0.75** and at least **5** peaks matched. Matches with fewer than 4 peaks, or under 15% of query intensity, have cosine 0.
   - **Tier 2 compete.** Verified spectral hits use `0.42 * spec + 0.46 * tanimoto + 0.12 * mass`. Class 2 and analogs have no spectrum, so they use `0.92 * (0.78 * tanimoto + 0.22 * mass)`. One slot per tautomer-canonical skeleton (`_emit_unique`, graph-key-gated canonicalization, output-equivalent to the reference `_merge_hits`).
   - **Required reranker.** The scored notebook always fits the current 22-feature `HistGradientBoostingClassifier` and passes it in. Only the top `rerank_window=200` unlocked candidates are reordered; the tail keeps the compete order and the locked prefix never moves. A missing, incompatible, or non-finite reranker raises. The linear 0.150 order is not a fallback.
   - **Tier 3 pad** to 25 with SMARTS expansion, then nearest library SMILES. The SMILES decoder is **off** (`TRAIN_DECODER = False`). It emits garbage and is not used.
6. Write `/kaggle/working/submission.csv` with columns `molecule_id`, `smiles`.

Modified cosine is GNPS-style: a peak matches if it is within 0.05 Da **or** matches after `query precursor_mz − candidate precursor` (`src/retrieval.py`). The scored run still passes the candidate's neutral exact mass as that candidate precursor. Observed precursor m/z is stored on the index and is used only by the separate opt-in experiment `precursor-mz-shift-v1`.

Class 2 lookup is `np.searchsorted` on sorted `exact_mass` inside ±15 ppm / 0.02 Da at query time. Do not prefilter the parquet.

Mass-shifted analogs (`src/analogs.py`) retrieve parents at ± these deltas, then apply a small SMARTS set, and keep products inside the query mass window: hexose 162.0528, deoxyhexose 146.0579, pentose 132.0423, glucuronide 176.0321, methyl 14.0157, oxygen 15.9949, acetyl 42.0106, water 18.0106. On the 0.139 run they almost never entered the top 25 (`analog_slots` ≈ 20). Do not spend the next experiment on analog padding.

## Spec2FP (the only trained model that is on)

`src/model.py`, class `Spec2FP`.

Inputs, all static:

- Peak list padded to **128**: m/z, intensity, neutral loss, mask. Each peak is 6 raw features plus 32 Fourier features of m/z, projected by `Linear(38, d)`.
- Binned spectrum, **512** bins from 0–600 m/z, 1D conv stack, global pool.
- Precursor feature vector (8) plus an adduct embedding.

A CLS token over the peak transformer, the binned vector, and the precursor/adduct vector are concatenated and mapped to **2048 logits** (Morgan/ECFP4, radius 2). Loss is multi-label focal BCE (`gamma=1.5`, `pos_weight=6`).

**Kaggle notebook overrides the config defaults** to the 0.139/0.150-sized net, on GPU and CPU alike:

- `d_model=128`, `n_heads=4`, `n_transformer_layers=2`, `n_conv_channels=64`
- `batch_size=64`
- `MAX_TRAIN_SPECTRA=120_000` unique library rows (`fitting_subset_indices(n, 120_000, seed=42)`, the same RNG call the 0.139/0.150 runs made)
- `NUM_EPOCHS=4`
- 3 seeds, each capped at **1.5 h**; a timeout is a hard failure, not a quiet fallback. This workflow has not been timed on a Kaggle GPU.

Config defaults in `src/config.py` (`d_model=256`, 8 epochs, 400k spectra, `restore_best_checkpoint=True`) are **not** what Kaggle runs. The scored run uses `RECIPE` in `src/pipeline.py`.

## Execution path

`src/pipeline.py` is the only path the notebooks call. The frozen recipe is `RECIPE` in that module. The notebook does not shrink epochs, the ensemble, or the 120k fitting subset to fit a timeout.

Train-and-predict (`run_train_and_predict`):

1. Preflight. Move an existing `submission.csv` to `submission_previous_<utc>.csv`. Bound OpenMP/MKL/OpenBLAS/NumExpr and Torch threads to 4. Require CUDA, sklearn `HistGradientBoostingClassifier`, and a Class 2 parquet with exactly 452,608 rows.
2. Build or load the spectral library and the neighbor bins. Caches are `library.npz` / `neighbor.npz` plus JSON sidecars (`casmi-spectral-library-v1`, `casmi-neighbor-bins-v1`). A cache is used only when schema, shapes, row-order hash, and input provenance match. The library cache is keyed by the train parquet content hash and the checkpoint source hash, not by `test.parquet`.
3. Train or reuse seeds 42, 7, and 123. Save after each completed seed. An incomplete checkpoint raises.
4. Build only the existing `RR_TRAIN` partition (1200 skeletons, validation seed 0) and its U and K regimes. Alias-in-fit rows stay excluded. Rank those queries in one process (`workers=1`). Do not rank DEV, AUDIT, or PANEL. Do not bootstrap, permute, or decide a promotion.
5. Fit the required reranker. A status other than `fitted` raises. There is no second estimator and no linear-score substitute.
6. Predict the test set with neighbor blending required, decoder off, and uniform fingerprint predictions forbidden. Print `STRATEGY_ACTIVE` before prediction.
7. Check the partial CSV, then `os.replace` it to `submission.csv`. Print `SUBMISSION_READY` only after that check. Write `summary.json` and print `SUMMARY_JSON`.

Predict-only (`run_predict_only`) loads that bundle and then does step 7. It never calls `train_spec2fp`. A missing or incompatible bundle raises with the text `does not train`.

Legitimate nearest-library padding to 25 guesses stays. "No fallback" means a failed required step raises instead of switching strategy or writing a dummy SMILES.

## How to run on Kaggle

GPU, Python 3.12, internet off. Kernel source stays under 1 MB. Execution budget is under 9 hours. Do not turn internet on to install packages; sklearn is already on the GPU image, and RDKit comes from the attached wheel.

**Train and predict.** Upload `kaggle_gpu_submission.ipynb` (same bytes as `kaggle_submission.ipynb`). Add inputs:

1. The competition dataset (`train.parquet`, `test.parquet`, `sample_submission.csv`).
2. `class2_candidates.parquet`, the 452,608-row COCONUT ∪ LOTUS pool. Do not filter it on public test masses.
3. The ~35 MB wheel `rdkit-2025.9.5-cp312-cp312-manylinux_2_28_x86_64.whl`. The 20 KB file named `rdkit-2025-9-5` is not a wheel.

Optional: a dataset containing a previous `casmi_run/` whose hashes match this source. A source-hash change trains fresh. Run All.

**Predict only.** After a train run finishes, download its `/kaggle/working/casmi_run/` output and upload that directory as its own Kaggle dataset. Attach the competition data, the same Class 2 pool, the RDKit wheel, and that dataset. Upload `kaggle_predict_only.ipynb`. Leave `BUNDLE_DIR = ""` when exactly one `run_manifest.json` is attached; if several are attached, set `BUNDLE_DIR` to the directory that contains the manifest. Run All.

The bundle is `run_manifest.json`, `checkpoints/spec2fp_s{42,7,123}.pt`, `reranker.pkl`, `library.npz` + `library.json`, and `neighbor.npz` + `neighbor.json`. Deleting the Kaggle session deletes `/kaggle/working`. A laptop path is not visible on Kaggle. Only the dataset you upload survives.

`summary.json` records strategy id, source hash, checkpoint hashes, complete seeds, reranker status, submission hash, row count, and per-stage elapsed time plus peak RSS. Those timings are measurements of that Kaggle run. Synthetic tests do not supply them.

## Code map

| Path | Role |
|---|---|
| `src/config.py` | Hyperparameters, adduct table, paths |
| `src/chem.py` | Formula mass, adducts, Morgan, InChIKey14, fp pack/unpack |
| `src/preprocessing.py` | Peak clean, binning, per-molecule aggregation |
| `src/data.py` | Parquet streaming, `dataset_from_structure_index` |
| `src/model.py` | `Spec2FP`, focal BCE |
| `src/tpu_trainer.py` | CPU/GPU training loop. Focal BCE only. Optional best-checkpoint restore is off in the notebook. |
| `src/retrieval.py` | `StructureIndex`, modified cosine, mass window |
| `src/neighbors.py` | Binned cosine kNN fingerprint blend |
| `src/ranker.py` | Linear score, dedup, submission CSV |
| `src/infer.py` | **`rank_molecules` / `_merge_tiers` — the 0.150 merge** |
| `src/analogs.py` | Mass-shift parent retrieval + SMARTS |
| `src/calibration.py` | Optional OR-merge probabilities. Ranking score used in the compete pool is the linear mix above, not only this table. |
| `src/models/smiles_decoder.py` | Class 3 decoder. **Disabled.** |
| `src/formula_head.py` | Built earlier, **disabled** in `predict_test` |
| `src/rerank.py` | Unlocked-window reranker: 22 inference-only features, fixed `HistGradientBoostingClassifier` with monotonic constraints, query-balanced weights, refit-determinism check, atomic save/load. Non-finite scores raise. The scored run requires a fitted model. |
| `src/validation.py` | Holdout machinery the train path uses for RR_TRAIN only: tautomer-aware partitions, masked library, answer-in-pool external copy, checkpoint completeness. `PROMOTION_RULE`, bootstrap, and the fork pool remain in the module and are not called by this path. |
| `src/pipeline.py` | Required-reranker orchestration, caches, atomic publish, predict-only loader. |
| `tests/test_ranking_gates.py`, `tests/test_reranker_experiment.py`, `tests/test_execution_workflow.py`, `tests/test_accuracy_fixes.py` | Synthetic gates (`python -m unittest discover -s tests`). No competition data, no fitting on real data. |
| `scripts/make_kaggle_notebook.py` | Emits the three identical train-and-predict notebooks and `kaggle_predict_only.ipynb`. |
| `scripts/build_class2_np.py` | Rebuilds the COCONUT ∪ LOTUS parquet. Needs `train.parquet` locally, which is not on disk. |
| `kaggle_dataset/class2_candidates.parquet` | The 452,608-row library (also on Kaggle) |
| `kaggle_gpu_submission.ipynb` | Train-and-predict notebook to upload (GPU, internet off). Identical to `kaggle_submission.ipynb` and `kaggle_submission_tpu.ipynb`. |
| `kaggle_predict_only.ipynb` | Ranks from an attached verified `casmi_run/` dataset. Does not train. |

## Scored experiment — peak-support ranking (0.150)

Reported 2026-10-07. Public MRR@25 **0.150**, and that submission is the new best. The screenshot is Kaggle version label V5. No run log was attached, so the lock count and `class2_slots` are still unknown.

The change versus 0.139 was ranking only. Training stayed focal BCE, 3 seeds, 120k structures, 4 epochs, neighbor blend on, decoder off. `USE_REPRESENTATIVE_METADATA` and `RESTORE_BEST_CHECKPOINT` stayed false.

1. Modified cosine is 0 unless at least 4 peaks match and those peaks cover at least 15% of query intensity. Supported matches are scaled by `sqrt(n_hits / 8)`.
2. A Class 1 lock requires cosine ≥ 0.75 and at least 5 matched peaks.
3. Class 2 and other hits with no verified spectrum score `0.92 * (0.78 * tanimoto + 0.22 * mass)`. Verified spectral hits still use `0.42 / 0.46 / 0.12`.
4. InChIKey14 dedup runs `TautomerEnumerator.Canonicalize` before `MolToInchiKey`.

Keep this submission selected. Do not retune these four gates in the next run. The gain does not establish a path to 0.25 or 0.350.

## Next scored run — `required-reranker-v1` (not yet scored)

**Hypothesis.** The hand-written compete score is the bottleneck for Class 2: the right skeleton is usually in the pool but not at rank 1. The current 22-feature HistGradientBoosting reranker, applied to the top 200 unlocked candidates, is the only change versus the 0.150 ordering. Kaggle MRR@25 decides whether it is better. This run does not select the old baseline if the fit succeeds, and it does not claim 0.350.

**Frozen.** Spec2FP at 120k structures, four completed epochs, seeds 42/7/123, current architecture and batch size, focal BCE, last-epoch selection, pre-sigmoid logit averaging. Neighbor k=20, alpha=0.5. Current peak-support gates. Full 452,608-row pool. Decoder off. Weak Class 1 stays eligible. Same candidate generation, mass windows, and analog generation. Locked prefix and its order stay put.

**Not repeated.** Dropping weak Class 1 (0.119), neighbor blend off (0.133), all-structure six-epoch training (0.126), representative-metadata training (0.134), and the rejected candidate-ranking loss (control 0.137).

**Reranker.** `max_iter=150`, `lr=0.05`, `max_leaf_nodes=15`, `max_depth=4`, `min_samples_leaf=40`, `l2=1.0`, `random_state=0`, no early stopping, monotonic constraints, query-balanced weights. Features stay the current 22 inference-only names. No answer fingerprint, no formula-derived query mass, no identity, no validation label.

**Training pairs.** Only `RR_TRAIN` (1200) and regimes U and K. U masks the library and the neighbor slice and inserts the truth into a copy of the external pool. K uses the full library, the production external pool, and an alternate acquisition. Pair features are cached under a signature of source, fit hash, library row order, checkpoint bytes, Class 2 bytes, features, booster params, window, and pair-source hash. Public test changes and operational paths do not invalidate that cache. DEV, AUDIT, PANEL, bootstrap, permutation importance, and the promotion rule are not executed. `PROMOTION_RULE` remains in `src/validation.py` for the old tests; it is not the submission selector.

**What to read.** `STRATEGY_ACTIVE` before prediction, `SUBMISSION_READY` after the CSV check, and `SUMMARY_JSON` (strategy id, source hash, checkpoint hashes, complete seeds, reranker status, submission hash, row count, timings). There is no `DECISION` line and no selected-arm field.

## Inactive experiments (not in the scored run)

**`precursor-mz-shift-v1`.** `_score_candidates` passes `cand_mass[pick]` (neutral exact mass) into `modified_cosine`, which subtracts it from the query precursor. The index now also stores the observed representative `precursor_mz` and never fills it from an answer formula. `cfg.use_observed_precursor` defaults to false. Turning it on uses the stored m/z and raises when that value is missing or non-finite. A synthetic pair (query peaks 100–130 at precursor 500, candidate peaks 50–80, neutral mass 450) scores above 0 on the default path and 0 when the flag is on. The scored recipe rejects the flag.

**`official-inchikey14-v14`.** `official_metric_inchikey14` follows the version-14 sequence: `MolFromSmiles`, `RemoveStereochemistry`, a fresh default `TautomerEnumerator.Canonicalize`, then `MolToInchiKey[:14]`. Invalid SMILES returns `""`. Production dedup still calls `inchikey14_from_smiles`. On this laptop (RDKit 2026.03.6) the two functions return the same key for `C[C@H](O)CC` / `C[C@@H](O)CC`, for `CC(=O)CC(=O)C` / `CC(=O)C=C(C)O`, and for `CCO`. That does not measure the 452,608-row pool and does not measure RDKit 2026.03.3, which is not installed here. No label, dedup, or ordering change was applied. Do not pip-upgrade RDKit to run this comparison.

## What was not measured

- Public MRR@25 of `required-reranker-v1`. Keep 0.150 selected until a new public score is strictly higher.
- Kaggle GPU wall time, peak memory, RR_TRAIN yield, and Class 2 load time. The manifest records them only after a real Kaggle run. Do not treat a synthetic test as that measurement, and do not cut epochs or seeds to fit a guessed timeout.
- Whether `official_metric_inchikey14` differs from production dedup on the Class 2 pool or under RDKit 2026.03.3.
- Fork-pool candidate generation on Kaggle Linux. This path does not use it. A local macOS fork pool hung after Torch/OpenMP initialized in the parent; that is why ranking here is one process.

## Rules for the next change

1. One hypothesis per Kaggle submit. A full run is ~5–9 h and costs a submission.
2. Keep the 0.150 merge unless a new idea replaces it with a measured reason: lock only when cosine ≥ 0.75 and at least 5 peaks match; score a missing spectrum as `0.92 * (0.78 * tanimoto + 0.22 * mass)`.
3. Keep neighbor blend on (`k=20`, `a=0.5`).
4. Keep Spec2FP at 120k / 4 epochs / 3 seeds unless the change is specifically a better training objective, not “train longer”.
5. Do not enable the SMILES decoder.
6. Do not filter Class 2 on public `test.parquet`.
7. Any new structure library must be a new Kaggle dataset (packed fp parquet, 50–2000 Da, train keys removed). NPAtlas adds only ~2.5k structures on top of the current 453k and is not worth a run by itself.
8. After editing `src/`, run `python scripts/make_kaggle_notebook.py` and `python -m unittest discover -s tests`. Upload `kaggle_gpu_submission.ipynb` for a from-scratch run, or `kaggle_predict_only.ipynb` only with a verified `casmi_run` dataset.
9. Leave the 0.150 submission selected on Kaggle until a new public score is strictly higher.
10. Reranker features never include the answer's fingerprint, a formula-derived query mass, identity, or validation labels. The scored run does not use a local promotion gate. `precursor-mz-shift-v1` and `official-inchikey14-v14` stay off until each has its own run.
11. Any further change to the reranker (features, window, estimator), the precursor shift, or the identity function is a new single-hypothesis experiment with its own id.
