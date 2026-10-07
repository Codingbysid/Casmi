# Enveda CASMI 2026 — model brief

This file is the source of truth for the current pipeline. Read it before proposing changes.
The Kaggle notebook is generated from `src/` by `scripts/make_kaggle_notebook.py`. Edit `src/` and the generator, then regenerate `kaggle_submission.ipynb` and `kaggle_gpu_submission.ipynb` (same content). Do not hand-edit the notebooks. `tests/test_reranker_experiment.py` fails when the embedded `src/` no longer matches the working tree.

## Task

Kaggle: **Enveda CASMI 2026 — Molecule ID from Mass Spectra**.

For each unknown molecule, rank up to **25 SMILES**. The metric is **MRR@25** on the first 14 characters of the RDKit InChIKey (InChIKey14, the connectivity skeleton). Stereochemistry and the second InChIKey block do not count. Exactly one row per `molecule_id`. Guesses are semicolon-separated SMILES. Empty rows are invalid.

RDKit versions matter for the key. Our dedup key is `TautomerEnumerator.Canonicalize` → InChIKey block 1 (`inchikey14_from_smiles`). The evaluation page reportedly scores with RDKit 2026.03.3 (the page is JS-rendered and could not be re-read offline); the notebook installs the 2025.09.5 wheel; the laptop has 2026.03.6. Measured on the Class 2 pool: 10% of raw `inchikey14` values change under tautomer canonicalization, canonicalization costs ~43 ms per SMILES versus ~0.4 ms for the heavy-atom graph key (`heavy_atom_graph_key`, a verified necessary condition for canonical-key equality: 0 mismatches over 487 tautomer-changed structures). Production dedup canonicalizes only when two candidates share a graph key; the output is identical to canonicalizing everything.

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
3. Train **3 Spec2FP seeds** (42, 7, 123) on the fixed 120k fitting subset (`fitting_subset_indices`, seed 42). Average **pre-sigmoid logits**, then sigmoid. Every seed must reach `completed_epochs == 4`; a time limit now raises instead of ranking with a partial ensemble. Checkpoints record `fit_subset_hash`, `model_config_hash`, `seed`, `completed_epochs`, `stop_reason`, and an attached previous run is reused only when all of them match.
4. Blend the predicted fingerprint with a spectral-neighbor fingerprint: binned-spectrum cosine kNN, **k=20**, against train spectra, **alpha=0.5** (`src/neighbors.py`).
5. `rank_molecules` in `src/infer.py`:
   - **Tier 1 lock.** Class 1 hits stay at the top only when modified cosine is **≥ 0.75** and at least **5** peaks matched. Matches with fewer than 4 peaks, or under 15% of query intensity, have cosine 0.
   - **Tier 2 compete.** Verified spectral hits use `0.42 * spec + 0.46 * tanimoto + 0.12 * mass`. Class 2 and analogs have no spectrum, so they use `0.92 * (0.78 * tanimoto + 0.22 * mass)`. One slot per tautomer-canonical skeleton (`_emit_unique`, graph-key-gated canonicalization, output-equivalent to the reference `_merge_hits`).
   - **Experiment arm (pending, see below).** When a fitted reranker is passed, only the top `rerank_window=200` unlocked candidates (by the compete score) are reordered by the learned score; the tail keeps the compete order and locked hits never move. With `reranker=None` the output is the 0.150 ordering.
   - **Tier 3 pad** to 25 with SMARTS expansion, then nearest library SMILES. The SMILES decoder is **off** (`TRAIN_DECODER = False`). It emits garbage and is not used.
6. Write `/kaggle/working/submission.csv` with columns `molecule_id`, `smiles`.

Modified cosine is GNPS-style: a peak matches if it is within 0.05 Da **or** matches after the precursor-mass shift (`src/retrieval.py`).

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
- 3 seeds, each capped at **1.5 h**; a timeout is a hard failure, not a quiet fallback. On a Kaggle GPU a seed takes roughly 15–25 min.

Config defaults in `src/config.py` (`d_model=256`, 8 epochs, 400k spectra) are **not** what Kaggle runs. The notebook cell is.

## Current notebook knobs (`scripts/make_kaggle_notebook.py`)

```
REQUIRE_CLASS2 = True
REQUIRE_CUDA = True          # GPU session required unless FAST_DEV_RUN
FAST_DEV_RUN = False
PREFER_XLA = False           # no TPU
TRAIN_DECODER = False
ENSEMBLE_SEEDS = 3
MAX_TRAIN_SPECTRA = 120_000
NUM_EPOCHS = 4
BATCH_SIZE = 64
TRAIN_TIME_LIMIT_S = 1.5h    # per seed
USE_NEIGHBOR_FP = True
NEIGHBOR_BLEND = 0.5
SESSION_BUDGET_S = 8.5h
RANK_RESERVE_S = 1h          # kept free for production ranking + artifacts

RERANKER_EXPERIMENT = True   # False = skip validation, ship the exact 0.150 ordering
RERANK_WINDOW = 200
REUSE_CHECKPOINTS = True     # only after fit-subset / config / seed / epoch checks
RANK_WORKERS = 4             # fork workers for validation candidate generation
VALIDATION_SEED = 0
N_RR_TRAIN, N_DEV, N_AUDIT, N_PANEL = 1200, 400, 800, 400
VALIDATION_BUDGET_S = 3h
```

Kaggle constraints: **GPU session (P100/T4), internet off**, kernel source **< 1 MB** (the notebook embeds `src/` as JSON; Class 2 is an attached dataset, never embedded), Python 3.12, Run All must finish inside **9 h** and write `submission.csv`. The run leaves `/kaggle/working/casmi_run/` in place (checkpoints, `reranker.pkl`, `run_manifest.json`, `splits.json`, `validation_results.json`, `candidate_traces.parquet`, `diagnostics/`, `casmi_diagnostics.zip`); nothing is deleted at the end, and `submission.csv` is the only CSV at the working root.

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
| `src/rerank.py` | Unlocked-window reranker: 22 inference-only features, fixed `HistGradientBoostingClassifier` with monotonic constraints, query-balanced weights, refit-determinism check, save/load. Used only when the notebook's promotion rule passes. |
| `src/validation.py` | Leak-free validation harness: tautomer-aware partitions outside the fit subset, acquisition pass, masked library, answer-in-pool external copy, fork-pool candidate generation with traces, both arms on identical pools, bootstrap CIs, `PROMOTION_RULE`, checkpoint completeness guard, strict submission check. |
| `tests/test_ranking_gates.py`, `tests/test_reranker_experiment.py` | Synthetic gates (`python -m unittest discover -s tests`). No competition data, no fitting on real data. |
| `scripts/make_kaggle_notebook.py` | Emits `kaggle_submission.ipynb`, `kaggle_gpu_submission.ipynb` (and the legacy `kaggle_submission_tpu.ipynb` name; all identical GPU notebooks) |
| `scripts/build_class2_np.py` | Rebuilds the COCONUT ∪ LOTUS parquet. Needs `train.parquet` locally, which is not on disk. |
| `kaggle_dataset/class2_candidates.parquet` | The 452,608-row library (also on Kaggle) |
| `kaggle_submission.ipynb` / `kaggle_gpu_submission.ipynb` | What gets uploaded to Kaggle (GPU, internet off) |

## Scored experiment — peak-support ranking (0.150)

Reported 2026-10-07. Public MRR@25 **0.150**, and that submission is the new best. The screenshot is Kaggle version label V5. No run log was attached, so the lock count and `class2_slots` are still unknown.

The change versus 0.139 was ranking only. Training stayed focal BCE, 3 seeds, 120k structures, 4 epochs, neighbor blend on, decoder off. `USE_REPRESENTATIVE_METADATA` and `RESTORE_BEST_CHECKPOINT` stayed false.

1. Modified cosine is 0 unless at least 4 peaks match and those peaks cover at least 15% of query intensity. Supported matches are scaled by `sqrt(n_hits / 8)`.
2. A Class 1 lock requires cosine ≥ 0.75 and at least 5 matched peaks.
3. Class 2 and other hits with no verified spectrum score `0.92 * (0.78 * tanimoto + 0.22 * mass)`. Verified spectral hits still use `0.42 / 0.46 / 0.12`.
4. InChIKey14 dedup runs `TautomerEnumerator.Canonicalize` before `MolToInchiKey`.

Keep this submission selected. Do not retune these four gates in the next run. The gain does not establish a path to 0.25 or 0.350.

## Pending experiment — learned reranker for the unlocked pool (built 2026-10-07, not yet scored)

**Hypothesis.** The hand-written compete score (`0.92 * (0.78 * tanimoto + 0.22 * mass)` for ~2,300 Class 2 candidates per query, `0.42/0.46/0.12` for weak Class 1) is the bottleneck for Class 2 molecules: the right skeleton is usually in the pool but not at rank 1. A small, regularized, source-aware gradient-boosted reranker applied to the top-200 unlocked candidates should place it higher. This is the only change versus 0.150.

**Frozen.** Spec2FP recipe (focal BCE, 3 seeds, 120k, 4 epochs), neighbor blend, spectral gates, the Class 1 lock, candidate generation, analog generation, dedup, RDKit 2025.09.5, decoder off.

**Reranker** (`src/rerank.py`). sklearn `HistGradientBoostingClassifier`, one fixed configuration (`max_iter=150`, `lr=0.05`, `max_leaf_nodes=15`, `max_depth=4`, `min_samples_leaf=40`, `l2=1.0`, `random_state=0`, no early stopping), monotonic constraints on the similarity features, query-balanced sample weights, refit-determinism check. Features are per-candidate inference quantities only: spectral cosine / matched peaks / intensity fraction / evaluated flag, Tanimoto, fingerprint cosine, mass Gaussian, |ppm|, source one-hots (Class 1 / Class 2 / analog), the 0.150 linear score and its log-rank, margins to the pool maximum, pool-size context, and query context (mass, peak count, adduct groups). No answer fingerprint, no formula-derived query mass, no identity, no validation label.

**Validation inside the Kaggle run** (`src/validation.py`). Holdout skeletons are drawn from the ~156k library rows outside the 120k fitting subset, keyed by tautomer-canonical InChIKey14, with any row whose tautomer alias sits in the fit subset excluded. Partitions are disjoint: `RR_TRAIN` 1200 (fit the reranker), `DEV` 400 (sanity only, never selection), `AUDIT` 800 (promotion decision), `PANEL` 400 (`enveda-np-examples`, Bruker, where eligible). Two regimes per skeleton: **U** (unseen: representative spectrum as the query, identity and aliases removed from the spectral library and the neighbor index, truth inserted into a validation copy of the Class 2 pool) and **K** (known: a different acquisition as the query, full library). Candidate pools are generated once with the production code; the rerank arm is reconstructed offline from the trace and verified equal to the production path. Diagnostics: coverage (truth locked / in window / in pool), false locks, MRR@25, Recall@1/5/25, truth-in-window-only slice, paired RR deltas with 4096-resample bootstrap CIs, per-query rows, candidate traces.

**Promotion rule (declared before the audit runs, `PROMOTION_RULE`):** AUDIT ≥ 300 ranked queries; pooled U+K paired ΔMRR ≥ +0.010; bootstrap 95% CI lower bound > 0; regime-K Δ ≥ −0.005; PANEL Δ ≥ −0.02 when PANEL has ≥ 50 queries; a reranker must have been fitted. Any failed or unmeasurable check ships the exact 0.150 ordering, and the manifest says why.

**What to read from the run.** `SUMMARY_JSON` on the last line (selected arm, audit MRR for both arms, CI, panel size, versions), `[rank:U]`/`[rank:K]` rates, `[holdout]` exclusions, the `[ok]/[FAIL]` lines under `DECISION`, and `test molecules where the two arms differ`. The validation estimate is not the public score: U uses library representatives with the truth inserted into the pool, so it bounds Class 2 behaviour from above; the hidden Class 3 share scores 0 in both arms.

**Measured locally (synthetic data only).** 26 unit tests pass; a `FAST_DEV_RUN` dry run of the generated notebook on synthetic spectra executes every cell, including fitting, audit, decision, production ranking, artifacts, and the strict submission check. Nothing was trained or fitted on competition data locally. No public score exists for this experiment yet.

## Rules for the next change

1. One hypothesis per Kaggle submit. A full run is ~5–9 h and costs a submission.
2. Keep the 0.150 merge unless a new idea replaces it with a measured reason: lock only when cosine ≥ 0.75 and at least 5 peaks match; score a missing spectrum as `0.92 * (0.78 * tanimoto + 0.22 * mass)`.
3. Keep neighbor blend on (`k=20`, `a=0.5`).
4. Keep Spec2FP at 120k / 4 epochs / 3 seeds unless the change is specifically a better training objective, not “train longer”.
5. Do not enable the SMILES decoder.
6. Do not filter Class 2 on public `test.parquet`.
7. Any new structure library must be a new Kaggle dataset (packed fp parquet, 50–2000 Da, train keys removed). NPAtlas adds only ~2.5k structures on top of the current 453k and is not worth a run by itself.
8. After editing `src/`, run `python scripts/make_kaggle_notebook.py` and upload the new `kaggle_submission.ipynb` (or `kaggle_gpu_submission.ipynb`; identical). Then run `python -m unittest discover -s tests`.
9. Leave the 0.150 submission selected on Kaggle until a new public score is strictly higher.
10. Reranker features never include the answer's fingerprint, a formula-derived query mass, identity, or validation labels. `DEV` is never used for selection. The promotion rule is fixed before the audit runs; do not relax it after seeing the numbers.
11. Any further change to the reranker (features, window, estimator) is a new single-hypothesis experiment with its own run.
