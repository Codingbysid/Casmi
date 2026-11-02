# Enveda CASMI 2026 — model brief

This file is the source of truth for the current pipeline. Read it before proposing changes.
The Kaggle notebook is generated from `src/` by `scripts/make_kaggle_notebook.py`. Edit `src/` and the generator, then regenerate `kaggle_submission.ipynb`. Do not hand-edit the notebook.

## Task

Kaggle: **Enveda CASMI 2026 — Molecule ID from Mass Spectra**.

For each unknown molecule, rank up to **25 SMILES**. The metric is **MRR@25** on
InChIKey14 **after RDKit tautomer canonicalization**, pinned by Kaggle to
**RDKit 2026.03.3**. Stereochemistry and tautomer forms do not count as distinct
answers under that procedure. Exactly one row per `molecule_id`. Guesses are
semicolon-separated SMILES. Empty rows are invalid.

**Metric audit, 2026-10-06:** the official
[Evaluation page](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/overview)
specifies the normalization above. Our current `src/chem.py` helper directly
calls `Chem.MolToInchiKey` without a `TautomerEnumerator` step, and the measured
notebook used RDKit 2025.09.5. The local metric, candidate deduplication, and
holdout identity checks therefore do not yet match the documented competition
procedure. Canonical SMILES alone does not supply that missing normalization.
Source behavior has not been changed in this audit; the official metric notebook
implementation and the pinned offline dependency still need verification before
a metric-alignment change. Reassess all local MRR results, including the apparent
merge gap, with that procedure before promoting a scoring change.

Earlier approximate hidden-test mix (unverified planning figures, not labels):

- **~16% Class 1** — the true structure is in `train.parquet` (same skeleton, spectrum may be from another instrument).
- **~45% Class 2** — not in train, but a known structure. The competition's PubChem/COCONUT coverage is broader than our COCONUT/LOTUS candidate file; presence in our library is not guaranteed.
- **~39% Class 3** — novel structures. Our current retrieval cannot recover an absent skeleton unless analog generation happens to produce it.

The current official [Data page](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/data)
says the class distribution is hidden. Do not treat the percentages above as
established proportions or use them to justify expected leaderboard gains.

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
| **V5 and V6** | **Best.** Lock cosine ≥ 0.75, then weak Class 1 competes with Class 2. Neighbor blend on. Spec2FP 120k structures, 4 epochs, 3 seeds, ~1.5 h/seed. | **0.139** |
| V7 | Weak Class 1 dropped; remaining slots Class 2 only | 0.119 |
| V8 | Same as 0.139 but neighbor blend off | 0.133 |
| V9 | Spec2FP on all ~276k skeletons, 6 epochs | 0.126 |
| Metadata experiment | Preserve representative spectrum metadata in training; supplied screenshot and run log, 2026-10-03 | 0.134 |
| Best-checkpoint change | Implemented; supplied runs do not verify this experiment with the switch on | No demonstrated gain |
| GPU candidate-ranking comparison | Two complete three-seed ensembles; candidate loss failed the holdout gate, so the focal-only control was submitted | **0.137**, user reported 2026-10-06; candidate ensemble has no public score |

**0.139 is the recipe to beat.** Do not retune the lock, drop weak Class 1, turn neighbor blend off, or train Spec2FP longer/harder. Those four experiments all lost.

**User's target is >0.350.** Aggregate MRR does not identify classwise accuracy:
0.139 cannot establish that Class 2 or Class 3 is near zero. Candidate coverage,
unseen-structure fingerprint ranking, and competition with weak spectral hits
are separate unknowns. Neither engineering fix above demonstrated a score gain.

## Pipeline (inference)

1. Install RDKit from the attached wheel. Fail if `from rdkit import Chem` does not work.
2. Build a train **StructureIndex**: one representative MS2 per InChIKey14 (the row with the most peaks), exact mass from formula, Morgan fingerprint, top-128 peaks.
3. Train **3 Spec2FP seeds** (42, 7, 123). Average **pre-sigmoid logits**, then sigmoid.
4. Blend the predicted fingerprint with a spectral-neighbor fingerprint: binned-spectrum cosine kNN, **k=20**, against train spectra, **alpha=0.5** (`src/neighbors.py`).
5. `rank_molecules` in `src/infer.py`:
   - **Tier 1 lock.** Class 1 hits with modified cosine **≥ 0.75** stay at the top, sorted by cosine. Constant `CLASS1_LOCK_COSINE = 0.75`.
   - **Tier 2 compete.** Everything else (weaker train hits, Class 2 COCONUT hits, mass-shifted analog products) is scored with `0.42 * spec + 0.46 * tanimoto + 0.12 * mass_gaussian` and sorted together. Class 2 and analogs have `spec = 0`.
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

A CLS token over the peak transformer, the binned vector, and the precursor/adduct vector are concatenated and mapped to **2048 logits** (Morgan/ECFP4, radius 2). Baseline loss is multi-label focal BCE (`gamma=1.5`, `pos_weight=6`). The new paired experiment adds a mass-matched candidate-ranking loss, described below; it does not alter this network.

**Kaggle notebook overrides the config defaults** to the original smaller net,
including on GPU:

- `d_model=128`, `n_heads=4`, `n_transformer_layers=2`, `n_conv_channels=64`
- `batch_size=64`
- `MAX_TRAIN_SPECTRA=120_000` unique library rows (random subset of ~276k)
- `NUM_EPOCHS=4`
- Train two complete ensembles, each with seeds 42, 7, 123: six fits in total,
  with three members in the selected ensemble. Every fit must finish four epochs.
- One independent fit per GPU, at most two concurrently. A single GPU runs them
  sequentially. Each fit is capped at **1.5 h**, or a smaller equal cap if the
  available session budget requires it. Session wall budget: 8.5 h.
- No automatic model enlargement, larger batch, AMP, or TF32. Both arms use FP32.

Config defaults in `src/config.py` (`d_model=256`, 8 epochs, 400k spectra) are **not** what Kaggle runs. The notebook cell is.

## Current notebook knobs (`scripts/make_kaggle_notebook.py`)

```
REQUIRE_CLASS2 = True
FAST_DEV_RUN = False
PREFER_XLA = False          # no TPU
TRAIN_DECODER = False
ENSEMBLE_SEEDS = 3
MAX_TRAIN_SPECTRA = 120_000
NUM_EPOCHS = 4
BATCH_SIZE = 64
USE_NEIGHBOR_FP = True
NEIGHBOR_BLEND = 0.5
PER_SEED_MAX_S = 1.5h
USE_REPRESENTATIVE_METADATA = False  # restore the 0.139 feature construction
RESTORE_BEST_CHECKPOINT = False     # no demonstrated gain; baseline behavior
CANDIDATE_LOSS_WEIGHT = 0.05        # single new hypothesis, subject to paired gate
CANDIDATE_TEMPERATURE = 0.05
CANDIDATE_NEGATIVES = 32
RETRIEVAL_HOLDOUT_N = 512
MIN_PAIRED_QUERIES = 256
PAIRED_EVAL_MAX_S = 60min
MAX_PARALLEL_FITS = 2
RANK_RESERVE_S = 2h
SESSION_BUDGET_S = 8.5h
EXPERIMENT_ID = "casmi-gpu-ensemble-rank-v2"
```

Kaggle constraints: **CUDA GPU required**, **internet off**, kernel source
**< 1 MB** (the notebook embeds `src/` as JSON; Class 2 is an attached dataset,
never embedded), Python 3.12. The first cell fails on CPU before installing RDKit
or reading data. GPU runtime is not yet measured; the notebook reserves time
inside an 8.5-hour budget and rejects incomplete training instead of silently
including partial fits. A 30-hour GPU allowance is not one 30-hour session.

## Code map

| Path | Role |
|---|---|
| `src/config.py` | Hyperparameters, adduct table, paths |
| `src/chem.py` | Formula mass, adducts, Morgan, InChIKey14, fp pack/unpack |
| `src/preprocessing.py` | Peak clean, binning, per-molecule aggregation |
| `src/data.py` | Parquet streaming, `dataset_from_structure_index` |
| `src/model.py` | `Spec2FP`, focal BCE |
| `src/candidate_training.py` | Packed, mass-matched decoys for the single ranking-loss experiment |
| `src/retrieval_validation.py` | Predicted-fingerprint, skeleton-disjoint paired retrieval evaluation |
| `src/tpu_trainer.py` | CPU/GPU training loop, seed and time-limit overrides |
| `src/gpu_training.py` | Fresh GPU worker processes, shared array cache, six-fit scheduler, completion guards |
| `src/retrieval.py` | `StructureIndex`, modified cosine, mass window |
| `src/neighbors.py` | Binned cosine kNN fingerprint blend |
| `src/ranker.py` | Linear score, dedup, submission CSV |
| `src/infer.py` | **`rank_molecules` / `_merge_tiers` — the 0.139 merge** |
| `src/analogs.py` | Mass-shift parent retrieval + SMARTS |
| `src/calibration.py` | Optional OR-merge probabilities. Ranking score used in the compete pool is the linear mix above, not only this table. |
| `src/models/smiles_decoder.py` | Class 3 decoder. **Disabled.** |
| `src/formula_head.py`, `src/rerank.py` | Built earlier, **disabled** in `predict_test` |
| `scripts/make_kaggle_notebook.py` | Emits the GPU notebook and identical legacy filename aliases |
| `scripts/build_class2_np.py` | Rebuilds the COCONUT ∪ LOTUS parquet. Needs `train.parquet` locally, which is not on disk. |
| `kaggle_dataset/class2_candidates.parquet` | The 452,608-row library (also on Kaggle) |
| `kaggle_gpu_submission.ipynb` | Explicit GPU upload filename; identical to `kaggle_submission.ipynb` |

## Rules for the next change

1. One hypothesis per Kaggle submit. A full run is ~5–9 h and costs a submission.
2. Keep the V5 merge unless the new idea replaces it with a measured reason: lock cosine ≥ 0.75, then compete weak Class 1 with Class 2 by `0.42/0.46/0.12`.
3. Keep neighbor blend on (`k=20`, `a=0.5`).
4. Keep Spec2FP at 120k / 4 epochs / 3 seeds unless the change is specifically a better training objective, not “train longer”.
5. Do not enable the SMILES decoder.
6. Do not filter Class 2 on public `test.parquet`.
7. Any new structure library must be a new Kaggle dataset (packed fp parquet, 50–2000 Da, train keys removed). NPAtlas adds only ~2.5k structures on top of the current 453k and is not worth a run by itself.
8. After editing `src/`, run `python scripts/make_kaggle_notebook.py` and upload the new `kaggle_submission.ipynb`.
9. Leave the existing 0.139 submission selected on Kaggle until a new public score is strictly higher.

## Previous experiment — representative feature fidelity (reported 0.134)

The single change is to preserve the representative spectrum's observed training
features. In the 0.139 code, `dataset_from_structure_index` discarded the original
metadata and assigned **every** training row adduct ID 0 (`[M+H]+`), positive ion
mode, and zero collision energy. It also computed `peak_nl` from neutral mass,
whereas inference's `featurize_spectrum` uses observed precursor m/z. Thus sodium,
negative-mode, and multiply charged spectra were presented with inconsistent
metadata, and nonzero adduct embeddings received no example-driven training.

`StructureIndex` now retains `peak_nl`, `precursor_feat`, and `adduct_id` from the
same winning representative as its peaks. The training dataset copies these
tensors using the existing sample indices. No extra train-parquet pass is needed;
the added arrays use about 152 MB for 276k structures at 128 peaks.

- Experiment switch: `USE_REPRESENTATIVE_METADATA = True` (config:
  `use_representative_metadata`). It is now `False` for the original 0.139 feature path.
- Keep 120k / 4 epochs / seeds 42, 7, 123, the model and focal loss, neighbor
  k=20 / alpha=0.5, lock 0.75, and the 0.42/0.46/0.12 compete score unchanged.
- No new candidate library, test-mass prefilter, decoder, or reranker.
- Source edits: `src/retrieval.py`, the library builders in `src/infer.py`,
  `src/data.py`, `src/config.py`, and `scripts/make_kaggle_notebook.py`.
- Synthetic feature-fidelity tests: `python -m unittest discover -s tests -v`.
  These do not train a model or measure competition MRR.
- The user supplied a **0.134** score and a log with this switch on. It did not
  beat the recorded 0.139 baseline. Do not combine it with the next experiment.

## Supplied run audit (2026-10-03)

- RDKit installed successfully. Train index: 275,810 structures. All 452,608
  Class 2 structures loaded. Three seeds were averaged and neighbor k=20,
  alpha=0.5 ran. Decoder stayed off. Submission had 400 nonempty rows, 25 guesses.
- Every seed reached its 5,400-second CPU cap during epoch 3; **four epochs were
  configured, not completed**. Last logged steps were 4,100 / 4,150 / 4,000;
  there are 1,782 training batches per epoch. Training took 16,211.1 seconds;
  submission was written at 17,616.1 seconds (~4.89 h).
- Seed 123 deteriorated from epoch-1 val_loss 0.0991 / bit F1 0.4050 to
  epoch-2 val_loss 0.2683 / bit F1 0.0597. Its partial epoch 3 was not evaluated.
- The trainer saved improving validation checkpoints but **overwrote the best
  on timeout and normal completion**, then returned the latest weights. Thus
  the ensemble did not necessarily contain the best validated state of any seed.
- The notebook's 0.6931 MRR uses train skeletons still in the retrieval index.
  It is not unseen-structure validation. The 340/400 locks and Class 2 slot counts
  in the visible run likewise do not establish hidden-test class performance.
- `scripts/eval_local.py` supplies true fingerprints to its Class 2 queries;
  those results are oracle-fingerprint diagnostics, not Spec2FP retrieval MRR.

## Supplied CPU run audit (2026-10-05)

The new pasted run is still the representative-metadata notebook, not the
candidate-ranking comparison or a verified checkpoint-selection experiment.

- `USE_REPRESENTATIVE_METADATA=True`, `torch 2.10.0+cpu`, training device CPU.
  No candidate-ranking modules or paired retrieval gate appear in that notebook.
- Train library: 275,810 structures; fitting dataset: 120,000; per-seed split:
  114,000 fit / 6,000 validation. All 452,608 external structures loaded.
- Every seed stopped at the 5,400-second limit during **epoch 4**, so none
  completed the configured four epochs. Total training: 16,206.0 seconds.
  Submission was written at 16,936.1 seconds (~4.70 h).
- Seed 123's poor epoch 2 (val_loss 0.2408 / F1 0.0701) recovered at epoch 3
  (0.0931 / 0.4344). The bad intermediate epoch alone does not establish that
  its final weights were bad.
- The reported sanity MRR 0.6941 uses train skeletons retained in the retrieval
  index; it is not unseen-structure MRR. The 340/400 locks and 198 Class 2 slots
  are public-placeholder statistics, not hidden-test classwise performance.
- Final cleanup retained only `submission.csv`; that notebook did not create
  the diagnostic archive requested for the new experiment.
- The supplied latest score screenshot shows 0.134, below the recorded 0.139.
  The supplied evidence does not measure the new candidate loss on Kaggle.

The GPU notebook requires CUDA before doing work, logs a unique run ID/source
hash, verifies completed epochs/optimizer steps for all six fits, and retains
the reports, model checkpoints, and worker logs.

## Latest supplied GPU run audit (2026-10-06)

Source: `Pasted text(2).txt`; public score **0.137** supplied by the user.
Run ID: `casmi-gpu-ensemble-rank-v2_7931f9a1e0`.
Embedded source SHA-256:
`6f5f99a7016b40bb7105826c82cfd7697fc2d1f63f2870917e15e850b9c87a50`.
Submission SHA-256:
`cded13817fefd8025608595b1eda76ae2efd3d44577111bc2580acd956a78d8f`.

- Correct GPU notebook executed: torch 2.10.0+cu128, CUDA 12.8, two Tesla T4s.
  Metadata and best-checkpoint experiments remained off. All six fits completed
  four epochs and 7,128 optimizer steps each. No CPU fallback or partial fits.
- Control fit times: 300.0 / 295.0 / 294.9 seconds. Candidate fit times:
  375.8 / 372.8 / 360.3 seconds. The final submission was written at 3,815.7
  seconds (~63.6 minutes), before final archive packaging.
- Same 120k fitting structures, 452,608 external candidates, neighbor k=20,
  alpha=0.5, original tier merge, and decoder off. Decoys were available for
  99.63% of fitting rows, so the new objective was not a no-op.
- All 512 paired holdout queries completed; the audit reports zero overlap with
  fitting, validation spectral, and validation neighbor skeletons. Truth
  structures were inserted into the evaluation candidate pool by design.

| Holdout diagnostic | Focal control | Candidate loss |
|---|---:|---:|
| External-candidate-only MRR@25 | 0.6697015 | 0.6714489 |
| Full-pipeline MRR@25 | 0.0491481 | 0.0486009 |

Full-pipeline paired delta: **-0.0005472**; paired bootstrap 95% interval:
**[-0.0017802, +0.0005847]**. The experiment did not demonstrate improvement;
the interval includes zero. Truth mass-window recall was 1.0 on this injected,
known-candidate benchmark, not a measurement of hidden-test library coverage.

The gate correctly retained the **control** ensemble. The reported public
**0.137 belongs to that control submission**, not to the candidate-loss model.
Do not attribute the 0.002 gap from the historical best to candidate loss or
declare its cause from the aggregate score. Keep 0.139 selected.

The large candidate-only/full-pipeline gap points to competition introduced by
the final merge under our current raw-key metric. The newly identified tautomer
normalization mismatch means some apparent wrong hits may be equivalent under
the competition metric; its contribution is not yet measured. The aggregate log
cannot separate incorrect strong Class 1 locks, weaker train hits, and analog
competition. First align metric identities and holdout exclusions. Do not respond
by dropping weak Class 1 (already measured worse), disabling neighbors, adding
epochs, or repeating the same candidate-loss fit. First reuse the saved control
checkpoints to trace truth rank and competitors through the merge, and include
Class 1 transfer validation before promoting any merge change.

Saved outputs include `run_manifest.json`, `retrieval_validation.json`, and
`casmi_diagnostics.zip` (28,166,175 bytes), containing all six checkpoints and
worker logs. The per-query JSON/archive has not yet been supplied in this chat.

## Prepared change — best validation checkpoint (execution unverified)

One change against 0.139: `RESTORE_BEST_CHECKPOINT=True` returns and persists the
checkpoint with lowest finite validation loss from this seed's completed epochs.
The snapshot includes parameters and buffers; timeout and normal completion use
the same selection. If no epoch has validated, use the current run's latest state
and log `last_no_validation`; never silently load a checkpoint from an older run.

`RESTORE_BEST_CHECKPOINT=False` restores last-state selection. With
`USE_REPRESENTATIVE_METADATA=False`, this supplies the original feature and
checkpoint-selection paths for a controlled baseline. The model, focal loss,
sample cap, seeds, epoch/time caps, neighbor blend, candidate pool, and tier merge
are unchanged. No ranking loss or additional training is bundled into this run.

Source changes: `src/tpu_trainer.py`, `src/config.py`, and
`scripts/make_kaggle_notebook.py`; regenerate the notebooks. The new
`[checkpoint]` lines report selected epoch, step, validation loss, actual completed
epochs, and stop reason. Checkpoint tests set weights manually without training.

The user subsequently reported **the score is still the same**, but the latest
supplied notebook log still runs the older metadata experiment. Do not attribute
a measured score to the checkpoint change without its run identity. There is no
demonstrated gain. Both previous experiment flags remain false for the controlled
candidate-loss comparison; neither is being retried as its hypothesis.

## Tested experiment — mass-matched candidate-ranking supervision (gate rejected)

The following describes the completed GPU comparison. Its measured results are
above. The candidate ensemble failed the promotion gate and has no public score;
do not repeat this exact experiment as a proposed improvement.

One model change: keep focal BCE and add

`0.05 * mean(CE(soft_Tanimoto(sigmoid(logits), [truth, decoys]) / 0.05) / log(1 + n_valid_decoys))`.

The truth occupies class 0. Rows without distinguishable mass-matched decoys
contribute zero ranking loss. Focal BCE still anchors bit probabilities for the
unchanged binary/soft Tanimoto mix and neighbor blend. This loss is a retrieval
surrogate, not MRR and not a calibrated probability of structural identity.

- Same 120k positive structures, four completed epochs, seeds 42/7/123, optimizer,
  learning-rate schedule, peak inputs, architecture, and final ensemble rule.
- Decoys: only the common fitting subset plus the full external structure pool.
  Query masses are training structure masses; `test.parquet` is never consulted.
  For each row, sample at most 256 candidates inside the existing 15 ppm / 0.02 Da
  window, then keep up to 16 highest-Tanimoto and 16 random remaining decoys.
  Exclude the true InChIKey14, duplicate skeletons, holdout keys, and identical
  hashed fingerprints (which this 2048-bit head cannot distinguish).
- Keep packed fingerprints and int32 decoy indices. Decode at most 32 fingerprints
  per item; never allocate a 120k × 32 × 2048 dense decoy tensor.
- Score, lock, weak train competition, neighbor k=20 / alpha=0.5, external
  production pool, analogs, and decoder-off behavior remain unchanged.

Rationale: the existing loss supervises bits independently, whereas the task
requires distinguishing the correct skeleton from plausible same-mass structures.
MIST's official implementation also uses molecular libraries as contrastive
decoys: https://github.com/samgoldman97/mist . This motivates the hypothesis;
we are not importing its architecture, pretrained weights, or reported accuracy.

### Paired GPU ensemble comparison and acceptance rule

1. Select up to 512 deterministic holdout skeletons from the complement of the
   common 120k fitting subset. All ensemble models use that same fitting subset.
   Exclude those keys from the training decoy pool and from a validation copy of
   the entire spectral/neighbor library, including every matching first InChIKey
   block. Assert the intersections are empty.
2. Use stored **observed** representative-spectrum features for queries. Do not
   substitute formula/SMILES exact mass into query metadata. Unknown adducts and
   invalid/empty representatives are ineligible. This uses one representative
   spectrum per molecule, not test-style multi-spectrum aggregation.
3. The distributed external parquet removed every train key, including these
   holdouts. Add holdout structures to a validation-only copy of the full pool.
   Add **no spectral features**. This measures ranking conditional on the answer
   being available; it does not measure actual hidden-test library coverage.
4. Train seeds 42, 7, and 123 twice: control weight 0 and experimental weight
   0.05. Each pair uses identical inputs/splits/initialization. Both keep
   last-state checkpoint selection, batch 64, FP32, and the same epoch/time caps.
   Independent fresh processes own one GPU each; no gradient averaging changes
   the effective batch. All six fits must finish four epochs and the expected
   7,128 optimizer steps. Any incomplete fit stops the run with its logs retained.
5. Average pre-sigmoid logits within each three-member ensemble. Evaluate the
   resulting predictions with the same leakage-free
   neighbor blend and the unchanged full `rank_molecules` pipeline, including
   weak train hits and analog generation. Also report candidate-only MRR and
   truth mass-window recall to distinguish a lookup miss from ranking failure.
6. Promote only if at least 256 query pairs complete, absolute paired MRR gain
   is at least 0.005, and the lower endpoint of a fixed-seed 4096-resample paired
   bootstrap 95% interval is positive. Otherwise retain the control objective.
   Select the entire candidate ensemble on a pass, otherwise the entire control
   ensemble. Never mix training objectives in the final three-member ensemble.
   No hyperparameter sweep is bundled.

The gate is an internal **three-seed, train-domain** diagnostic. It is not an
unbiased final estimate after model selection, does not measure Class 1 domain
transfer or Class 3 recovery, and cannot guarantee leaderboard improvement.
A failed/inconclusive comparison is not a reason to spend a submission.
Nothing yet supports a claim that this reaches 0.350.

### Runtime, artifacts, and verification

- Six fits at a maximum of 90 minutes each, at most two running concurrently.
  Equal per-fit caps are reduced on one GPU to reserve two hours for paired
  evaluation and final inference inside the 8.5-hour session budget. Paired
  ranking is capped at 60 minutes. The supplied two-T4 run wrote its submission
  after 63.6 minutes; paired ranking took 1,137.0 seconds. These are observations
  for that run, not guaranteed timings for other hardware or hidden-test inputs.
- Training runs only on Kaggle. Local checks use synthetic spectra, manual
  logits/gradients, and constant models; no optimizer steps or score measurements.
- Keep `/kaggle/working/submission.csv`, `run_manifest.json`,
  `retrieval_validation.json`, and `casmi_diagnostics.zip`. The zip contains all
  six model checkpoints, both reports, worker logs/configs, and embedded source.
  The manifest records the run ID, source/submission hashes, selected arm,
  settings, GPU hardware, and each fit's completed epochs/steps. Only the current
  run's archived temporary directory is removed; unrelated outputs are preserved.
- Local verification: 32 tests pass, covering decoys/loss, metadata, checkpoint
  selection, held-out evaluation, cache integrity, one-/two-GPU scheduling with
  mock workers, notebook selection, and output packaging. This is not an actual
  CUDA training run or a Kaggle score measurement.
- Changed implementation: `src/model.py`, `src/candidate_training.py`,
  `src/retrieval_validation.py`, `src/gpu_training.py`, `src/data.py`, `src/tpu_trainer.py`,
  `src/config.py`, and `scripts/make_kaggle_notebook.py`.
- Regenerate `kaggle_gpu_submission.ipynb`, `kaggle_submission.ipynb`, and the
  legacy `kaggle_submission_tpu.ipynb` alias using the generator. All now require
  GPU; the TPU filename is retained only for compatibility. Do not edit notebook
  JSON manually. Keep the 0.139 submission selected until a higher score exists.
