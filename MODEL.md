# Enveda CASMI 2026 — model brief

This file is the source of truth for the current pipeline. Read it before proposing changes.
The Kaggle notebook is generated from `src/` by `scripts/make_kaggle_notebook.py`. Edit `src/` and the generator, then regenerate `kaggle_submission.ipynb`. Do not hand-edit the notebook.

## Task

Kaggle: **Enveda CASMI 2026 — Molecule ID from Mass Spectra**.

For each unknown molecule, rank up to **25 SMILES**. The metric is **MRR@25** on the first 14 characters of the RDKit InChIKey (InChIKey14, the connectivity skeleton). Stereochemistry and the second InChIKey block do not count. Exactly one row per `molecule_id`. Guesses are semicolon-separated SMILES. Empty rows are invalid.

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
| **V5 and V6** | **Best.** Lock cosine ≥ 0.75, then weak Class 1 competes with Class 2. Neighbor blend on. Spec2FP 120k structures, 4 epochs, 3 seeds, ~1.5 h/seed. | **0.139** |
| V7 | Weak Class 1 dropped; remaining slots Class 2 only | 0.119 |
| V8 | Same as 0.139 but neighbor blend off | 0.133 |
| V9 | Spec2FP on all ~276k skeletons, 6 epochs | 0.126 |

**0.139 is the recipe to beat.** Do not retune the lock, drop weak Class 1, turn neighbor blend off, or train Spec2FP longer/harder. Those four experiments all lost.

0.139 is roughly 56 rank-1 hits out of 400, which is about the whole Class 1 slice. Class 2 and Class 3 are still near zero on the hidden test. The bottleneck is **candidate coverage and fingerprint ranking inside COCONUT**, not the Class 1 lock.

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

A CLS token over the peak transformer, the binned vector, and the precursor/adduct vector are concatenated and mapped to **2048 logits** (Morgan/ECFP4, radius 2). Loss is multi-label focal BCE (`gamma=1.5`, `pos_weight=6`).

**Kaggle notebook overrides the config defaults** to a smaller CPU net:

- `d_model=128`, `n_heads=4`, `n_transformer_layers=2`, `n_conv_channels=64`
- `batch_size=64` (128 if CUDA)
- `MAX_TRAIN_SPECTRA=120_000` unique library rows (random subset of ~276k)
- `NUM_EPOCHS=4`
- 3 seeds, each capped at **1.5 h**, total train budget 4.5 h, session budget 8.5 h
- GPU may widen the net to `d_model=256`, 8 heads, 3 layers

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
TRAIN_TIME_LIMIT_S = 4.5h
USE_NEIGHBOR_FP = True
NEIGHBOR_BLEND = 0.5
PER_SEED_MAX_S = 1.5h
```

Kaggle constraints: CPU or GPU, **internet off**, kernel source **< 1 MB** (the notebook embeds `src/` as JSON; Class 2 is an attached dataset, never embedded), Python 3.12, Run All must finish inside **9 h** and write `submission.csv`.

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
| `src/infer.py` | **`rank_molecules` / `_merge_tiers` — the 0.139 merge** |
| `src/analogs.py` | Mass-shift parent retrieval + SMARTS |
| `src/calibration.py` | Optional OR-merge probabilities. Ranking score used in the compete pool is the linear mix above, not only this table. |
| `src/models/smiles_decoder.py` | Class 3 decoder. **Disabled.** |
| `src/formula_head.py`, `src/rerank.py` | Built earlier, **disabled** in `predict_test` |
| `scripts/make_kaggle_notebook.py` | Emits `kaggle_submission.ipynb` and `kaggle_gpu_submission.ipynb` |
| `scripts/build_class2_np.py` | Rebuilds the COCONUT ∪ LOTUS parquet. Needs `train.parquet` locally, which is not on disk. |
| `kaggle_dataset/class2_candidates.parquet` | The 452,608-row library (also on Kaggle) |
| `kaggle_submission.ipynb` | What gets uploaded to Kaggle |

## Current experiment (not yet scored)

Three ranking changes against the 0.139 recipe. Training stays focal BCE, 3 seeds, 120k structures, 4 epochs, neighbor blend on, decoder off. The candidate-ranking loss from the GPU v2 run failed its holdout gate (the public 0.137 was the focal-only control) and is not in this notebook. `USE_REPRESENTATIVE_METADATA` and `RESTORE_BEST_CHECKPOINT` stay false.

1. Modified cosine is 0 unless at least 4 peaks match and those peaks cover at least 15% of query intensity. Supported matches are scaled by `sqrt(n_hits / 8)`.
2. A Class 1 lock requires cosine ≥ 0.75 and at least 5 matched peaks. The previous run locked 339/400 molecules. The log line is `[rank] molecules with locked Class 1 (cosine>=0.75, n_match>=5)`.
3. Class 2 and other hits with no verified spectrum score `0.92 * (0.78 * tanimoto + 0.22 * mass)` instead of treating a missing spectrum as a 0.42 penalty. Verified spectral hits still use `0.42 / 0.46 / 0.12`.
4. InChIKey14 dedup runs `TautomerEnumerator.Canonicalize` before `MolToInchiKey`, matching the competition metric.

The final log also prints `class2_slots`. That count should rise above the 206 slots in the false-lock run if the gates fire. Keep 0.139 selected until a public score is strictly higher. This does not by itself establish 0.25 or 0.350.

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
