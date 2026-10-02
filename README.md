# CASMI 2026

Spectrum-to-structure ranking for the Enveda CASMI 2026 Kaggle competition.
The model, the 0.139 recipe, and what not to change are in [MODEL.md](MODEL.md). Start there.

Training and submission run on Kaggle (internet off, Python 3.12, CPU or GPU).
`kaggle_submission.ipynb` is generated from `src/`:

```bash
python scripts/make_kaggle_notebook.py
```

Kaggle inputs: the competition, `class2_candidates.parquet` (this repo’s `kaggle_dataset/`), and a ~35 MB RDKit cp312 manylinux wheel (`kaggle_rdkit_wheels/`).
