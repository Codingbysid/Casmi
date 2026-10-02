#!/usr/bin/env python3
"""Build a frozen SMILES vocabulary from train.parquet for offline TPU runs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pyarrow.parquet as pq

from src.chem_tokenizer import SMILESVocabulary
from src.config import get_config


def main() -> None:
    cfg = get_config()
    train_path = cfg.train_path
    print(f"Loading unique SMILES from {train_path} ...")
    table = pq.read_table(str(train_path), columns=["normalized_smiles"])
    smiles = table.column(0).unique().to_pylist()
    smiles = [s for s in smiles if isinstance(s, str) and s]
    print(f"unique SMILES: {len(smiles)}")

    vocab = SMILESVocabulary()
    print("Extracting tokens (min_freq=5)...")
    vocab.build_vocab(smiles, min_freq=5)

    out_src = ROOT / "src" / "smiles_vocab.json"
    out_art = cfg.artifact_dir / "smiles_vocab.json"
    # Bare token2idx as specified, plus a sibling copy with idx2token for loaders.
    out_src.write_text(json.dumps(vocab.token2idx, indent=4))
    vocab.save(out_art)
    print(f"wrote {out_src} ({len(vocab)} tokens)")
    print(f"wrote {out_art}")


if __name__ == "__main__":
    main()
