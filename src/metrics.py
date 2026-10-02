"""Competition metric: Mean Reciprocal Rank @ K on InChIKey14."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from src.chem import inchikey14_from_smiles


def reciprocal_rank(ranked_keys: Sequence[str], true_key: str, k: int = 25) -> float:
    if not true_key:
        return 0.0
    for i, key in enumerate(ranked_keys[:k]):
        if key == true_key:
            return 1.0 / float(i + 1)
    return 0.0


def smiles_to_keys(smiles_list: Sequence[str]) -> list[str]:
    return [inchikey14_from_smiles(s) for s in smiles_list]


def mrr_at_k(
    predictions: Sequence[Sequence[str]],
    true_smiles: Sequence[str] | None = None,
    true_keys: Sequence[str] | None = None,
    k: int = 25,
) -> float:
    """``predictions[i]`` is a ranked SMILES list for query i."""
    if true_keys is None:
        if true_smiles is None:
            raise ValueError("Provide true_smiles or true_keys")
        true_keys = [inchikey14_from_smiles(s) for s in true_smiles]
    scores = []
    for pred, gold in zip(predictions, true_keys):
        keys = smiles_to_keys(pred)
        scores.append(reciprocal_rank(keys, gold, k=k))
    if not scores:
        return 0.0
    return float(np.mean(scores))
