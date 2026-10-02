"""Compatibility wrapper: decoder/train code uses ``SmilesTokenizer``.

The chemically correct regex lives in ``src/chem_tokenizer.py``. This module
exposes the padded encode/decode API the Spec2FP decoder already calls, and
loads the frozen offline vocab from ``src/smiles_vocab.json`` when present.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

from src.chem_tokenizer import (
    PACKAGED_VOCAB_PATH,
    SMILESTokenizer,
    SMILESVocabulary,
    load_packaged_vocabulary,
)

PAD, BOS, EOS, UNK = "<PAD>", "<BOS>", "<EOS>", "<UNK>"
_RAW = SMILESTokenizer()


def tokenize_smiles(smiles: str) -> list[str]:
    return _RAW.tokenize(smiles)


class SmilesTokenizer:
    """Thin adapter around ``SMILESVocabulary`` with TPU-padded ``encode``."""

    def __init__(self, vocab: SMILESVocabulary | None = None) -> None:
        self.vocab = vocab or load_packaged_vocabulary()

    @property
    def pad_id(self) -> int:
        return self.vocab.pad_idx

    @property
    def bos_id(self) -> int:
        return self.vocab.bos_idx

    @property
    def eos_id(self) -> int:
        return self.vocab.eos_idx

    @property
    def unk_id(self) -> int:
        return self.vocab.unk_idx

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    @property
    def token_to_id(self) -> dict[str, int]:
        return self.vocab.token2idx

    @property
    def id_to_token(self) -> list[str]:
        return [self.vocab.idx2token[i] for i in range(len(self.vocab))]

    def encode(self, smiles: str, max_len: int) -> list[int]:
        return self.vocab.encode_padded(smiles, max_len)

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        return self.vocab.decode(ids, strip_special=skip_special)

    def save(self, path: Path | str) -> None:
        self.vocab.save(path)

    @classmethod
    def load(cls, path: Path | str) -> "SmilesTokenizer":
        return cls(SMILESVocabulary.load(path))

    @classmethod
    def from_tokens(cls, tokens: Sequence[str]) -> "SmilesTokenizer":
        vocab = SMILESVocabulary()
        for tok in tokens:
            if tok:
                vocab.add_token(str(tok))
        return cls(vocab)

    @classmethod
    def default(cls) -> "SmilesTokenizer":
        return cls(load_packaged_vocabulary())


def build_tokenizer(
    smiles_list: Iterable[str],
    *,
    min_count: int = 5,
    max_size: int | None = None,
) -> SmilesTokenizer:
    """Fit on ``smiles_list``. Prefer the packaged vocab when it already exists."""
    if PACKAGED_VOCAB_PATH.exists() and min_count >= 5:
        return SmilesTokenizer.default()
    vocab = SMILESVocabulary()
    vocab.build_vocab(smiles_list, min_freq=min_count)
    if max_size is not None and len(vocab) > max_size:
        # Keep specials + the first (max_size) tokens in insertion order.
        keep = [vocab.idx2token[i] for i in range(min(max_size, len(vocab)))]
        trimmed = SMILESVocabulary()
        for tok in keep:
            if tok not in trimmed.token2idx:
                trimmed.add_token(tok)
        vocab = trimmed
    return SmilesTokenizer(vocab)
