"""Regex SMILES tokenizer for natural-product chemical language models.

Bracketed atoms (``[C@@H]``, ``[nH]``, ``[O-]``), multi-letter halogens, and
``%10``-style macrocycle ring closures are kept as *single* tokens so the
decoder does not waste length on shattered brackets.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import re

PACKAGED_VOCAB_PATH = Path(__file__).resolve().parent / "smiles_vocab.json"

# Always keep these as single IDs even if train SMILES were de-stereo'd.
# Natural-product generation still needs them as atomic units.
RESERVED_CHEM_TOKENS: tuple[str, ...] = (
    "/",
    "\\",
    "+",
    ".",
    "@",
    "[C@@H]",
    "[C@H]",
    "[C@@]",
    "[C@]",
    "[NH3+]",
    "[NH2]",
    "[nH+]",
    "[O+]",
    "[NH+]",
    "%13",
    "%14",
    "%15",
)


class SMILESTokenizer:
    """Regex tokenizer: longer / more specific matches precede single characters."""

    def __init__(self) -> None:
        # ORDER MATTERS: longer/specific matches must precede single-character matches.
        pattern = (
            r"("
            r"\[[^\]]+\]|"  # 1. Bracketed atoms ([C@@H], [nH], [O-], [13C])
            r"Br|Cl|"  # 2. Multi-letter halogens (before B / C)
            r"c|n|o|s|p|b|"  # 3. Aromatic lower-case atoms
            r"C|N|O|S|P|F|I|B|"  # 4. Aliphatic upper-case atoms
            r"\%[0-9]{2}|"  # 5. Double-digit ring closures (%10, %99)
            r"\(|\)|\.|=|#|"  # 6. Branching, disconnection, bonds
            r"-|\+|\\\\|\/|"  # 7. Charges and cis/trans stereochem
            r":|~|@|"  # 8. Aromatic / arbitrary bonds, chiral @
            r"[0-9]"  # 9. Single-digit ring closures
            r")"
        )
        self.pattern = re.compile(pattern)

    def tokenize(self, smiles: str) -> List[str]:
        """Split a SMILES string into chemically meaningful tokens."""
        if not smiles:
            return []
        tokens: List[str] = []
        for chunk in self.pattern.split(str(smiles)):
            if not chunk:
                continue
            if self.pattern.fullmatch(chunk):
                tokens.append(chunk)
            else:
                # Unknown leftover (e.g. 'Z'): fall back to characters.
                tokens.extend(list(chunk))
        return tokens


class SMILESVocabulary:
    """Token ↔ ID map with PAD/BOS/EOS/UNK. Vocab size is frozen for XLA."""

    def __init__(self) -> None:
        self.pad_token = "<PAD>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        self.unk_token = "<UNK>"

        self.special_tokens = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]

        self.idx2token: Dict[int, str] = {i: tok for i, tok in enumerate(self.special_tokens)}
        self.token2idx: Dict[str, int] = {tok: i for i, tok in enumerate(self.special_tokens)}

        self.pad_idx = self.token2idx[self.pad_token]
        self.bos_idx = self.token2idx[self.bos_token]
        self.eos_idx = self.token2idx[self.eos_token]
        self.unk_idx = self.token2idx[self.unk_token]

        self.tokenizer = SMILESTokenizer()

    def __len__(self) -> int:
        return len(self.idx2token)

    def add_token(self, token: str) -> None:
        if token not in self.token2idx:
            idx = len(self.idx2token)
            self.token2idx[token] = idx
            self.idx2token[idx] = token

    def build_vocab(self, smiles_iterable: Iterable[str], min_freq: int = 1) -> None:
        """Add tokens seen at least ``min_freq`` times (drops isotopic junk)."""
        freqs: Counter[str] = Counter()
        for smiles in smiles_iterable:
            if not smiles or str(smiles).lower() in {"nan", "none"}:
                continue
            freqs.update(self.tokenizer.tokenize(str(smiles)))
        # Stable order: specials already present; then by descending freq, then token.
        ranked = sorted(freqs.items(), key=lambda kv: (-kv[1], kv[0]))
        for token, freq in ranked:
            if freq >= min_freq:
                self.add_token(token)
        for token in RESERVED_CHEM_TOKENS:
            self.add_token(token)
        print(f"Vocabulary built: {len(self)} total tokens.")

    def encode(self, smiles: str, add_bos: bool = True, add_eos: bool = True) -> List[int]:
        tokens = self.tokenizer.tokenize(smiles)
        token_ids = [self.token2idx.get(tok, self.unk_idx) for tok in tokens]
        if add_bos:
            token_ids = [self.bos_idx] + token_ids
        if add_eos:
            token_ids = token_ids + [self.eos_idx]
        return token_ids

    def encode_padded(self, smiles: str, max_len: int) -> List[int]:
        """BOS + tokens + EOS, truncated/padded to ``max_len`` (XLA-static)."""
        max_len = int(max_len)
        ids = self.encode(smiles, add_bos=True, add_eos=True)
        if len(ids) > max_len:
            # Keep BOS; drop the tail before EOS; force EOS in the last slot.
            ids = ids[: max_len - 1] + [self.eos_idx]
        if len(ids) < max_len:
            ids = ids + [self.pad_idx] * (max_len - len(ids))
        return ids[:max_len]

    def decode(self, token_ids: Sequence[int], strip_special: bool = True) -> str:
        tokens: List[str] = []
        for idx in token_ids:
            tok = self.idx2token.get(int(idx), self.unk_token)
            if tok == self.eos_token:
                break
            if strip_special and tok in self.special_tokens:
                continue
            if tok == self.unk_token and strip_special:
                continue
            tokens.append(tok)
        return "".join(tokens)

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "token2idx": self.token2idx,
            "idx2token": [self.idx2token[i] for i in range(len(self.idx2token))],
        }
        path.write_text(json.dumps(payload, indent=4))

    @classmethod
    def load(cls, path: Path | str) -> "SMILESVocabulary":
        raw = json.loads(Path(path).read_text())
        vocab = cls()
        if isinstance(raw, dict) and "idx2token" in raw:
            tokens = list(raw["idx2token"])
        elif isinstance(raw, dict) and "token2idx" in raw:
            mapping = raw["token2idx"]
            size = max(int(v) for v in mapping.values()) + 1
            tokens = [""] * size
            for tok, idx in mapping.items():
                tokens[int(idx)] = tok
        elif isinstance(raw, dict) and all(isinstance(v, int) for v in raw.values()):
            # Bare token2idx map (scripts/build_vocab.py export).
            size = max(int(v) for v in raw.values()) + 1
            tokens = [""] * size
            for tok, idx in raw.items():
                tokens[int(idx)] = tok
        elif isinstance(raw, list):
            tokens = list(raw)
        else:
            raise ValueError(f"Unrecognized vocab JSON in {path}")
        for tok in tokens:
            if tok and tok not in vocab.token2idx:
                vocab.add_token(tok)
        return vocab

    @classmethod
    def from_token2idx(cls, token2idx: Dict[str, int]) -> "SMILESVocabulary":
        vocab = cls()
        size = max(token2idx.values()) + 1 if token2idx else len(vocab)
        ordered = [""] * size
        for tok, idx in token2idx.items():
            ordered[int(idx)] = tok
        for tok in ordered:
            if tok and tok not in vocab.token2idx:
                vocab.add_token(tok)
        return vocab


def load_packaged_vocabulary() -> SMILESVocabulary:
    """Load the frozen offline vocab shipped in ``src/smiles_vocab.json``."""
    if PACKAGED_VOCAB_PATH.exists():
        return SMILESVocabulary.load(PACKAGED_VOCAB_PATH)
    return SMILESVocabulary()
