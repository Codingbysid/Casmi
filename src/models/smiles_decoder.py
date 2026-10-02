"""Prefix-conditioned causal Transformer for de novo SMILES generation.

Conditioning (predicted Morgan fingerprint + precursor mass + adduct) is
projected into a handful of virtual prefix tokens and prepended to the SMILES
embeddings, so a standard causal self-attention stack is sufficient — no
cross-attention stream, and all sequence lengths stay static for XLA.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.chem import canonical_smiles, exact_mass_from_smiles, inchikey14_from_smiles, morgan_fingerprint, smiles_to_mol
from src.config import NUM_ADDUCTS, Config
from src.retrieval import tanimoto
from src.smiles_tokenizer import SmilesTokenizer


class SinusoidalMassEmbedding(nn.Module):
    """Encodes continuous precursor neutral mass into a dense vector."""

    def __init__(self, dim: int = 64) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, mass: torch.Tensor) -> torch.Tensor:
        # mass shape: (B, 1) or (B,)
        if mass.dim() == 1:
            mass = mass.unsqueeze(-1)
        device = mass.device
        half_dim = self.dim // 2
        emb = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=mass.dtype) * -emb)
        ang = mass * emb.unsqueeze(0)
        return torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)


class LightweightSmilesDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        fp_dim: int = 2048,
        num_adducts: int = 16,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        prefix_len: int = 8,
        max_seq_len: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.prefix_len = prefix_len
        self.max_seq_len = max_seq_len
        self.vocab_size = vocab_size

        self.mass_encoder = SinusoidalMassEmbedding(dim=64)
        self.adduct_embed = nn.Embedding(num_adducts, 32)

        cond_input_dim = fp_dim + 64 + 32
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_input_dim, d_model * 2),
            nn.GELU(),
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, prefix_len * d_model),
        )

        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Embedding(max_seq_len + prefix_len, d_model)

        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            decoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.head.weight = self.token_embed.weight

        total = prefix_len + max_seq_len
        causal = torch.triu(torch.ones(total, total, dtype=torch.bool), diagonal=1)
        causal[:prefix_len, :prefix_len] = False
        self.register_buffer("causal_mask", causal, persistent=False)

    def _causal_mask(self, total_len: int, device: torch.device) -> torch.Tensor:
        if total_len == self.causal_mask.size(0):
            return self.causal_mask.to(device)
        mask = torch.triu(torch.ones(total_len, total_len, dtype=torch.bool, device=device), diagonal=1)
        mask[: self.prefix_len, : self.prefix_len] = False
        return mask

    def forward(
        self,
        fp: torch.Tensor,
        mass: torch.Tensor,
        adduct_idx: torch.Tensor,
        tgt_tokens: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        fp: (B, 2048) predicted probabilities or ground-truth bits
        mass: (B, 1) neutral precursor mass
        adduct_idx: (B,) integer adduct class
        tgt_tokens: (B, L) SMILES token ids (BOS-prefixed, PAD-padded)
        tgt_mask: (B, L) True = padding (ignored by attention)
        """
        bsz, length = tgt_tokens.shape
        device = fp.device

        mass_enc = self.mass_encoder(mass)
        adduct_enc = self.adduct_embed(adduct_idx.long())
        cond_vec = torch.cat([fp.float(), mass_enc, adduct_enc], dim=-1)
        prefix = self.cond_mlp(cond_vec).view(bsz, self.prefix_len, self.d_model)

        token_emb = self.token_embed(tgt_tokens)
        seq = torch.cat([prefix, token_emb], dim=1)
        pos = torch.arange(0, self.prefix_len + length, device=device).unsqueeze(0)
        seq = seq + self.pos_embed(pos)

        total_len = self.prefix_len + length
        causal_mask = self._causal_mask(total_len, device)
        key_padding = None
        if tgt_mask is not None:
            prefix_pad = torch.zeros(bsz, self.prefix_len, dtype=torch.bool, device=device)
            key_padding = torch.cat([prefix_pad, tgt_mask.bool()], dim=1)

        out = self.transformer(seq, mask=causal_mask, src_key_padding_mask=key_padding)
        out = self.norm(out)
        return self.head(out[:, self.prefix_len :, :])


def decoder_from_config(tokenizer: SmilesTokenizer, cfg: Config) -> LightweightSmilesDecoder:
    return LightweightSmilesDecoder(
        vocab_size=tokenizer.vocab_size,
        fp_dim=cfg.fp_bits,
        num_adducts=NUM_ADDUCTS,
        d_model=cfg.smiles_d_model,
        nhead=cfg.smiles_nhead,
        num_layers=cfg.smiles_num_layers,
        dim_feedforward=cfg.smiles_dim_feedforward,
        prefix_len=cfg.smiles_prefix_len,
        max_seq_len=cfg.smiles_max_len,
        dropout=cfg.smiles_dropout,
    )


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def _top_p_logits(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
    probs = F.softmax(sorted_logits, dim=-1)
    cum = torch.cumsum(probs, dim=-1)
    mask = cum > top_p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
    return torch.zeros_like(logits).scatter_(-1, sorted_idx, sorted_logits)


@torch.no_grad()
def sample_autoregressive(
    model: LightweightSmilesDecoder,
    fp: torch.Tensor,
    mass: torch.Tensor,
    adduct_idx: torch.Tensor,
    tokenizer: SmilesTokenizer,
    *,
    max_len: int = 100,
    temp: float = 0.75,
    top_p: float = 0.90,
) -> list[list[int]]:
    """CPU-friendly ancestral sampling. Sequence length grows; do not run on XLA."""
    model.eval()
    device = fp.device
    bsz = int(fp.shape[0])
    max_len = min(int(max_len), model.max_seq_len)
    tokens = torch.full((bsz, 1), tokenizer.bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(bsz, dtype=torch.bool, device=device)
    temp = max(float(temp), 1e-5)

    for _ in range(max_len - 1):
        pad_mask = tokens == tokenizer.pad_id
        logits = model(fp, mass, adduct_idx, tokens, tgt_mask=pad_mask)
        step = logits[:, -1, :] / temp
        step[:, tokenizer.pad_id] = float("-inf")
        step[:, tokenizer.bos_id] = float("-inf")
        step = _top_p_logits(step, top_p)
        probs = F.softmax(step, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1).squeeze(-1)
        nxt = torch.where(finished, torch.full_like(nxt, tokenizer.pad_id), nxt)
        tokens = torch.cat([tokens, nxt.unsqueeze(-1)], dim=1)
        finished = finished | (nxt == tokenizer.eos_id)
        if bool(finished.all()):
            break
    return [row.tolist() for row in tokens.cpu()]


def compute_tanimoto_against_pred(gen_fp: np.ndarray, pred_fp_bits: np.ndarray) -> float:
    return float(tanimoto(pred_fp_bits, gen_fp.reshape(1, -1))[0])


def generate_candidates_for_molecule(
    model: LightweightSmilesDecoder,
    pred_fp: torch.Tensor,
    neutral_mass: float,
    adduct_idx: int,
    tokenizer: SmilesTokenizer,
    num_samples: int = 100,
    mass_tolerance_ppm: float = 20.0,
    target_k: int = 25,
    temp: float = 0.75,
    top_p: float = 0.90,
    max_len: int = 100,
    keep_mass_misses: bool = False,
) -> list[str]:
    """Generate, mass-filter, InChIKey14-dedup, and rank de novo SMILES (Class 3)."""
    model.eval()
    device = next(model.parameters()).device
    if pred_fp.dim() == 1:
        pred_fp = pred_fp.unsqueeze(0)
    pred_fp = pred_fp.float().to(device)
    n = int(num_samples)
    b_fp = pred_fp.repeat(n, 1)
    b_mass = torch.full((n, 1), float(neutral_mass), dtype=torch.float32, device=device)
    b_adduct = torch.full((n,), int(adduct_idx), dtype=torch.long, device=device)

    with torch.no_grad():
        generated_tokens = sample_autoregressive(
            model,
            b_fp,
            b_mass,
            b_adduct,
            tokenizer,
            max_len=max_len,
            temp=temp,
            top_p=top_p,
        )

    pred_np = pred_fp[0].detach().cpu().numpy()
    pred_fp_bits = (pred_np >= 0.5).astype(np.float32)
    if pred_fp_bits.sum() == 0:
        k = min(64, pred_np.size)
        top = np.argpartition(pred_np, -k)[-k:]
        pred_fp_bits = np.zeros_like(pred_np, dtype=np.float32)
        pred_fp_bits[top] = 1.0

    valid: dict[str, dict[str, float | str]] = {}
    misses: dict[str, dict[str, float | str]] = {}
    query_mass = float(neutral_mass)
    for token_seq in generated_tokens:
        smiles_str = tokenizer.decode(token_seq)
        mol = smiles_to_mol(smiles_str)
        if mol is None:
            continue
        calc_mass = exact_mass_from_smiles(smiles_str)
        if calc_mass <= 0:
            continue
        ppm_error = (
            abs(calc_mass - query_mass) / query_mass * 1e6 if query_mass > 0 else 1e9
        )
        key = inchikey14_from_smiles(smiles_str)
        if not key:
            continue
        can = canonical_smiles(smiles_str)
        gen_fp = morgan_fingerprint(can, n_bits=pred_fp_bits.size)
        score = compute_tanimoto_against_pred(gen_fp, pred_fp_bits) - (ppm_error * 0.001)
        bucket = valid if ppm_error <= mass_tolerance_ppm else misses
        prev = bucket.get(key)
        if prev is None or score > float(prev["score"]):
            bucket[key] = {"smiles": can, "score": float(score)}

    ranked = sorted(valid.values(), key=lambda x: float(x["score"]), reverse=True)
    out = [str(item["smiles"]) for item in ranked[: int(target_k)]]
    if keep_mass_misses and len(out) < int(target_k):
        extra = sorted(misses.values(), key=lambda x: float(x["score"]), reverse=True)
        have = set(out)
        for item in extra:
            smi = str(item["smiles"])
            if smi in have:
                continue
            out.append(smi)
            have.add(smi)
            if len(out) >= int(target_k):
                break
    return out
