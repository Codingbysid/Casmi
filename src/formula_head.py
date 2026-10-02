"""Lightweight formula-count head (MIST-CF lite).

Predicts C/H/N/O/P/S/Cl/Br counts from a fingerprint + exact mass so Class 2
isobaric formulas can be down-weighted without enumerating SENIOR formulas.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.chem import FORMULA_ELEMENTS, formula_count_vector


class FormulaNet(nn.Module):
    def __init__(self, fp_bits: int = 2048, n_out: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(fp_bits + 1, 128),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Linear(128, n_out),
        )

    def forward(self, fp: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
        if mass.ndim == 1:
            mass = mass.unsqueeze(-1)
        return self.net(torch.cat([fp, mass / 1000.0], dim=-1))


def train_formula_head(
    fps: np.ndarray,
    masses: np.ndarray,
    formulas: list[str] | np.ndarray,
    *,
    epochs: int = 4,
    batch_size: int = 256,
    lr: float = 1e-3,
    max_n: int = 40_000,
    seed: int = 42,
) -> FormulaNet | None:
    y = np.stack([formula_count_vector(f) for f in formulas]).astype(np.float32)
    keep = y.sum(axis=1) > 0
    fps = np.asarray(fps, dtype=np.float32)[keep]
    masses = np.asarray(masses, dtype=np.float32)[keep]
    y = y[keep]
    n = int(fps.shape[0])
    if n < 50:
        print("[formula] not enough labeled formulas")
        return None
    rng = np.random.default_rng(seed)
    if n > max_n:
        pick = rng.choice(n, size=max_n, replace=False)
        fps, masses, y = fps[pick], masses[pick], y[pick]
        n = max_n
    # Spec2FP fingerprints are noisy; jitter bits so the head is not overconfident.
    fps = np.clip(fps + rng.normal(0.0, 0.12, fps.shape).astype(np.float32), 0.0, 1.0)
    device = torch.device("cpu")
    model = FormulaNet(fp_bits=int(fps.shape[1]), n_out=y.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    x_fp = torch.from_numpy(fps)
    x_m = torch.from_numpy(masses)
    y_t = torch.from_numpy(y)
    model.train()
    for epoch in range(int(epochs)):
        perm = rng.permutation(n)
        losses = []
        for start in range(0, n, int(batch_size)):
            sl = perm[start : start + int(batch_size)]
            pred = model(x_fp[sl], x_m[sl])
            loss = torch.mean((pred - y_t[sl]).abs())
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        print(f"[formula] epoch={epoch+1}/{epochs} l1={np.mean(losses):.3f} n={n} elems={len(FORMULA_ELEMENTS)}")
    model.eval()
    return model


@torch.no_grad()
def predict_formula_counts(model: FormulaNet | None, fp: np.ndarray, mass: float) -> np.ndarray | None:
    if model is None:
        return None
    fp_t = torch.from_numpy(np.asarray(fp, dtype=np.float32).reshape(1, -1))
    mass_t = torch.tensor([float(mass)], dtype=torch.float32)
    out = model(fp_t, mass_t).cpu().numpy().ravel()
    return np.clip(out, 0.0, None).astype(np.float32)
