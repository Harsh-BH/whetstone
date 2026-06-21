"""Knowledge-model eval metrics (docs/07 A1/A2). Pure functions; no DB."""

import random

import numpy as np
from sklearn.metrics import log_loss, roc_auc_score


def auc(y: list[int], p: list[float]) -> float:
    if len(set(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def log_loss_(y: list[int], p: list[float]) -> float:
    return float(log_loss(y, np.clip(p, 1e-12, 1 - 1e-12), labels=[0, 1]))


def accuracy(y: list[int], p: list[float], thresh: float = 0.5) -> float:
    yp = [1 if pi >= thresh else 0 for pi in p]
    return sum(int(a == b) for a, b in zip(yp, y)) / len(y)


def brier(y: list[int], p: list[float]) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def _bin_edges(n_bins: int) -> np.ndarray:
    return np.linspace(0.0, 1.0, n_bins + 1)


def _bin_mask(p_arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    # left-closed only for the first bin so p==0 lands somewhere; else (lo, hi].
    if lo == 0.0:
        return (p_arr >= lo) & (p_arr <= hi)
    return (p_arr > lo) & (p_arr <= hi)


def ece(y: list[int], p: list[float], n_bins: int = 10) -> float:
    y_arr, p_arr = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    edges = _bin_edges(n_bins)
    total = len(p_arr)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = _bin_mask(p_arr, lo, hi)
        if not mask.any():
            continue
        conf = p_arr[mask].mean()
        acc = y_arr[mask].mean()
        e += (mask.sum() / total) * abs(acc - conf)
    return float(e)


def ece_noise_floor(
    p: list[float], n_bins: int = 10, n_sims: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """ECE that a PERFECTLY-calibrated model would show at this sample size and these
    predicted probabilities. ECE is positively biased on small samples, so on ~200-400
    test points even a perfect model scores well above 0. Returns (mean, 95th pct) of
    the null distribution: y_i ~ Bernoulli(p_i). Deterministic (seeded)."""
    rng = random.Random(seed)
    vals = []
    for _ in range(n_sims):
        y_sim = [1 if rng.random() < pi else 0 for pi in p]
        vals.append(ece(y_sim, p, n_bins))
    vals.sort()
    return sum(vals) / len(vals), vals[int(0.95 * len(vals))]


def reliability(y: list[int], p: list[float], n_bins: int = 10) -> list[dict]:
    y_arr, p_arr = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    edges = _bin_edges(n_bins)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = _bin_mask(p_arr, lo, hi)
        n = int(mask.sum())
        out.append(
            {
                "lo": float(lo),
                "hi": float(hi),
                "n": n,
                "conf": float(p_arr[mask].mean()) if n else None,
                "acc": float(y_arr[mask].mean()) if n else None,
            }
        )
    return out
