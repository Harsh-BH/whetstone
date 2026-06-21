"""Power law of practice (docs/07 A3; Newell & Rosenbloom 1981): does first-attempt error
fall as a skill gets more opportunities? Binned empirical error per skill, fit
error ≈ a·n^(-c) in log-log space (robust). c ≥ 0 ⇒ error non-increasing ⇒ "fits".
"""

import numpy as np

from config import MIN_OPPORTUNITIES
from eval.dataset import Record


def opportunity_error(
    records: list[Record], tag: str, n_bins: int = 5
) -> list[tuple[float, float]]:
    """Binned (opportunity index, mean first-attempt error) for one skill, in time order."""
    errs = [1 - r.y for r in records if tag in r.tags]
    n = len(errs)
    if n == 0:
        return []
    size = max(1, n // n_bins)
    points = []
    for start in range(0, n, size):
        chunk = errs[start : start + size]
        if chunk:
            mid = start + len(chunk) / 2.0 + 0.5  # 1-based opportunity index of the bin centre
            points.append((mid, sum(chunk) / len(chunk)))
    return points


def fit_power_law(points: list[tuple[float, float]]) -> dict:
    """Fit error ≈ a·n^(-c) via log-log linear regression. ok = error non-increasing (c≥0)."""
    if len(points) < 2:
        return {"a": None, "c": 0.0, "ok": False}
    ns = np.array([p[0] for p in points], dtype=float)
    errs = np.clip(np.array([p[1] for p in points], dtype=float), 1e-3, 1.0)
    slope, intercept = np.polyfit(np.log(ns), np.log(errs), 1)
    c = -float(slope)
    return {"a": float(np.exp(intercept)), "c": c, "ok": c >= 0.0}


def curve_report(records: list[Record], min_opps: int = MIN_OPPORTUNITIES, n_bins: int = 5) -> dict:
    tags = {t for r in records for t in r.tags}
    fits = {}
    for t in tags:
        if sum(1 for r in records if t in r.tags) >= min_opps:
            fits[t] = fit_power_law(opportunity_error(records, t, n_bins))
    frac = (sum(f["ok"] for f in fits.values()) / len(fits)) if fits else 0.0
    return {"skills": fits, "fraction_fit": frac, "n_skills": len(fits)}
