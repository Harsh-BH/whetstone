"""Assess mode (CAT): select unseen problems that most reduce uncertainty about the
high-σ target topics — max Fisher information ⇒ P(solve) ≈ 0.5 (docs/04 Mode A, docs/03).

Scored by expected posterior-variance reduction σ_t²·I(θ_t), where the Fisher info
I = P(1−P)/s² is maximal at P=0.5 (b ≈ μ_t). This focuses on the most uncertain
relevant skill and on items that discriminate best.
"""

import math

from recommender.candidates import Candidate


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def assess_select(model, cands: list[Candidate], topics: set[str], k: int) -> list[dict]:
    targets = set(topics)
    scored: list[tuple[float, dict]] = []
    for c in cands:
        relevant = [t for t in c.tags if t in targets]
        if not relevant:
            continue
        best = 0.0
        for t in relevant:
            sk = model.tags.get(t)
            mu = sk.mu if sk else model.prior_mu
            sigma = sk.sigma if sk else model.prior_sigma
            p_t = _sigmoid((mu - c.b) / model.s)
            info_t = p_t * (1.0 - p_t) / (model.s * model.s)
            best = max(best, sigma * sigma * info_t)  # expected variance reduction
        p_pred, info_pred = model.predict_solve(c.b, c.tags)
        scored.append(
            (
                best,
                {
                    "candidate": c,
                    "predicted_p": p_pred,
                    "predicted_info": info_pred,
                    "mode": "assess",
                    "score": best,
                },
            )
        )
    scored.sort(key=lambda x: -x[0])
    return [d for _, d in scored[:k]]
