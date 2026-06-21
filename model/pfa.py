"""L2 — Performance Factors Analysis / Additive Factors Model (docs/03; Pavlik 2009,
Cen-Koedinger 2006):

    logit P = Σ_k [ β_k + γ_k·#prior_correct_k + ρ_k·#prior_incorrect_k ] + (θ_eff − b)

γ_k (coefficient on a skill's prior-correct count) is that skill's learning rate. The
(θ_eff − b) term is the L1 IRT difficulty signal, so **L2 builds on L1** (docs/03) — given
a fitted IRT model it should be ≥ L1. Features are built in temporal order (no leakage)
and standardized (for convergence). numpy/sklearn only; no torch (docs/05).
"""

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from eval.dataset import Record


def _difficulty_feature(r: Record, irt) -> float:
    if irt is not None:
        return (irt.theta_eff(r.tags) - r.b) / irt.s  # IRT logit gap (docs/03 PFA term)
    return r.b / 1000.0  # fallback when no IRT model is supplied


def _featurize(records, vocab, counts, irt=None) -> tuple[list, list]:
    idx = {t: i for i, t in enumerate(vocab)}
    n = len(vocab)
    rows, ys = [], []
    for r in records:
        row = [0.0] * (3 * n + 1)
        for t in r.tags:
            j = idx.get(t)
            if j is None:
                continue
            row[j] = 1.0
            row[n + j] = float(counts[t]["c"])
            row[2 * n + j] = float(counts[t]["f"])
        row[-1] = _difficulty_feature(r, irt)
        rows.append(row)
        ys.append(r.y)
        for t in r.tags:
            if t in counts:
                counts[t]["c" if r.y else "f"] += 1
    return rows, ys


def build_features(records, irt=None):
    vocab = sorted({t for r in records for t in r.tags})
    counts = {t: {"c": 0, "f": 0} for t in vocab}
    x, y = _featurize(records, vocab, counts, irt)
    names = (
        [f"beta:{t}" for t in vocab]
        + [f"gamma:{t}" for t in vocab]
        + [f"rho:{t}" for t in vocab]
        + ["difficulty"]
    )
    return x, y, names, vocab


class PFAModel:
    def __init__(self, irt=None) -> None:
        self.irt = irt  # fitted L1 SkillModel; supplies the (θ_eff − b) term
        self.pipe = Pipeline(
            [("scale", StandardScaler()), ("lr", LogisticRegression(max_iter=5000, C=1.0))]
        )
        self.vocab: list[str] = []
        self._train_counts: dict[str, dict] = {}

    def fit(self, records: list[Record]) -> "PFAModel":
        self.vocab = sorted({t for r in records for t in r.tags})
        counts = {t: {"c": 0, "f": 0} for t in self.vocab}
        x, y = _featurize(records, self.vocab, counts, self.irt)
        self._train_counts = counts
        self.pipe.fit(x, y)
        return self

    def predict(self, records: list[Record]) -> list[float]:
        counts = {t: dict(v) for t, v in self._train_counts.items()}
        x, _ = _featurize(records, self.vocab, counts, self.irt)
        return self.pipe.predict_proba(x)[:, 1].tolist()

    def learning_rates(self) -> dict[str, float]:
        n = len(self.vocab)
        coef = self.pipe.named_steps["lr"].coef_[0]  # signs survive standardization
        return {t: float(coef[n + i]) for i, t in enumerate(self.vocab)}
