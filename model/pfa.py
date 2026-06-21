"""L2 — Performance Factors Analysis / Additive Factors Model (docs/03; Pavlik 2009,
Cen-Koedinger 2006). Adds per-skill learning rates on top of difficulty:

    logit P = Σ_k [ β_k + γ_k·#prior_correct_k + ρ_k·#prior_incorrect_k ] + δ·b

γ_k (coefficient on a skill's prior-correct count) is that skill's learning rate — how
fast practice improves it. Features are built in temporal order so counts reflect only
prior opportunities (no leakage). Backed by sklearn LogisticRegression (numpy/sklearn,
no torch — docs/05).
"""

from sklearn.linear_model import LogisticRegression

from eval.dataset import Record


def _featurize(
    records: list[Record], vocab: list[str], counts: dict[str, dict]
) -> tuple[list, list]:
    """Mutates `counts` as it walks records in order (prequential). Row layout:
    [β_k for k in vocab] + [γ-feat=prior_correct_k] + [ρ-feat=prior_incorrect_k] + [b/1000]."""
    idx = {t: i for i, t in enumerate(vocab)}
    n = len(vocab)
    rows, ys = [], []
    for r in records:
        row = [0.0] * (3 * n + 1)
        for t in r.tags:
            j = idx.get(t)
            if j is None:
                continue
            row[j] = 1.0  # β: skill present
            row[n + j] = float(counts[t]["c"])  # prior correct
            row[2 * n + j] = float(counts[t]["f"])  # prior incorrect
        row[-1] = r.b / 1000.0  # scaled item difficulty
        rows.append(row)
        ys.append(r.y)
        for t in r.tags:  # update opportunity counts AFTER scoring this item
            if t in counts:
                counts[t]["c" if r.y else "f"] += 1
    return rows, ys


def build_features(records: list[Record]):
    vocab = sorted({t for r in records for t in r.tags})
    counts = {t: {"c": 0, "f": 0} for t in vocab}
    x, y = _featurize(records, vocab, counts)
    names = (
        [f"beta:{t}" for t in vocab]
        + [f"gamma:{t}" for t in vocab]
        + [f"rho:{t}" for t in vocab]
        + ["b"]
    )
    return x, y, names, vocab


class PFAModel:
    def __init__(self) -> None:
        self.lr = LogisticRegression(max_iter=2000, C=1.0)
        self.vocab: list[str] = []
        self._train_counts: dict[str, dict] = {}

    def fit(self, records: list[Record]) -> "PFAModel":
        self.vocab = sorted({t for r in records for t in r.tags})
        counts = {t: {"c": 0, "f": 0} for t in self.vocab}
        x, y = _featurize(records, self.vocab, counts)
        self._train_counts = counts  # final state -> prequential test continues from here
        self.lr.fit(x, y)
        return self

    def predict(self, records: list[Record]) -> list[float]:
        counts = {t: dict(v) for t, v in self._train_counts.items()}
        x, _ = _featurize(records, self.vocab, counts)
        return self.lr.predict_proba(x)[:, 1].tolist()

    def learning_rates(self) -> dict[str, float]:
        n = len(self.vocab)
        coef = self.lr.coef_[0]
        return {t: float(coef[n + i]) for i, t in enumerate(self.vocab)}
