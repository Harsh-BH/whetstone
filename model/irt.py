"""L1 Bayesian per-topic IRT (Rasch/1PL) skill model (docs/03).

State: per-tag Gaussian N(mu, sigma^2) on the CF rating scale.
Outcome: P(first-attempt solve) = sigmoid((theta_eff - b) / s), theta_eff = min over tags.
Update: online Laplace — mu += sigma^2 * grad; 1/sigma^2 += Fisher info. sigma is
first-class (it drives Assess vs Train in M2).

Credit assignment (the docs/03 "messiest choice"): theta_eff aggregates a problem's
tag skills via `agg` ("min" = weakest required skill gates; "mean"). One observation's
evidence flows to each tag in proportion to d theta_eff / d mu (argmin for "min",
1/n for "mean") — conserved, never duplicated. `agg` is selected per user on train
log-loss; the choice is the docs/07 aggregation ablation.
"""

import math
from dataclasses import dataclass, field

from config import IRT_S, PRIOR_MU, PRIOR_SIGMA


@dataclass
class TagSkill:
    mu: float
    sigma: float


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


@dataclass
class SkillModel:
    s: float = IRT_S
    prior_mu: float = PRIOR_MU
    prior_sigma: float = PRIOR_SIGMA
    agg: str = "min"  # tag aggregation: "min" (weakest gates) or "mean" (docs/07 ablation)
    tags: dict[str, TagSkill] = field(default_factory=dict)

    def _skill(self, tag: str) -> TagSkill:
        sk = self.tags.get(tag)
        if sk is None:
            sk = TagSkill(mu=self.prior_mu, sigma=self.prior_sigma)
            self.tags[tag] = sk
        return sk

    def _aggregate(self, tags: list[str]) -> tuple[float, list[float]]:
        """Return (theta_eff, weights) where weights[i] = d theta_eff / d mu_i.
        Evidence on an update flows to each tag in proportion to its weight."""
        mus = [self._skill(t).mu for t in tags]
        if self.agg == "mean":
            n = len(mus)
            return sum(mus) / n, [1.0 / n] * n
        # default "min": weakest required skill gates; gradient flows to the argmin.
        j = min(range(len(mus)), key=lambda i: mus[i])
        return mus[j], [1.0 if i == j else 0.0 for i in range(len(mus))]

    def theta_eff(self, tags: list[str]) -> float:
        if not tags:
            return self.prior_mu
        return self._aggregate(tags)[0]

    def predict_solve(self, b: float, tags: list[str]) -> tuple[float, float]:
        theta = self.prior_mu if not tags else self._aggregate(tags)[0]
        p = _sigmoid((theta - b) / self.s)
        info = p * (1.0 - p) / (self.s * self.s)
        return p, info

    def update(self, b: float, tags: list[str], y: int) -> None:
        if not tags:
            return
        theta, weights = self._aggregate(tags)
        p = _sigmoid((theta - b) / self.s)
        # Evidence flows to each tag by its aggregation weight (argmin for "min",
        # 1/n for "mean") — conserves one observation's evidence, never duplicates.
        grad = (y - p) / self.s  # d/dtheta of the log-likelihood
        info = p * (1.0 - p) / (self.s * self.s)
        for t, w in zip(tags, weights):
            if w == 0.0:
                continue
            sk = self._skill(t)
            sk.mu += sk.sigma * sk.sigma * grad * w
            sk.sigma = math.sqrt(1.0 / (1.0 / (sk.sigma * sk.sigma) + info * w))
