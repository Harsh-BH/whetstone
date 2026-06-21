"""L1 Bayesian per-topic IRT (Rasch/1PL) skill model (docs/03).

State: per-tag Gaussian N(mu, sigma^2) on the CF rating scale.
Outcome: P(first-attempt solve) = sigmoid((theta_eff - b) / s), theta_eff = min over tags.
Update: online Laplace — mu += sigma^2 * grad; 1/sigma^2 += Fisher info. sigma is
first-class (it drives Assess vs Train in M2).

Credit assignment (the docs/03 "messiest choice"): predict with theta_eff = min, and
SPLIT one observation's evidence across its n contributing tags (gradient and Fisher
info each /n) rather than duplicating the full error to every tag — duplicating
inflates mu and over-predicts. The split-vs-argmin-vs-softmin choice is a docs/07
aggregation ablation.
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
    tags: dict[str, TagSkill] = field(default_factory=dict)

    def _skill(self, tag: str) -> TagSkill:
        sk = self.tags.get(tag)
        if sk is None:
            sk = TagSkill(mu=self.prior_mu, sigma=self.prior_sigma)
            self.tags[tag] = sk
        return sk

    def theta_eff(self, tags: list[str]) -> float:
        if not tags:
            return self.prior_mu
        return min(self._skill(t).mu for t in tags)

    def predict_solve(self, b: float, tags: list[str]) -> tuple[float, float]:
        theta = self.theta_eff(tags)
        p = _sigmoid((theta - b) / self.s)
        info = p * (1.0 - p) / (self.s * self.s)
        return p, info

    def update(self, b: float, tags: list[str], y: int) -> None:
        if not tags:
            return
        p, _ = self.predict_solve(b, tags)
        # Split the evidence across contributing tags (conserve, don't duplicate):
        # one observation is 1/n of the evidence for each of its n tags. Duplicating
        # the full gradient over co-tags inflates mu and over-predicts (docs/03
        # credit-assignment; docs/07 aggregation ablation).
        n = len(tags)
        grad = (y - p) / self.s  # d/dtheta of the log-likelihood
        info = p * (1.0 - p) / (self.s * self.s)
        for t in tags:
            sk = self._skill(t)
            sk.mu += sk.sigma * sk.sigma * grad / n
            sk.sigma = math.sqrt(1.0 / (1.0 / (sk.sigma * sk.sigma) + info / n))
