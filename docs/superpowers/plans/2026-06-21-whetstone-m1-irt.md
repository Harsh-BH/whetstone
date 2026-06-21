# Whetstone M1 (IRT baseline + skill radar) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans (inline, DB-coupled) to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** An accurate, **calibrated** per-topic skill estimate (μ,σ per CF tag) you can see — the L1 Bayesian Rasch model + eval gate + a skill radar.

**Architecture:** `model/irt.py` holds a per-tag Gaussian skill state with a Rasch outcome model (`P=σ((θ_eff−b)/s)`, `θ_eff=min` over tags) and an online Laplace update (μ moves by `σ²·grad`, σ shrinks by Fisher info). `eval/` does a temporal split, replays the model prequentially, and scores AUC/ECE vs baselines (the M1→M2 gate). `web/` renders μ±σ per tag as a recharts radar from a JSON dump.

**Tech Stack:** Python 3.12, numpy, scipy, scikit-learn (metrics), existing psycopg3/pydantic; React + Vite + Tailwind + recharts (radar).

## Global Constraints (carried from `CLAUDE.md` / docs)

- **Predict first-attempt success**, not eventual: `y = (first_verdict == 'OK')`. Docs/03: "the model cares most about whether the *first* submission succeeded." (On SecondThread: ~57% positive — balanced.)
- Use only episodes with a known problem rating `b_p` (`p.rating IS NOT NULL`) and ≥1 tag.
- **Temporal split only, never random** (docs/07 A1) — leakage. Order by `first_seen_at`.
- `θ and b` share the **CF rating scale**; logistic scale `s` initialised to the Elo scale `400/ln(10) ≈ 173.7`, then fit (docs/03).
- `θ_eff = min` over a problem's tags (docs/03 default; alternatives are docs/07 ablations).
- **Calibration is a release gate** (docs/07 A2, P7). **Gate: ECE ≤ 0.05 AND test AUC ≥ 0.70 AND AUC ≥ per-tag-baseline + 0.05.** A red gate is reported, never loosened (`CLAUDE.md`, `/eval`).
- Pedagogical/scale constants live in `config.py`, annotated (docs/05).
- Reproducible: deterministic; no RNG in the fit/eval path except seeded synthetic tests.

### Validation data (decided this session)
`harsh-bh` has 2 episodes — too few. M1 is built and gated against a **data-rich public handle, `SecondThread`** (2,895 episodes, 1,939 with `b_p`, 39 tags), ingested under `user_id='SecondThread'`. The single-user design is unchanged; the real user's account plugs in once it has history.

### Credit-assignment choice (the known-messy part, docs/03)
Predict with `θ_eff = min`. On update, distribute the **same** error `(y−p)` to **all** contributing tags (each with its own gain `σ_t²`). This matches docs/03 "update the contributing μ_t toward the outcome," is simple and testable, and its bias (over-crediting strong co-tags) is an explicit docs/07 aggregation ablation — not fixed in M1.

### Branch
`m1-irt` (already cut from `main`).

---

### Task 1: IRT constants in config + dependencies

**Files:**
- Modify: `config.py` (add IRT constants)
- Modify: `pyproject.toml` (add numpy, scipy, scikit-learn)

**Interfaces:**
- Produces: `config.IRT_S` (float), `config.PRIOR_MU` (float), `config.PRIOR_SIGMA` (float).

- [ ] **Step 1: Add deps**

Run: `uv add numpy scipy scikit-learn`
Expected: resolves and installs; `uv run python -c "import numpy, scipy, sklearn; print('ok')"` → `ok`.

- [ ] **Step 2: Add IRT constants to `config.py`** (append after the pedagogical block)

```python
import math

# --- IRT / knowledge-model constants (docs/03). Scale shared with CF ratings. ---
IRT_S = 400.0 / math.log(10)   # logistic scale ~ Elo (a 400-pt gap ≈ CF win prob); fit per user
PRIOR_MU = 1500.0              # cold-start θ prior when CF rating unknown (docs/03 cold-start)
PRIOR_SIGMA = 350.0            # high initial per-tag uncertainty (drives Assess in M2)
```

- [ ] **Step 3: Verify**

Run: `uv run python -c "import config; print(round(config.IRT_S,1), config.PRIOR_MU, config.PRIOR_SIGMA)"`
Expected: `173.7 1500.0 350.0`

- [ ] **Step 4: Commit**

```bash
git add config.py pyproject.toml uv.lock
git commit -m "feat(m1): IRT scale/prior constants + numpy/scipy/sklearn deps"
```

---

### Task 2: The Rasch skill model (`model/irt.py`)

**Files:**
- Create: `model/__init__.py` (empty)
- Create: `model/irt.py`
- Test: `tests/test_irt.py`

**Interfaces:**
- Produces:
  - `TagSkill(mu: float, sigma: float)` dataclass.
  - `SkillModel(s: float = IRT_S, prior_mu: float = PRIOR_MU, prior_sigma: float = PRIOR_SIGMA)`:
    - `theta_eff(tags: list[str]) -> float` — min μ over tags (prior_mu if none).
    - `predict_solve(b: float, tags: list[str]) -> tuple[float, float]` — `(p, fisher_info)`.
    - `update(b: float, tags: list[str], y: int) -> None` — online Laplace update of each contributing tag.
    - `.tags: dict[str, TagSkill]` — current state (read by the radar/dashboard).
- Consumed by: `eval/` (Task 4), `web/` data dump (Task 5).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_irt.py
import math
import random
from model.irt import SkillModel, TagSkill


def test_p_half_when_theta_equals_b():
    m = SkillModel()
    m.tags["dp"] = TagSkill(mu=1500, sigma=100)
    p, info = m.predict_solve(b=1500, tags=["dp"])
    assert abs(p - 0.5) < 1e-9
    # Fisher info is maximal at p=0.5: 0.25 / s^2
    assert abs(info - 0.25 / m.s**2) < 1e-12


def test_monotonic_in_skill():
    m = SkillModel()
    m.tags["dp"] = TagSkill(mu=1800, sigma=100)
    p_easy, _ = m.predict_solve(b=1200, tags=["dp"])
    p_hard, _ = m.predict_solve(b=2400, tags=["dp"])
    assert p_easy > 0.5 > p_hard


def test_theta_eff_is_min_over_tags():
    m = SkillModel()
    m.tags["dp"] = TagSkill(mu=2000, sigma=100)
    m.tags["math"] = TagSkill(mu=1400, sigma=100)
    assert m.theta_eff(["dp", "math"]) == 1400  # gated by the weakest


def test_update_moves_mu_toward_outcome_and_shrinks_sigma():
    m = SkillModel()
    m.tags["dp"] = TagSkill(mu=1500, sigma=200)
    s0 = m.tags["dp"].sigma
    m.update(b=1700, tags=["dp"], y=1)  # solved a harder-than-skill problem
    assert m.tags["dp"].mu > 1500          # skill revised up
    assert m.tags["dp"].sigma < s0          # more certain


def test_unknown_tag_starts_at_prior():
    m = SkillModel(prior_mu=1500, prior_sigma=350)
    p, _ = m.predict_solve(b=1500, tags=["never_seen"])
    assert abs(p - 0.5) < 1e-9
    assert m.tags["never_seen"].mu == 1500 and m.tags["never_seen"].sigma == 350


def test_recovers_true_skill_on_synthetic_stream():
    # A single-tag learner with fixed true theta; generate first-attempt outcomes
    # across a range of problem difficulties; online update should converge near truth.
    rng = random.Random(0)
    true_theta, s = 1900.0, 173.7
    m = SkillModel(s=s, prior_mu=1500, prior_sigma=350)
    for _ in range(4000):
        b = rng.uniform(1000, 2800)
        p_true = 1 / (1 + math.exp(-(true_theta - b) / s))
        y = 1 if rng.random() < p_true else 0
        m.update(b=b, tags=["dp"], y=y)
    assert abs(m.tags["dp"].mu - true_theta) < 120  # within ~0.7 sigma of Elo scale
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_irt.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'model.irt'`.

- [ ] **Step 3: Write the implementation**

```python
# model/__init__.py
```

```python
# model/irt.py
"""L1 Bayesian per-topic IRT (Rasch/1PL) skill model (docs/03).

State: per-tag Gaussian N(mu, sigma^2) on the CF rating scale.
Outcome: P(first-attempt solve) = sigmoid((theta_eff - b) / s), theta_eff = min over tags.
Update: online Laplace — mu += sigma^2 * grad; 1/sigma^2 += Fisher info. sigma is
first-class (it drives Assess vs Train in M2).
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
        grad = (y - p) / self.s            # d/dtheta of the log-likelihood
        info = p * (1.0 - p) / (self.s * self.s)
        for t in tags:
            sk = self._skill(t)
            sk.mu += sk.sigma * sk.sigma * grad
            sk.sigma = math.sqrt(1.0 / (1.0 / (sk.sigma * sk.sigma) + info))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_irt.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add model/__init__.py model/irt.py tests/test_irt.py
git commit -m "feat(m1): Bayesian per-topic Rasch IRT model + Fisher info"
```

---

### Task 3: Eval metrics (`eval/metrics.py`)

**Files:**
- Create: `eval/__init__.py` (empty)
- Create: `eval/metrics.py`
- Test: `tests/test_metrics.py`

**Interfaces:**
- Produces (pure functions over `list[float]` preds and `list[int]` labels):
  - `auc(y, p) -> float` (sklearn roc_auc_score; returns `float('nan')` if one class).
  - `log_loss_(y, p) -> float`, `accuracy(y, p, thresh=0.5) -> float`, `brier(y, p) -> float`.
  - `ece(y, p, n_bins=10) -> float` — expected calibration error.
  - `reliability(y, p, n_bins=10) -> list[dict]` — per-bin `{conf, acc, n}` for the diagram.
- Consumed by: `eval/run_m1.py` (Task 4).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_metrics.py
import math
from eval import metrics


def test_auc_perfect_separation():
    y = [0, 0, 1, 1]
    p = [0.1, 0.2, 0.8, 0.9]
    assert metrics.auc(y, p) == 1.0


def test_auc_single_class_is_nan():
    assert math.isnan(metrics.auc([1, 1, 1], [0.5, 0.6, 0.7]))


def test_perfect_calibration_has_zero_ece():
    # two bins, predictions exactly match empirical frequency
    y = [0, 1] * 50              # 50% positive
    p = [0.5] * 100
    assert metrics.ece(y, p, n_bins=10) < 1e-9


def test_ece_detects_miscalibration():
    y = [0] * 100               # always negative
    p = [0.9] * 100             # but model says 90%
    assert abs(metrics.ece(y, p, n_bins=10) - 0.9) < 1e-9


def test_accuracy_and_brier():
    y = [1, 0, 1, 0]
    p = [0.9, 0.1, 0.4, 0.2]
    assert metrics.accuracy(y, p) == 0.75      # third one wrong at 0.5 thresh
    assert abs(metrics.brier(y, p) - (0.01 + 0.01 + 0.36 + 0.04) / 4) < 1e-9


def test_reliability_bins_sum_to_n():
    y = [0, 1, 1, 0, 1]
    p = [0.2, 0.8, 0.6, 0.3, 0.9]
    bins = metrics.reliability(y, p, n_bins=5)
    assert sum(b["n"] for b in bins) == 5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eval.metrics'`.

- [ ] **Step 3: Write the implementation**

```python
# eval/__init__.py
```

```python
# eval/metrics.py
"""Knowledge-model eval metrics (docs/07 A1/A2). Pure functions; no DB."""
import math

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


def ece(y: list[int], p: list[float], n_bins: int = 10) -> float:
    y_arr, p_arr = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    edges = _bin_edges(n_bins)
    total = len(p_arr)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p_arr > lo) & (p_arr <= hi) if lo > 0 else (p_arr >= lo) & (p_arr <= hi)
        if not mask.any():
            continue
        conf = p_arr[mask].mean()
        acc = y_arr[mask].mean()
        e += (mask.sum() / total) * abs(acc - conf)
    return float(e)


def reliability(y: list[int], p: list[float], n_bins: int = 10) -> list[dict]:
    y_arr, p_arr = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    edges = _bin_edges(n_bins)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p_arr > lo) & (p_arr <= hi) if lo > 0 else (p_arr >= lo) & (p_arr <= hi)
        n = int(mask.sum())
        out.append({
            "lo": float(lo), "hi": float(hi), "n": n,
            "conf": float(p_arr[mask].mean()) if n else None,
            "acc": float(y_arr[mask].mean()) if n else None,
        })
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_metrics.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add eval/__init__.py eval/metrics.py tests/test_metrics.py
git commit -m "feat(m1): eval metrics (AUC, ECE, Brier, reliability)"
```

---

### Task 4: M1 eval harness + gate (`eval/dataset.py`, `eval/run_m1.py`)

**Files:**
- Create: `eval/dataset.py`
- Create: `eval/run_m1.py`
- Test: `tests/test_eval_m1.py`

**Interfaces:**
- `eval/dataset.py`:
  - `Record(b: float, tags: list[str], y: int, t)` dataclass.
  - `load_records(conn, user_id: str) -> list[Record]` — episodes with rating + tags, `y=(first_verdict=='OK')`, ordered by `first_seen_at`.
  - `temporal_split(recs, frac=0.8) -> tuple[list, list]` — chronological.
- `eval/run_m1.py`:
  - `fit_s(train, prior_mu, grid=...) -> float` — replay train, return s minimizing train log-loss.
  - `per_tag_baseline(train, test) -> list[float]` — predict mean train `y` over a problem's tags.
  - `evaluate(conn, user_id, cf_rating=None) -> dict` — fit, prequential test, return metrics + baselines + gate booleans.
  - `main()` — connect, look up CF rating, run `evaluate`, print the gate table. Wired to `/eval`.

- [ ] **Step 1: Write the failing test** (synthetic learner end-to-end; no DB)

```python
# tests/test_eval_m1.py
import math
import random
from eval.dataset import Record, temporal_split
from eval import run_m1


def _synthetic(n=3000, true_theta=1900.0, s=173.7, seed=0):
    rng = random.Random(seed)
    recs = []
    for i in range(n):
        b = rng.uniform(1000, 2800)
        p = 1 / (1 + math.exp(-(true_theta - b) / s))
        y = 1 if rng.random() < p else 0
        recs.append(Record(b=b, tags=["dp"], y=y, t=i))
    return recs


def test_temporal_split_is_chronological():
    recs = [Record(b=1, tags=["x"], y=0, t=i) for i in range(10)]
    tr, te = temporal_split(recs, frac=0.8)
    assert len(tr) == 8 and len(te) == 2
    assert te[0].t == 8


def test_evaluate_on_synthetic_beats_baseline_and_calibrates():
    recs = _synthetic()
    res = run_m1.evaluate_records(recs, cf_rating=1500)
    # A correct Rasch model on data generated by a Rasch model: strong AUC, low ECE.
    assert res["auc"] >= 0.70
    assert res["auc"] >= res["per_tag_auc"] + 0.05
    assert res["ece"] <= 0.05
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_eval_m1.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'eval.dataset'`.

- [ ] **Step 3: Write `eval/dataset.py`**

```python
# eval/dataset.py
"""Load + split interaction records for knowledge-model eval (docs/07 A1)."""
from dataclasses import dataclass


@dataclass
class Record:
    b: float
    tags: list[str]
    y: int
    t: object  # sort key (first_seen_at)


def load_records(conn, user_id: str) -> list["Record"]:
    rows = conn.execute(
        """
        SELECT p.rating AS b, p.tags AS tags, i.first_verdict AS fv, i.first_seen_at AS t
        FROM interactions i JOIN problems p ON p.id = i.problem_id
        WHERE i.user_id = %s AND p.rating IS NOT NULL
          AND p.tags IS NOT NULL AND array_length(p.tags, 1) > 0
          AND i.first_verdict IS NOT NULL
        ORDER BY i.first_seen_at
        """,
        (user_id,),
    ).fetchall()
    return [Record(b=float(r["b"]), tags=list(r["tags"]),
                   y=1 if r["fv"] == "OK" else 0, t=r["t"]) for r in rows]


def temporal_split(recs: list["Record"], frac: float = 0.8) -> tuple[list, list]:
    k = int(len(recs) * frac)
    return recs[:k], recs[k:]
```

- [ ] **Step 4: Write `eval/run_m1.py`**

```python
# eval/run_m1.py
"""M1 knowledge-model eval + acceptance gate (docs/07 A1/A2).

Temporal split, fit s on train, replay train, prequential test (predict then
update). Compare to global + per-tag baselines. Gate: AUC>=0.70, AUC>=per_tag+0.05,
ECE<=0.05. Deterministic; seeds fixed.
"""
from eval import metrics
from eval.dataset import Record, load_records, temporal_split
from model.irt import SkillModel

S_GRID = [80.0, 120.0, 173.7, 250.0, 350.0, 500.0]


def _replay_train(recs: list[Record], s: float, prior_mu: float) -> SkillModel:
    m = SkillModel(s=s, prior_mu=prior_mu)
    for r in recs:
        m.update(r.b, r.tags, r.y)
    return m


def _prequential(model: SkillModel, recs: list[Record]) -> tuple[list[int], list[float]]:
    ys, ps = [], []
    for r in recs:
        p, _ = model.predict_solve(r.b, r.tags)
        ys.append(r.y)
        ps.append(p)
        model.update(r.b, r.tags, r.y)
    return ys, ps


def fit_s(train: list[Record], prior_mu: float, grid=S_GRID) -> float:
    best_s, best_ll = grid[0], float("inf")
    for s in grid:
        m = SkillModel(s=s, prior_mu=prior_mu)
        ys, ps = _prequential(m, train)  # prequential train log-loss
        if len(set(ys)) < 2:
            continue
        ll = metrics.log_loss_(ys, ps)
        if ll < best_ll:
            best_ll, best_s = ll, s
    return best_s


def per_tag_baseline(train: list[Record], test: list[Record]) -> list[float]:
    rate: dict[str, list[int]] = {}
    glob = sum(r.y for r in train) / max(1, len(train))
    for r in train:
        for t in r.tags:
            rate.setdefault(t, []).append(r.y)
    means = {t: sum(v) / len(v) for t, v in rate.items()}
    out = []
    for r in test:
        vals = [means[t] for t in r.tags if t in means]
        out.append(sum(vals) / len(vals) if vals else glob)
    return out


def evaluate_records(recs: list[Record], cf_rating: float | None) -> dict:
    prior_mu = float(cf_rating) if cf_rating else SkillModel().prior_mu
    train, test = temporal_split(recs)
    s = fit_s(train, prior_mu)
    model = _replay_train(train, s, prior_mu)
    ys, ps = _prequential(model, test)
    bl = per_tag_baseline(train, test)
    auc = metrics.auc(ys, ps)
    per_tag_auc = metrics.auc(ys, bl)
    ece = metrics.ece(ys, ps)
    return {
        "n_train": len(train), "n_test": len(test), "s": s,
        "auc": auc, "accuracy": metrics.accuracy(ys, ps),
        "log_loss": metrics.log_loss_(ys, ps), "brier": metrics.brier(ys, ps),
        "ece": ece, "reliability": metrics.reliability(ys, ps),
        "per_tag_auc": per_tag_auc,
        "gate_auc": (auc >= 0.70) and (auc >= per_tag_auc + 0.05),
        "gate_ece": ece <= 0.05,
    }


def evaluate(conn, user_id: str, cf_rating: float | None = None) -> dict:
    return evaluate_records(load_records(conn, user_id), cf_rating)


def main(user_id: str = "SecondThread") -> None:
    from ingest import db
    from ingest.cf_client import CFClient

    client = CFClient()
    try:
        rating = client.user_info(user_id).rating
    except Exception:
        rating = None
    finally:
        client.close()
    conn = db.connect()
    try:
        res = evaluate(conn, user_id, cf_rating=rating)
    finally:
        conn.close()

    print(f"\n=== M1 knowledge-model eval — user={user_id} (cf_rating={rating}) ===")
    print(f"train/test: {res['n_train']}/{res['n_test']}   fit s={res['s']:.1f}")
    print(f"AUC        : {res['auc']:.3f}   (per-tag baseline {res['per_tag_auc']:.3f})")
    print(f"accuracy   : {res['accuracy']:.3f}")
    print(f"log-loss   : {res['log_loss']:.3f}   Brier {res['brier']:.3f}")
    print(f"ECE        : {res['ece']:.3f}")
    print("\nreliability (conf -> acc, n):")
    for bn in res["reliability"]:
        if bn["n"]:
            print(f"  [{bn['lo']:.1f}-{bn['hi']:.1f}] conf={bn['conf']:.2f} acc={bn['acc']:.2f} n={bn['n']}")
    g_auc = "PASS" if res["gate_auc"] else "FAIL"
    g_ece = "PASS" if res["gate_ece"] else "FAIL"
    print(f"\n--- GATE ---")
    print(f"AUC>=0.70 & +0.05 over baseline : {g_auc}")
    print(f"ECE<=0.05                        : {g_ece}")
    print("M1 -> M2:", "PASS" if (res["gate_auc"] and res["gate_ece"]) else "BLOCKED")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the synthetic test**

Run: `uv run pytest tests/test_eval_m1.py -v`
Expected: PASS (2 passed).

- [ ] **Step 6: Run the REAL gate against SecondThread**

Run: `uv run python -m eval.run_m1`
Expected: prints the eval + gate table. **Record the real AUC/ECE.** If a gate is FAIL, report it (do not loosen) — diagnose (s grid, credit assignment, prior).

- [ ] **Step 7: Commit**

```bash
git add eval/dataset.py eval/run_m1.py tests/test_eval_m1.py
git commit -m "feat(m1): temporal-split eval harness + M1 acceptance gate"
```

---

### Task 5: Wire `/eval` + persist a θ snapshot + radar data dump

**Files:**
- Create: `model/snapshot.py`
- Modify: `Makefile` (add `eval`, `train` targets)
- Test: `tests/test_snapshot.py`

**Interfaces:**
- `model/snapshot.py`:
  - `fit_user(conn, user_id, cf_rating) -> SkillModel` — replay ALL records (no split) for the live state.
  - `save_snapshot(conn, user_id, model) -> int` — write each tag's μ/σ to `topic_skill` with `snapshot_at=now()`.
  - `dump_radar(model, path) -> None` — write `web/public/skills.json` = `[{tag, mu, sigma}]`.

- [ ] **Step 1: Write the failing test** (DB-backed; uses `whetstone_test`)

```python
# tests/test_snapshot.py
import os, json, psycopg, pytest
from model.irt import SkillModel, TagSkill
from model import snapshot
from ingest import db

DB = os.environ.get("DATABASE_URL", "postgresql://whetstone:whetstone@localhost:5432/whetstone")


def _db_up():
    try:
        psycopg.connect(DB, connect_timeout=2).close(); return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _db_up(), reason="Postgres not running")


def test_save_snapshot_writes_topic_skill():
    c = db.connect()
    c.execute("DELETE FROM topic_skill WHERE user_id='_snap'")
    c.commit()
    m = SkillModel()
    m.tags["dp"] = TagSkill(mu=1900, sigma=120)
    m.tags["math"] = TagSkill(mu=1700, sigma=90)
    n = snapshot.save_snapshot(c, "_snap", m)
    c.commit()
    assert n == 2
    rows = c.execute("SELECT tag, mu, sigma FROM topic_skill WHERE user_id='_snap'").fetchall()
    assert {r["tag"] for r in rows} == {"dp", "math"}
    c.close()


def test_dump_radar_writes_json(tmp_path):
    m = SkillModel()
    m.tags["dp"] = TagSkill(mu=1900, sigma=120)
    path = tmp_path / "skills.json"
    snapshot.dump_radar(m, str(path))
    data = json.loads(path.read_text())
    assert data[0]["tag"] == "dp" and data[0]["mu"] == 1900
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_snapshot.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'model.snapshot'`.

- [ ] **Step 3: Write `model/snapshot.py`**

```python
# model/snapshot.py
"""Fit the live skill state on all data, persist a topic_skill snapshot, dump radar JSON."""
import json

from eval.dataset import load_records
from model.irt import SkillModel


def fit_user(conn, user_id: str, cf_rating: float | None) -> SkillModel:
    prior_mu = float(cf_rating) if cf_rating else SkillModel().prior_mu
    m = SkillModel(prior_mu=prior_mu)
    for r in load_records(conn, user_id):
        m.update(r.b, r.tags, r.y)
    return m


def save_snapshot(conn, user_id: str, model: SkillModel) -> int:
    rows = [(user_id, tag, sk.mu, sk.sigma) for tag, sk in model.tags.items()]
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO topic_skill (user_id, tag, mu, sigma, mastered, snapshot_at)
               VALUES (%s, %s, %s, %s, false, now())""",
            rows,
        )
    return len(rows)


def dump_radar(model: SkillModel, path: str) -> None:
    data = [{"tag": t, "mu": sk.mu, "sigma": sk.sigma}
            for t, sk in sorted(model.tags.items())]
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
```

- [ ] **Step 4: Add Makefile targets**

```makefile
eval:
	uv run python -m eval.run_m1

train:
	uv run python -m model.snapshot
```

And add a `__main__` to `model/snapshot.py`:

```python
if __name__ == "__main__":
    import os
    from ingest import db
    from ingest.cf_client import CFClient

    user = os.environ.get("CF_HANDLE") or "SecondThread"
    client = CFClient()
    try:
        rating = client.user_info(user).rating
    except Exception:
        rating = None
    finally:
        client.close()
    conn = db.connect()
    try:
        m = fit_user(conn, user, rating)
        n = save_snapshot(conn, user, m)
        conn.commit()
    finally:
        conn.close()
    os.makedirs("web/public", exist_ok=True)
    dump_radar(m, "web/public/skills.json")
    print(f"fit {len(m.tags)} tags for {user}; snapshot rows={n}; radar -> web/public/skills.json")
```

- [ ] **Step 5: Run tests + produce the snapshot/radar data**

Run: `uv run pytest tests/test_snapshot.py -v` → PASS.
Run: `uv run python -m model.snapshot` (env `CF_HANDLE=SecondThread`) → writes `web/public/skills.json`.

- [ ] **Step 6: Commit**

```bash
git add model/snapshot.py Makefile tests/test_snapshot.py web/public/skills.json
git commit -m "feat(m1): topic_skill snapshot + radar JSON dump + /eval,/train wiring"
```

---

### Task 6: Skill radar (React + Vite + recharts)

**Files:**
- Create: `web/package.json`, `web/vite.config.js`, `web/index.html`, `web/src/main.jsx`, `web/src/Radar.jsx`, `web/tailwind.config.js`, `web/src/index.css`
- (Uses `web/public/skills.json` from Task 5.)

**Interfaces:**
- Produces: a Vite dev app rendering a recharts `RadarChart` of μ per tag (radius), with a second series at `μ−σ` to show the uncertainty band. Title shows the user + target band line at `R=1900`.

- [ ] **Step 1: Scaffold Vite React app**

Run: `cd web && npm create vite@latest . -- --template react` (accept overwrite into the dir), then `npm install recharts`.

- [ ] **Step 2: `web/src/Radar.jsx`** (reads `/skills.json`)

```jsx
import { useEffect, useState } from "react";
import { Radar, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Legend, ResponsiveContainer } from "recharts";

export default function SkillRadar() {
  const [data, setData] = useState([]);
  useEffect(() => {
    fetch("/skills.json").then((r) => r.json()).then((rows) =>
      setData(rows.map((d) => ({ tag: d.tag, mu: Math.round(d.mu), low: Math.round(d.mu - d.sigma) })))
    );
  }, []);
  return (
    <div style={{ width: "100%", height: 600 }}>
      <h2 style={{ textAlign: "center" }}>Skill radar (μ per tag, μ−σ band)</h2>
      <ResponsiveContainer>
        <RadarChart data={data} outerRadius="75%">
          <PolarGrid />
          <PolarAngleAxis dataKey="tag" />
          <PolarRadiusAxis domain={[800, 3000]} />
          <Radar name="μ (skill)" dataKey="mu" stroke="#2563eb" fill="#2563eb" fillOpacity={0.4} />
          <Radar name="μ−σ (conservative)" dataKey="low" stroke="#9ca3af" fill="#9ca3af" fillOpacity={0.2} />
          <Legend />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  );
}
```

- [ ] **Step 3: `web/src/main.jsx`** render `<SkillRadar/>`.

```jsx
import React from "react";
import ReactDOM from "react-dom/client";
import SkillRadar from "./Radar.jsx";

ReactDOM.createRoot(document.getElementById("root")).render(<SkillRadar />);
```

- [ ] **Step 4: Verify it renders**

Run: `cd web && npm run dev`; open the served URL; confirm the radar shows ~39 tag spokes with μ and the μ−σ band. Screenshot for the PR.

- [ ] **Step 5: Commit**

```bash
git add web
git commit -m "feat(m1): skill radar (React+recharts) reading skills.json"
```

---

## Self-Review

**1. Spec coverage** (docs/08 M1 checklist):
- `irt.py` Rasch + Bayesian (μ,σ) online update + Fisher info, θ_eff=min → **Task 2**. ✓
- Cold-start prior from CF rating → `prior_mu=cf_rating` in **Tasks 4 & 5** (unrated → PRIOR_MU). ✓
- `eval/` temporal-split next-step AUC + calibration (ECE, reliability) → **Tasks 3 & 4**. ✓
- `web/` skill radar (μ per tag, σ band) → **Task 6**. ✓
- Gate ECE≤0.05 & AUC≥0.70 (+0.05 vs per-tag) → **Task 4** (`gate_auc`, `gate_ece`). ✓

**2. Placeholder scan:** No TBD/"add error handling"/"similar to". Every code step is complete and runnable.

**3. Type consistency:** `SkillModel`/`TagSkill` identical across Tasks 2/4/5. `Record(b,tags,y,t)` identical in Tasks 3/4. `evaluate_records` is the seam both the synthetic test (Task 4) and `evaluate` (DB) call. `metrics.*` signatures consistent. `web/public/skills.json` schema (`{tag,mu,sigma}`) written in Task 5, read in Task 6.

**Known risks for execution (flagged honestly):**
- **Real gate may FAIL.** SecondThread first-attempt data is ~57% positive (good), but the min-credit-assignment bias and a coarse `s` grid could miss ECE≤0.05 or the AUC margin. If so: report the red gate, then iterate on (a) finer s grid, (b) credit assignment (argmin vs all-tags) as a docs/07 ablation — never loosen the threshold.
- **Radar (Task 6) needs Node/npm** and is visualization, not the gate. The milestone's rigor blocker is Tasks 1–5; Task 6 can land separately.
