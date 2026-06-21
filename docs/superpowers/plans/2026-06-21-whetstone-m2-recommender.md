# Whetstone M2 (greedy two-mode recommender + daily set) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (inline, DB-coupled). Steps use checkbox (`- [ ]`) syntax.

**Goal:** A usable daily problem set produced by the two-mode (Assess/Train) greedy policy — gap-weighted, frontier-restricted, interleaved, desirable-difficulty — with every served problem logged (mode, predicted_p, propensity) for offline policy evaluation, and an eval that shows it beats random + fixed-curriculum.

**Architecture:** `recommender/` consumes the M1 `SkillModel` (μ,σ per tag). `prereq_dag.py` gates which topics are open. `assess.py` picks max-Fisher-info problems for high-σ topics; `train.py` gap-weights + frontier-restricts + stochastically samples topics, then picks band-difficulty problems; `compose.py` blends them into an interleaved daily set and logs to `recommendations`. `eval/sim_m2.py` simulates each policy against the M1 model as the learner oracle and reports the gate.

**Tech Stack:** Python 3.12 (numpy/scipy), psycopg3, existing model/eval; React/recharts for the daily-set view.

## Global Constraints (from `CLAUDE.md` / docs)

- Two modes, **different objectives**, chosen by uncertainty — do not collapse them (`CLAUDE.md`, docs/04).
- Assess targets first-attempt P(solve) ≈ 0.5 (max Fisher info); Train targets the band `(0.55, 0.80)` (docs/02 P2, `config.TRAIN_TARGET_BAND`).
- Interleave: `max_consecutive_same_tag = 1` (docs/02 P4, `config.MAX_CONSECUTIVE_SAME_TAG`).
- Recommend only on the prereq-DAG frontier (docs/02 P6, `config.FRONTIER_ONLY`).
- **`recommendations.propensity` is mandatory** and must be the true probability the action was sampled (docs/06) — so the policy is **stochastic**, not deterministic argmax.
- Pedagogical constants live in `config.py`, annotated (docs/05).
- Any model/recommender change runs `/eval`; a red gate is stopped and reported, never loosened (`CLAUDE.md`).

---

## ⚠️ Key design decisions — FLAGGED FOR YOUR REVIEW

M2 depends on three things that later milestones formalize. My calls (each reversible):

1. **Frontier without mastery (mastery is M3).** `frontier_only` needs "prereqs satisfied," but the mastery criterion (`docs/03`) ships in M3. **Decision:** M2 proxy — a prereq tag is "satisfied" when `μ_prereq ≥ R_band − FRONTIER_MARGIN` (default margin 200). Replaced by the real mastery criterion in M3. *Alternative: open all topics in M2, add frontier in M3.*

2. **Reviews need FSRS (FSRS is M3).** The docs' 65/20/15 (train/review/stretch) blend needs the spaced-review queue, which is M3. **Decision:** M2 ships **65% train + 15% stretch reallocated to 80% train / 20% stretch** (no review slice yet); the full 65/20/15 lands in M3 when FSRS exists. *Alternative: a crude "re-serve an old solved problem" review stub now.*

3. **OPE needs logged deployment data (we have none yet).** Real IPS/replay OPE (docs/07 B2) needs a history of served problems with outcomes — a fresh system has none. **Decision:** M2's gate uses a **simulated comparison** (docs/07 B3-style) with the **M1 model as the learner oracle** (outcomes ~ M1 P(solve); skill updated by the M1 online rule), comparing greedy vs random vs fixed-curriculum vs difficulty-match on skill-gain-per-item. The `recommendations` logging (incl. real `propensity`) is wired now so **real OPE becomes available once the tool is actually used**. Caveat (docs/07): a simulator built from the same model family it evaluates is circular — reported as suggestive, validated properly against held-out reality in M4/M5. *This is the honest limit of evaluating a brand-new policy with no logs.*

If you'd prefer different calls on any of these, say so before I build.

---

## Repo additions

```
recommender/__init__.py
recommender/prereq_dag.py   # hand-seeded DAG over the ~37 CF tags + frontier()
recommender/assess.py       # max-Fisher-info problem selection
recommender/train.py        # gap-weighted, frontier-restricted, stochastic topic+problem selection
recommender/compose.py      # daily-set blend + interleave + log to recommendations
recommender/candidates.py   # unseen-problem candidate pool from the catalog
eval/sim_m2.py              # simulated policy comparison + mode-behavior + anti-Goodhart (the M2 gate)
api/app.py, api/routes.py   # FastAPI: GET /daily-set, GET /skills
web/src/DailySet.jsx        # "Today's set" with predicted P(solve) + why-this-problem
```

---

### Task 1: M2 config + prereq DAG

**Files:** Create `recommender/__init__.py`, `recommender/prereq_dag.py`; modify `config.py`; Test `tests/test_prereq_dag.py`.

**Interfaces:**
- `config`: `ASSESS_SIGMA_THRESHOLD=120.0` (σ above → Assess; P8), `FRONTIER_MARGIN=200.0`, `STRETCH_TARGET_P=0.40` (docs/04), `DAILY_BLEND={"train":0.8,"stretch":0.2}` (M2; M3 adds review), `MINUTES_PER_PROBLEM=30`.
- `prereq_dag.py`:
  - `PREREQS: dict[str, list[str]]` — CF-tag DAG (edges child→parents).
  - `frontier(model, all_tags, r_band, margin) -> set[str]` — tags whose every prereq has `μ ≥ r_band − margin` (roots always open).

- [ ] **Step 1: failing test**

```python
# tests/test_prereq_dag.py
from model.irt import SkillModel, TagSkill
from recommender import prereq_dag


def test_roots_always_open():
    m = SkillModel()
    fr = prereq_dag.frontier(m, ["math", "implementation"], r_band=1900, margin=200)
    assert "math" in fr and "implementation" in fr  # roots have no prereqs


def test_child_gated_by_unmastered_prereq():
    m = SkillModel(prior_mu=1000)  # everything weak
    # 'number theory' requires 'math'; math mu=1000 < 1900-200 -> gated
    fr = prereq_dag.frontier(m, ["math", "number theory"], r_band=1900, margin=200)
    assert "number theory" not in fr


def test_child_opens_when_prereq_strong():
    m = SkillModel()
    m.tags["math"] = TagSkill(mu=1800, sigma=100)  # >= 1900-200
    fr = prereq_dag.frontier(m, ["math", "number theory"], r_band=1900, margin=200)
    assert "number theory" in fr
```

- [ ] **Step 2-4:** implement `prereq_dag.py` (seed a reasonable CF-tag DAG: roots = implementation/math/greedy/brute force/sortings; then binary search, two pointers, dp, dfs and similar, dsu, strings, bitmasks, number theory → parents; trees, shortest paths, data structures→segment tree, flows, combinatorics, geometry, hashing, fft, etc. as deeper nodes). `frontier()` returns roots + any tag all of whose prereqs satisfy `μ ≥ r_band−margin`. Add config constants. Run tests → PASS.

- [ ] **Step 5: commit** `feat(m2): prereq DAG over CF tags + frontier + config`

---

### Task 2: Candidate pool + Assess (max Fisher info)

**Files:** Create `recommender/candidates.py`, `recommender/assess.py`; Test `tests/test_assess.py`.

**Interfaces:**
- `candidates.py`: `Candidate(pid, b, tags, solved_count)`; `load_unseen(conn, user_id, tags=None, b_range=None) -> list[Candidate]` — catalog problems with rating+tags the user has NO interaction with, optionally filtered to tags / a `b` window.
- `assess.py`: `assess_select(model, cands, topics, k) -> list[Candidate]` — for high-σ `topics`, pick the `k` unseen problems maximizing Fisher info about those topics' θ (i.e. `b` closest to `μ_t`, P≈0.5). Returns candidates with attached `(p, info, mode='assess')`.

- [ ] **Step 1: failing test** — Fisher info maximized when `b≈μ`; assess picks problems near μ of the targeted high-σ topic; never picks seen problems.

```python
# tests/test_assess.py (sketch — full code in build)
def test_assess_picks_problem_near_mu():
    # model: tag 'dp' mu=1700 sigma=300 (high); candidates at b=1200,1700,2400
    # assess_select must rank b=1700 first (max info / P~0.5)
    ...
```

- [ ] **Steps 2-5:** implement, test PASS, commit `feat(m2): candidate pool + max-Fisher Assess selection`.

---

### Task 3: Train policy (gap-weighted, frontier, stochastic, band)

**Files:** Create `recommender/train.py`; Test `tests/test_train.py`.

**Interfaces:**
- `goal_weights(conn, tags, r_band, band=200) -> dict[str,float]` — rating-mode: tag frequency in the catalog within `[r_band-band, r_band+band]`, normalized. (Interview-mode = fixed FAANG weights, config-switchable; M2 default rating-mode.)
- `topic_scores(model, open_tags, r_band, weights) -> dict[str,float]` — `gap_t · weight_t`, `gap_t=max(0, r_band−μ_t)`.
- `sample_topics(scores, k, rng) -> list[str]` — sample ∝ score WITHOUT replacement; returns topics **and** the per-draw probability (propensity) used.
- `pick_in_band(model, cands_for_topic, rng, band=TRAIN_TARGET_BAND) -> (Candidate, propensity)` — among unseen candidates with predicted P in band, sample (weight by quality=solved_count & diversity); return choice + its selection probability.

- [ ] **Steps:** failing tests (gap zero when μ≥r_band; sampling respects weights with a seeded RNG; only in-band problems chosen; propensity in (0,1]); implement; PASS; commit `feat(m2): gap-weighted stochastic Train topic+problem selection`.

> Stochastic (not argmax) so `propensity` is a real probability for OPE (docs/06).

---

### Task 4: Compose daily set + log recommendations

**Files:** Create `recommender/compose.py`; Test `tests/test_compose.py`.

**Interfaces:**
- `daily_set(conn, model, user_id, r_band, h_hours, rng) -> list[dict]` —
  1. `k = round(h_hours*60/MINUTES_PER_PROBLEM / 7)` problems/day (≥1).
  2. open = `frontier(...)`; goal-relevant open tags.
  3. high-σ open tags (`σ>ASSESS_SIGMA_THRESHOLD`) → Assess slots; rest → Train.
  4. Train slots split train-band vs stretch per `DAILY_BLEND`.
  5. **Interleave** so `max_consecutive_same_tag=1`.
  6. Each served item logged to `recommendations` (mode, predicted_p, predicted_info, propensity, served_at) and returned with a `why` rationale string.
- `insert_recommendation(conn, row) -> int`.

- [ ] **Steps:** failing tests (set size from H; no 2 consecutive same tag; every item has propensity logged; mode split tracks σ); implement; PASS; commit `feat(m2): daily-set blend + interleave + recommendations logging`.

---

### Task 5: Simulated policy comparison — the M2 gate (`eval/sim_m2.py`)

**Files:** Create `eval/sim_m2.py`; Test `tests/test_sim_m2.py`.

**Interfaces:**
- `simulate(policy, model0, cands, horizon, rng) -> dict` — clone the fitted M1 model as the learner; each step the policy serves a problem, outcome `y~Bernoulli(model_true.P)` (oracle = the fitted model), learner-model updates; track cumulative gap-reduction toward `R`, served-difficulty distribution, topic coverage.
- Policies: `greedy_two_mode`, `random_policy`, `fixed_curriculum` (CF rating ladder: ascending b), `difficulty_match` (band only, no coverage/interleave).
- `run_gate(conn, user_id) -> dict` — run all policies; **gate**: greedy beats random AND fixed-curriculum on skill-gain-per-item; **mode behavior** (B1): a high-σ topic routed to Assess reaches `σ<τ` in fewer items than random; **anti-Goodhart** (B4): greedy's served-difficulty stays in band and frontier coverage grows (doesn't collapse to easy churn).

- [ ] **Steps:** failing tests on a tiny synthetic model (greedy ≥ random on gain; anti-Goodhart flags a degenerate easy-only policy); implement; run `run_gate` on Vish2503; **report the gate table** (paste into PR). Commit `feat(m2): simulated policy comparison + M2 gate`.

> Honest caveat embedded in output: oracle == model family ⇒ suggestive, not validated (docs/07); real OPE via logged `recommendations` once deployed.

---

### Task 6: API + "Today's set" UI

**Files:** Create `api/__init__.py`, `api/app.py`; `web/src/DailySet.jsx`, route in `web/src/main.jsx`; add `fastapi`,`uvicorn` deps; `make serve`.

**Interfaces:**
- `GET /skills?user=` → per-tag μ/σ (radar). `GET /daily-set?user=` → today's set with `pid, b, tags, predicted_p, mode, why`.
- `DailySet.jsx` lists each problem with its CF link, predicted P(solve), mode badge, and the why-this-problem rationale.

- [ ] **Steps:** implement FastAPI app reading the M1 snapshot + `compose.daily_set`; minimal React list; `make serve` runs API+web; verify `GET /daily-set` returns a valid interleaved set; commit `feat(m2): FastAPI daily-set/skills endpoints + Today's set UI`.

---

## Self-Review
- Covers docs/08 M2 checklist: assess ✓(T2), train+compose+blend+interleave ✓(T3,T4), prereq_dag+frontier ✓(T1), propensity logging ✓(T4), Today's-set UI ✓(T6), gate (mode behavior + beats baselines + anti-Goodhart) ✓(T5).
- Three dependencies-on-later-milestones explicitly flagged above with reversible decisions.
- Stochastic policy ⇒ real propensity (not the deterministic-argmax trap that makes IPS degenerate).
- No placeholders in shipped tasks; T2/T3/T5 test bodies are sketched here and written in full during build (per executing-plans).
