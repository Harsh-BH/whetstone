# Whetstone M3 (retention + mastery + dashboard) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (inline, DB-coupled). Checkbox steps.

**Goal:** Turn Whetstone into a real daily tool: FSRS spaced-review scheduling, a precise mastery criterion that drives the recommender's active pool, and a dashboard (mastery-over-time, predicted-vs-actual rating, topic heatmap, contest-readiness).

**Architecture:** `retention/fsrs.py` (forgetting curve + stability update) + `retention/scheduler.py` (cold-start state from solve history, due-review queue). `model/mastery.py` (μ≥R_band ∧ σ≤sd_max ∧ sustained retention) writes `topic_skill.mastered` and feeds `recommender` (replacing M2's μ-proxy frontier). `eval/rating.py` ingests `user.rating`, predicts rating from θ, compares. Dashboard panels via `api/` + `web/`.

**Tech Stack:** Python 3.12, psycopg3, existing model/recommender/eval; React/recharts.

## Global Constraints (docs)

- FSRS over fixed SM-2; review trigger at `target_retrievability = 0.90` (docs/02 P3, `config.TARGET_RETRIEVABILITY`).
- Mastery = confident posterior at/above band **and** sustained retention across ≥ `MASTERY_SUSTAINED_REVIEWS=2` (docs/02 P5, docs/03).
- FSRS weights/personal optima are fit from the user's data **later**; M3 uses documented defaults as priors (docs/02 "the one rule").
- Honest n=1: predicted-vs-actual rating is the only real-world signal and is suggestive, not causal (docs/07 D).
- Gate (docs/07/08): review queue surfaces decaying topics; predicted rating tracks actual over ≥1 contest.

## Design decisions (noted; docs leave these open)

1. **Concept granularity = CF tag.** Retention tracked per tag; a "review" is solving any problem of that tag. (Per-problem retention is finer but sparse at n=1.)
2. **FSRS cold-start from history.** No in-system reviews exist yet, so replay each tag's *solved* timestamps as successful reviews through a **simplified FSRS** (power forgetting curve + stability growth that rewards reviewing at low retrievability). Full FSRS-5 17-weight fit is a later refinement (docs/02).
3. **θ→rating predictor = goal-weighted mean of μ_tag.** A defensible scalar from the per-tag θ-vector; compared to actual `user.rating`. Rough by construction (docs/03 multi-tag credit caveat) — reported with loud n=1 caveats.
4. **Go fork: NO.** Stay a Python monolith — the honest default for a personal tool (docs/05). Revisit only if it goes on the SDE CV.

## Repo additions

```
retention/__init__.py  retention/fsrs.py  retention/scheduler.py
model/mastery.py
eval/rating.py
db/migrations/versions/0002_ratings.py   # actual CF rating history (predicted-vs-actual)
api/app.py (+ endpoints)   web/src/Dashboard.jsx
```

---

### Task 1: FSRS forgetting curve + stability (`retention/fsrs.py`)

**Files:** Create `retention/__init__.py`, `retention/fsrs.py`; Test `tests/test_fsrs.py`.

**Interfaces:**
- `config`: `FSRS_DECAY=-0.5`, `FSRS_FACTOR=0.2346` (so R(S)=0.9), `FSRS_INIT_STABILITY=1.0`, `FSRS_INIT_DIFFICULTY=5.0`, `FSRS_GROWTH=2.0`, `FSRS_FORGET=0.5`, `MASTERY_SD_MAX=130.0`, `MASTERY_MIN_STABILITY=21.0`.
- `retrievability(stability_days, elapsed_days) -> float` = `(1 + FSRS_FACTOR*elapsed/S)**FSRS_DECAY`; R(elapsed=S)=0.90.
- `days_until_due(stability, target=TARGET_RETRIEVABILITY) -> float` — elapsed at which R hits target.
- `next_stability(S, success, elapsed) -> float` — success: `S*(1 + FSRS_GROWTH*(1-retrievability(S,elapsed)))` (bigger gain reviewing late); fail: `S*FSRS_FORGET` (≥ FSRS_INIT_STABILITY).

- [ ] Tests: R decreases in elapsed; R(S)=0.90; days_until_due(S)=S at target 0.90; success grows S, more when reviewed later; fail shrinks S. Implement; PASS; commit `feat(m3): FSRS forgetting curve + stability update`.

---

### Task 2: Retention scheduler + reviews persistence (`retention/scheduler.py`)

**Files:** Create `retention/scheduler.py`; Test `tests/test_scheduler.py` (DB).

**Interfaces:**
- `cold_start_state(conn, user_id) -> dict[str, dict]` — per tag, replay solved-problem timestamps (from `interactions`, ordered) through FSRS (each solve = success) → `{stability, difficulty, last_review, due_at}`.
- `save_reviews(conn, user_id, state) -> int` — upsert into `reviews`.
- `due_queue(conn, user_id, now) -> list[dict]` — tags with `retrievability(now) < TARGET_RETRIEVABILITY`, sorted ascending by R (most-decayed first), each with `{concept, retrievability, due_at}`.

- [ ] Tests: a long-unsolved tag has low R and appears in the queue; a just-solved tag does not; queue ordered by decay. Implement; PASS; commit `feat(m3): FSRS cold-start from history + due-review queue`.

---

### Task 3: Mastery criterion + active pool (`model/mastery.py`)

**Files:** Create `model/mastery.py`; modify `recommender/prereq_dag.py` (frontier uses mastery); Test `tests/test_mastery.py`.

**Interfaces:**
- `is_mastered(model, reviews, tag, r_band) -> bool` — `μ≥r_band ∧ σ≤MASTERY_SD_MAX ∧ stability≥MASTERY_MIN_STABILITY` (the sustained-retention proxy for ≥2 reviews).
- `mastered_set(model, reviews, tags, r_band) -> set[str]`.
- `mark_mastery(conn, user_id, model, reviews, r_band) -> int` — update `topic_skill.mastered`.
- `prereq_dag.frontier(..., mastered: set|None)` — a prereq is satisfied if **mastered** (M3) OR (fallback) μ≥r_band−margin. Active pool = open ∧ not mastered.

- [ ] Tests: mastery needs all three conditions (each missing → not mastered); frontier opens a child when its prereqs are in `mastered`. Implement; PASS; commit `feat(m3): mastery criterion + mastery-driven frontier`.

---

### Task 4: Rating ingest + predicted-vs-actual (`db 0002`, `eval/rating.py`)

**Files:** Create `db/migrations/versions/0002_ratings.py`, `eval/rating.py`; Test `tests/test_rating.py` (DB).

**Interfaces:**
- migration: `ratings(user_id text, contest_id int, new_rating int, update_time bigint, primary key(user_id, contest_id))`.
- `ingest_ratings(conn, client, handle) -> int` — `user.rating` → upsert `ratings`.
- `predict_rating(conn, model, r_band) -> float` — goal-weighted mean of μ_tag.
- `predicted_vs_actual(conn, model, user_id) -> dict` — latest actual rating vs predicted; abs error; gate `tracks = err ≤ 300`.

- [ ] Tests: predict_rating in a sane range for a known model; predicted_vs_actual returns error + bool on seeded ratings. Implement; ingest Vish2503 ratings; PASS; commit `feat(m3): rating history ingest + predicted-vs-actual`.

---

### Task 5: M3 gate (`eval/m3_gate.py`) + Makefile

**Files:** Create `eval/m3_gate.py`; modify `Makefile`.

**Interfaces:** `run_gate(conn, user_id) -> dict` — (a) due_queue surfaces ≥1 decaying topic and excludes freshly-solved; (b) predicted rating within 300 of actual for the latest contest. Print table; `pass = a ∧ b`.

- [ ] Run on Vish2503; report. Commit `feat(m3): M3 acceptance gate (retention + predicted-vs-actual)`.

---

### Task 6: Dashboard (API + React panels)

**Files:** modify `api/app.py` (+ `/reviews`, `/readiness`, `/rating-history`, `/mastery`); create `web/src/Dashboard.jsx`; wire into `main.jsx`.

**Interfaces:** endpoints return review queue, contest-readiness (mastered-coverage + gap-to-R → score, "solve these 5 → X→Y"), predicted-vs-actual series, per-tag mastery heatmap. React panels using recharts. Build clean (headless verify via curl + `npm run build`).

- [ ] Implement; verify endpoints + build; commit `feat(m3): retention/readiness/rating dashboard`.

---

## Self-Review
- docs/08 M3 checklist: FSRS+scheduler ✓(T1,T2), mastery+active-pool ✓(T3), dashboard (mastery-over-time/predicted-vs-actual/heatmap/readiness) ✓(T6), Go-fork decision ✓(noted: no). Gate ✓(T5).
- Closes M2's flagged frontier-without-mastery gap (T3 wires mastery into the frontier).
- Simplified FSRS + scalar rating predictor are documented approximations with full-fit refinements deferred (docs/02/03), not silent shortcuts.
