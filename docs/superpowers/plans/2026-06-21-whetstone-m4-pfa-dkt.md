# Whetstone M4 (PFA/AFM + DKT) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans (inline). Checkbox steps.

**Goal:** Predictive depth + learning-rate modeling: PFA/AFM (per-skill learning rates + power-law learning curves), a DKT sequence model, and an honest L1-vs-L2-vs-L3 next-step AUC comparison.

**Architecture:** `model/pfa.py` builds opportunity-count features in temporal order and fits a logistic AFM (per-skill β/γ/ρ). `eval/learning_curves.py` fits the power law of practice per skill. `model/dkt.py` is a small PyTorch LSTM over the interaction stream. `eval/compare_models.py` produces the L1/L2/L3 table on the same temporal split.

**Tech Stack:** numpy/scipy/sklearn (PFA, curves), **PyTorch (DKT only)** — CPU wheel to keep install light; the Python 3.12 pin makes torch 2.12 available.

## Global Constraints (docs)
- PFA logit = `Σ_k [β_k + γ_k·#prior_correct_k + ρ_k·#prior_incorrect_k] + δ·b` (docs/03 L2; Pavlik 2009 / Cen-Koedinger AFM).
- Learning curves validate the **power law of practice** (Newell & Rosenbloom): error decreases with opportunities (docs/07 A3).
- Temporal split only (docs/07 A1). First-attempt outcome `y` as in M1.
- **Honest L3 expectation:** DKT may NOT beat L1/L2 at n=1 — "IRT wins at n=1" is a legitimate reported result, not a failure (docs/03, docs/07 A1).
- Gate (docs/07/08): learning curves fit on the **majority of active skills**; L1/L2/L3 comparison table produced (L2 ≥ L1; L3 reported either way).

## Design decisions (noted)
1. **DKT scope:** train on the user's own sequence (temporal split), CPU torch, small LSTM. Public-CF-dataset cold-start (docs/03) is a documented refinement, **not** built now (data-engineering heavy; expected to matter only modestly at n=1).
2. **PFA features:** per-tag one-hot β + (tag × prior_correct) + (tag × prior_incorrect) + a single difficulty coefficient on `b`. sklearn LogisticRegression (L2-reg). γ_k (coef on prior_correct) = the skill's learning rate.
3. **Curves on active skills:** skills with ≥ `MIN_OPPORTUNITIES=5` first-attempts; "fits" = power-law exponent indicates non-increasing error.

## Repo additions
```
model/pfa.py   eval/learning_curves.py   model/dkt.py   eval/compare_models.py
```

---

### Task 1: PFA/AFM model (`model/pfa.py`)

**Files:** Create `model/pfa.py`; Test `tests/test_pfa.py`.

**Interfaces:**
- `build_features(records) -> (X, y, feature_names)` — temporal pass tracking per-tag prior correct/incorrect *before* each record; one-hot tags (β), tag·prior_correct (γ), tag·prior_incorrect (ρ), and `b`.
- `PFAModel.fit(records)` / `.predict(records) -> list[float]` — sklearn LogisticRegression under the hood; `.learning_rates() -> dict[str,float]` (γ per skill).

- [ ] Tests: features count prior opportunities correctly (a tag's 3rd attempt has prior_correct+prior_incorrect=2); a synthetic learner whose success rate rises with practice yields **positive** learning rate γ for that skill; fit/predict returns probabilities in (0,1). Implement; PASS; commit `feat(m4): PFA/AFM model with per-skill learning rates`.

---

### Task 2: Learning curves + power-law fit (`eval/learning_curves.py`)

**Files:** Create `eval/learning_curves.py`; Test `tests/test_learning_curves.py`.

**Interfaces:**
- `opportunity_error(records, tag) -> list[(n, error)]` — mean first-attempt error at the k-th opportunity of `tag`.
- `fit_power_law(points) -> dict` — fit `error ≈ a·n^(-c)`; return `{a, c, ok: c≥0}` (error non-increasing).
- `curve_report(records, min_opps=MIN_OPPORTUNITIES) -> dict` — per active skill fit; `fraction_fit` = share with `ok`.

- [ ] Tests: a decaying synthetic error series fits `c>0` (ok); a flat/noisy series → not ok; `curve_report` returns fraction in [0,1]. Implement; run on Vish2503; PASS; commit `feat(m4): learning-curve power-law fits per skill`.

---

### Task 3: DKT sequence model (`model/dkt.py`)

**Files:** Create `model/dkt.py`; add `torch` (CPU) dep; Test `tests/test_dkt.py`.

**Interfaces:**
- `DKT` (torch LSTM): input per step = tag multi-hot ⊕ b-bucket ⊕ prev-correct; output P(solve next).
- `train_dkt(train_records, epochs, seed) -> DKT` (deterministic seed); `predict(model, records) -> list[float]`.

- [ ] Tests (tiny, seeded, CPU): trains without error; on a separable synthetic stream test AUC > 0.6; deterministic given seed. Implement; PASS; commit `feat(m4): DKT LSTM sequence model (PyTorch, CPU)`.
- Fallback: if torch install is infeasible here, mark DKT skipped and proceed with L1/L2 comparison (gate allows "L3 reported either way").

---

### Task 4: L1/L2/L3 comparison + M4 gate (`eval/compare_models.py`)

**Files:** Create `eval/compare_models.py`; Makefile `m4` target.

**Interfaces:** `compare(conn, user_id) -> dict` — same temporal split; test AUC/logloss for L1 (IRT, from `eval.run_m1`), L2 (PFA), L3 (DKT if available). `run_gate` = (a) `curve_report.fraction_fit ≥ 0.5`; (b) comparison table produced and `L2_auc ≥ L1_auc − 0.01` (L2 not worse). Print table; honest L3 line.

- [ ] Run on Vish2503; report table + gate. Commit `feat(m4): L1/L2/L3 comparison + M4 gate`.

---

## Self-Review
- docs/08 M4: PFA + learning curves ✓(T1,T2), DKT ✓(T3), L1/L2/L3 honest comparison ✓(T4). Gate = curves fit majority + table produced.
- DKT public-cold-start deferred (documented); L3 allowed to lose at n=1 (docs).
- torch is the only new heavy dep and is isolated to L3 (kept out of the L1/L2 hot path, docs/05).
