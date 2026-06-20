# 01 — Product

## Problem

Self-directed CP practice is inefficient. People grind problems that are too easy (comfortable but low-growth), neglect weak topics, forget what they learned, and have no honest measure of "how much do I actually know." Whetstone replaces guesswork with an adaptive policy grounded in learning science.

## User

A single competitive programmer (you) with an existing Codeforces history and a concrete goal. Single-tenant by design — this simplifies the data and lets the model specialize.

## Objective (formal)

The user provides a goal triple:

- `R` — target rating (default 1900, Candidate Master)
- `D` — target date (default ≈ 6 months out)
- `H` — weekly study budget in hours (default 8)

The system maximizes expected progress toward `R` by `D` under the time budget:

```
maximize   Σ_t  E[ Δskill_relevant_to_goal ]      s.t.   weekly_minutes ≤ 60·H
```

"Relevant to goal" is a topic weighting (see modes). Skill is the latent θ vector over topics (`docs/03`).

## Two scoring modes (weighting only — the engine is the same)

- **Rating mode** — topics weighted by their frequency in rated rounds at the user's band.
- **Interview mode** — topics weighted toward FAANG-relevant patterns (binary search, DP, graphs, two-pointer, heaps, intervals, hashing).

## Two operating modes (the engine — see `docs/04`)

- **Assess** — when uncertain about the user's skill on relevant topics, select problems that most reduce that uncertainty (adaptive diagnosis). Fast, accurate placement and periodic recalibration.
- **Train** — when skill is well-estimated, select problems that maximize learning gain.

## Scope

**In (v1):** Codeforces ingestion; Bayesian per-topic IRT skill model; mastery + retention tracking; the two-mode recommender (greedy baseline → bandit); dashboard; eval suite.

**In (v2+):** DKT predictive model; offline-RL recommender + A/B; Claude tutor loop; LeetCode/AtCoder behind the `Interaction` interface; continuous difficulty inference for LeetCode.

**Out:** multi-user/social; mobile app; anything requiring CF auth or scraping ToS-protected data.

## Success criteria

Two independent bars — both must hold.

**Tool success (does it help you?)**
- You use it to choose problems ≥4 days/week for a month.
- Predicted rating tracks actual rating within a sensible error band over ≥3 contests.
- Subjectively: recommended problems feel "right at the edge," not trivial or hopeless.

**Rigor success (is it actually principled?)**
- Knowledge model is **calibrated** (ECE below the gate in `docs/07`).
- Next-step prediction beats a frequency baseline by a margin (AUC gate).
- Per-skill learning curves fit the power law of practice; estimated learning rates are stable.
- The recommender beats random / fixed-curriculum / pure-difficulty-match baselines under offline policy evaluation.

## Non-goals / explicit honesty

This is n=1. Causal efficacy claims ("this made me improve faster") cannot be proven without a controlled trial we will not run. We validate via offline/simulated evaluation and predicted-vs-actual outcomes, and we say so plainly.
