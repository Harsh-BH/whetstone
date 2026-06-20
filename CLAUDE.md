# CLAUDE.md — Whetstone

> Read this first. Then read `docs/` in order before writing code. Work milestone by milestone (`docs/08-milestones.md`). The eval suite (`docs/07-evaluation.md`) must stay green after any model or recommender change.

## What this is

Whetstone is a single-user adaptive trainer for competitive programming. It ingests the user's **Codeforces** activity, infers a **per-topic latent skill** with quantified uncertainty, and serves the *optimal next problems* to maximize the rate of skill growth toward a target. CF-first; a unified `Interaction` abstraction lets LeetCode/AtCoder plug in later.

It is **not** a problem aggregator or a CRUD tracker. The point is a principled instructional policy. Every pedagogical parameter must trace to a cited principle in `docs/02-learning-science.md` — no vibes-based difficulty targets, no efficacy claims without the metrics in `docs/07-evaluation.md`.

## The objective

```
maximize  E[ skill growth toward target R ]   subject to   H hours/week
```

The user sets `(R, D, H)` = (target rating, target date, weekly budget). Default `R=1900` (Candidate Master), `D≈6mo`, `H=8`.

## The core idea: two modes

The recommender operates in two regimes with **different objectives** (see `docs/04-recommender.md`):

| Mode | Goal | Objective | Item choice |
|---|---|---|---|
| **Assess** (CAT) | Learn what the user knows, fast | minimize posterior variance of θ | maximize Fisher information ⇒ P(solve)≈0.5 |
| **Train** | Make the user better, fast | maximize expected learning gain | desirable-difficulty band, interleaved, spaced |

Mode is chosen by uncertainty: high posterior variance on the relevant skills → **Assess**; otherwise → **Train**. This split is the spine of the system. Do not collapse it.

## Architecture (see `docs/05-architecture.md`)

```
CF API → ingest → Postgres → { knowledge model (IRT→PFA→DKT), recommender (CAT/learning-gain) } → FastAPI → React/recharts dashboard → (optional) Claude tutor loop
```

## Repo map

```
whetstone/
├── CLAUDE.md                  # this file
├── README.md
├── docs/                      # specs — governing source of truth
│   ├── 01-product.md
│   ├── 02-learning-science.md # WHY every parameter exists (rigor)
│   ├── 03-knowledge-model.md  # the math (IRT/Fisher/PFA/DKT, mastery)
│   ├── 04-recommender.md      # the two-mode policy
│   ├── 05-architecture.md
│   ├── 06-data-model.md       # CF API contract + DB schema
│   ├── 07-evaluation.md       # how we prove it teaches (rigor)
│   └── 08-milestones.md       # the build plan + acceptance gates
├── .claude/commands/          # custom slash commands
│   ├── ingest.md
│   └── eval.md
├── ingest/   model/   api/   web/   infra/   tests/
```

## Conventions

- **Language:** Python 3.12, FastAPI, PyTorch (DKT only), numpy/scipy/sklearn (IRT/PFA). React + Tailwind + recharts for `web/`.
- **Style:** fully type-hinted, `ruff` + `black`, `pytest`. No function over ~50 lines without a reason.
- **Data:** Postgres + Alembic migrations. Never store raw API dumps as the source of truth — normalize into `interactions`.
- **CF etiquette:** ≤1 request / 1.5s, exponential backoff on 503/429, cache the problem catalog (refresh nightly). Public endpoints only — no auth, no scraping.
- **No LeetCode scraping in v1.** CF only behind the `Interaction` interface.
- **Reproducibility:** seed everything; every metric in a report is regenerable via `/eval`.

## Ways of working (for the agent)

1. Before any milestone, re-read its section in `docs/08-milestones.md` and the docs it references.
2. Use **plan mode** for any change touching >2 files; surface the plan before editing.
3. Keep commits PR-sized and milestone-scoped. One concern per commit.
4. **Any change to `model/` or `recommender/` must run `/eval` and keep all gates green.** If a gate regresses, stop and report — do not "fix" by loosening the gate.
5. Difficulty/spacing/interleaving constants live in one `config.py` and each is annotated with the `docs/02-learning-science.md` rule it implements. Don't scatter magic numbers.
6. When unsure whether a design choice is rigorous, the test is: *can I point to the principle and the metric that validates it?* If no → ask, don't guess.

## Commands

- `/ingest` — pull new CF submissions + refresh catalog, normalize into `interactions`.
- `/eval` — run the full evaluation suite, print the metrics table + calibration + learning-curve fit.
- `make serve` — run API + web.
- `make train` — refit knowledge model on current `interactions`.

## Non-negotiables (guardrails)

- Rigor over polish. A miscalibrated difficulty model teaches badly — calibration is a release gate, not a nice-to-have.
- The "optimal difficulty" is **estimated per-user from observed learning gains**, never hardcoded. The default band is a prior, not a truth.
- Honest about n=1: this is a personal-optimization system validated by offline/simulated evaluation + predicted-vs-actual rating, not a controlled trial. Don't overclaim in the README.
