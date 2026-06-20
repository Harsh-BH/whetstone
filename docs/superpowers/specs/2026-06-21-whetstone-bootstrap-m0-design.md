# Whetstone — Bootstrap + M0 Design

**Date:** 2026-06-21
**Status:** approved (brainstorming → writing-plans)
**Scope of this engagement:** bootstrap the repo, then build **M0** (CF data flows end to end). Detailed plans for M1–M7 are produced per-milestone, just before each is built.

> This file records only the decisions the governing docs leave open. The source of truth is `docs/01`…`docs/08` + `CLAUDE.md`. Do not duplicate them here.

---

## Governing docs (source of truth)

- `docs/06-data-model.md` — CF API contract + DB schema (governs M0).
- `docs/08-milestones.md` — M0 task checklist + acceptance gate.
- `docs/05-architecture.md` — repo layout, stack, data flow.
- `CLAUDE.md` — conventions, guardrails, ways of working.

---

## Stack decisions (the forks the docs leave open)

| Decision | Pick | Rationale / alternative |
|---|---|---|
| Env + dependency manager | **uv**, Python **3.12** | `uv 0.11.11` + `python3.12.13` already installed; uv pins Python for free. Pinning 3.12 (vs the system's 3.14) avoids the PyTorch-on-3.14 risk at M4. Alt: pip+venv — more steps. |
| DB access | **psycopg3** + raw SQL, **no ORM** | Schema is fixed and simple (`docs/06`). Add SQLAlchemy only if the query layer hurts (≥ M1). `ponytail`: fewest moving parts. |
| Migrations | **Alembic**, hand-written migration (raw DDL via `op.execute`) | Required by `docs/05` conventions. No ORM metadata needed. |
| CF client | **httpx (sync)** + `pydantic` response models | A nightly+incremental poller doesn't need async. Alt: async — YAGNI. |
| Config / secrets | **`pydantic-settings`** + `.env` | `CF_HANDLE`, `DATABASE_URL`, goal triple `R/D/H`. `config.py` is the single home for pedagogical constants (`docs/05`), each annotated with its `docs/02` rule. |
| Postgres | **docker-compose** service `db` | Matches README quickstart; no local pg running. |
| Lint/format/test | **ruff + black + pytest** as uv dev-deps | `docs/05`/`CLAUDE.md` conventions. |

Goal triple defaults (until the user overrides): `R=1900`, `D≈2026-12-21` (~6 months), `H=8`.

---

## M0 — Spike: prove CF data flows end to end

**Objective (docs/08):** prove CF data flows end to end.
**Governing:** `docs/06`.

### Components
- **`ingest/cf_client.py`** — typed httpx wrappers for `problemset.problems`, `user.status` (paginated via `from`/`count`), `user.rating`, `user.info`. Rate-limit **≤1 req / 1.5s**, exponential backoff on **429/503**. Catalog cached (refresh if older than 24h).
- **`ingest/normalize.py`** — collapse raw submissions → one episode per `(user, problem)`:
  - `solved` (any `OK`), `n_attempts`, `first_verdict`, `solved_in_contest` (from `author.participantType`: `CONTESTANT`/`MANAGER` ≈ in-contest, `PRACTICE`/`VIRTUAL` ≈ upsolve).
  - Verdict mapping: `OK`→solve; `WRONG_ANSWER`/`TLE`/`RUNTIME_ERROR`/`MLE`→failed attempt (keep specific verdict); compilation-error & skipped → ignored.
  - Missing problem `rating` → flag for `b_p` estimation later (out of scope M0; keep the row).
  - **This is the only piece with real logic → it gets a `pytest` unit test** (the `ponytail` runnable check: a fixed list of fake submissions in, expected episodes out).
- **`ingest/poller.py`** — incremental: persist max `creationTimeSeconds` seen; fetch only newer. Upsert into `problems` + `interactions`.
- **DB** — one Alembic migration creating all 6 tables from `docs/06`: `problems`, `interactions`, `topic_skill`, `reviews`, `recommendations` (incl. non-optional `propensity`), `learned_params`.
- **Glue** — `config.py` (settings + pedagogical-constants stub), `infra/docker-compose.yml` (Postgres `db`), `pyproject.toml` (uv), `.env.example`, `Makefile` targets (`migrate`, `ingest`), `/ingest` command already in `.claude/commands/`.

### Data flow (docs/05, step 1)
CF API → `cf_client` → `normalize` → upsert `problems` + `interactions`. No model update here (that's `make train` / M1+).

### Acceptance gate (M0 → M1)
Episode counts and per-tag / per-rating distributions match the user's actual CF profile (sanity check against the CF web profile). No green-light to M1 until this holds.

### Out of scope for M0 (explicit YAGNI)
- Any knowledge model (`irt.py` etc.) — M1.
- `b_p` estimation for rating-less problems — keep the row, estimate later.
- Async client, ORM, Go fork, public cold-start dataset.

### Open input
- **CF handle** is required to *run* ingest; all M0 code is handle-agnostic and built without it. Prompt the user at the ingest step.

---

## Roadmap M1–M7 (one-liners; detailed plan precedes each build)

- **M1** — `irt.py` Rasch + Bayesian (μ,σ) update + Fisher info (`θ_eff=min`); cold-start from CF rating; eval temporal-split AUC + ECE; React skill radar. **Gate: ECE≤0.05 & AUC≥0.70 (+0.05 vs per-tag).**
- **M2** — `assess.py` (max-Fisher), `train.py`+`compose.py` (gap-weighted, frontier-restricted, interleaved, 65/20/15 blend), `prereq_dag.py`; log `mode`/`predicted_p`/`propensity`; "Today's set" UI. **Gate: mode-switch + in-band + beats random/fixed-curriculum on OPE + anti-Goodhart.**
- **M3** — `fsrs.py`+`scheduler.py`, mastery criterion, full dashboard; Go-fork decision. **Gate: review queue surfaces decay; predicted tracks actual ≥1 contest.**
- **M4** — `pfa.py` (learning rates + power-law curves), `dkt.py` (LSTM/SAKT, public-CF cold-start). **Gate: curves fit majority of skills; L1/L2/L3 table.**
- **M5** — `bandit.py` (LinUCB/Thompson), personal-optimal-difficulty estimator, offline RL; `ope.py`+`sim.py`. **Gate: learned policy beats greedy/random/fixed-curriculum on OPE + sim; report speedup.**
- **M6** — Claude tutor hint-ladder (retrieval-first), concept diagnosis → model feedback. **Gate: post-mortem produces correct concept tag.**
- **M7** — deploy + results write-up + demo. **Gate: every number regenerable via `/eval`.**

---

## Non-negotiables carried from `CLAUDE.md`/`docs/07`
- Any change to `model/` or `recommender/` runs `/eval`; a regressed gate is **stopped and reported**, never loosened.
- No hardcoded pedagogical truths — defaults are priors; the system estimates from the user's data where it can.
- Honest about n=1: validated by offline/simulated eval + predicted-vs-actual rating, not a controlled trial.
