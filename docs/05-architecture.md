# 05 — Architecture

## Components

```
┌────────────┐   poll    ┌──────────┐         ┌─────────────────────────────┐
│ Codeforces │ ────────► │ ingest/  │ ──────► │ Postgres                    │
│   API      │           │ poller   │ normalize│ problems · interactions ·   │
└────────────┘           └──────────┘         │ topic_skill · reviews · recs │
                                              └───────────────┬─────────────┘
                          ┌───────────────────────────────────┼───────────────────────┐
                          ▼                                   ▼                         ▼
                  ┌───────────────┐                  ┌──────────────────┐      ┌────────────────┐
                  │ model/        │                  │ recommender/     │      │ retention/     │
                  │ irt·pfa·dkt   │ predict_solve/   │ assess + train   │      │ FSRS scheduler │
                  │ mastery       │ info ◄────────── │ (greedy→bandit)  │ ◄─── │ due reviews    │
                  └───────┬───────┘                  └────────┬─────────┘      └────────────────┘
                          └───────────────┬───────────────────┘
                                          ▼
                                  ┌───────────────┐
                                  │ api/ (FastAPI)│
                                  └───────┬───────┘
                                          ▼
                                  ┌───────────────┐        ┌─────────────────────┐
                                  │ web/ React    │        │ (opt) Claude tutor  │
                                  │ recharts dash │ ◄────► │ hint ladder / dx    │
                                  └───────────────┘        └─────────────────────┘
```

## Stack

| Layer | Choice | Note |
|---|---|---|
| Ingest + API | Python 3.12, FastAPI, httpx | one service for MVP |
| Model | numpy/scipy/sklearn (IRT/PFA), PyTorch (DKT only) | keep torch out of the hot path until L3 |
| Store | Postgres + Alembic | single-tenant; partition `interactions` only if needed |
| Scheduler | APScheduler / cron loop | ingest + nightly refit + due-review recompute |
| Web | React + Vite + Tailwind + recharts | radar, heatmap, lines, daily set |
| Tutor (opt) | Anthropic API | prototype as a Claude artifact first |
| Infra | docker-compose | k8s/Terraform only if it's going on the SDE CV |

### The Go fork (optional, CV-driven)

For the SDE narrative, the **ingest + serving** layer can be a Go service (poller + Postgres + API) with Python kept as a model sidecar exposing `predict_solve`/`update` over gRPC. Decide at M3 — don't pre-build it. Python monolith is the honest default for a personal tool.

## Repo layout

```
whetstone/
├── ingest/         # cf_client.py, poller.py, normalize.py
├── model/          # irt.py, pfa.py, dkt.py, mastery.py, prereq_dag.py, config.py
├── recommender/    # assess.py, train.py, bandit.py, compose.py
├── retention/      # fsrs.py, scheduler.py
├── api/            # app.py, routes/, schemas.py
├── web/            # React dashboard
├── eval/           # metrics.py, calibration.py, learning_curves.py, ope.py, sim.py, report.py
├── infra/          # docker-compose.yml (+ optional terraform/)
├── tests/
├── config.py       # ALL pedagogical constants, each annotated with its docs/02 rule
└── docs/
```

## Data flow

1. **Ingest** polls CF (`user.status` incremental, catalog nightly), normalizes into `interactions` + `problems`.
2. **Model** refits μ/σ per topic (online on each interaction; full refit nightly), updates mastery.
3. **Retention** recomputes due reviews.
4. **Recommender** assembles the daily set on request (mode chosen by σ).
5. **API** serves dashboard data + daily set + the why-this-problem rationale.
6. **Eval** runs on demand (`/eval`) and in CI; gates block model/recommender regressions.

## Config discipline

`config.py` is the single home for every pedagogical constant (`train_target_band`, `target_retrievability`, `max_consecutive_same_tag`, mastery thresholds, blend ratios). Each constant carries a comment naming its `docs/02` principle. Auto-tuned values (personal optimal difficulty, FSRS weights) are read from a learned-params store, with the config value as the prior/fallback.
