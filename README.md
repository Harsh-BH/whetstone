# Whetstone

Adaptive Codeforces trainer. It learns your per-topic skill (with uncertainty), figures out where you're weakest relative to a target rating, and serves the problems that grow you fastest — using a principled instructional policy, not a static problem list.

## Why it's different

- **Two modes, two objectives.** An *assessment* mode pins down what you know fast (adaptive testing — maximize information about your latent skill); a *training* mode maximizes learning gain (desirable difficulty, spacing, interleaving).
- **Every knob is principled.** Difficulty targets, review intervals, and topic mixing each trace to a cited learning-science result (`docs/02-learning-science.md`). The optimal difficulty is *estimated from your own learning curves*, not guessed.
- **It proves it works.** Calibration, next-step prediction AUC, learning-curve fits, and offline policy evaluation against baselines (`docs/07-evaluation.md`).

## Quickstart

```bash
cp .env.example .env          # set CF_HANDLE, target R/D/H, DATABASE_URL
docker compose up -d db
make migrate
make ingest                   # pulls your CF history (be patient; rate-limited)
make train                    # fits the knowledge model
make serve                    # API + dashboard at localhost:3000
```

## Status

See `docs/08-milestones.md`. M0–M3 → a daily-use tool. M4–M5 → DKT + offline-RL recommender (the research/CV core).

## Docs

Read `CLAUDE.md`, then `docs/` in order. The specs are the source of truth.
