---
description: Pull new Codeforces submissions and refresh the problem catalog, normalizing into the interactions table.
---

Run the Codeforces ingestion pipeline for the configured handle.

Steps:
1. Refresh the problem catalog from `problemset.problems` if the cache is older than 24h (respect rate limits: ≤1 req / 1.5s, backoff on 503/429).
2. Pull new submissions from `user.status` incrementally — only those newer than the max `creationTimeSeconds` already stored.
3. Normalize submissions into per-(user, problem) attempt episodes: `solved`, `n_attempts`, `first_verdict`, `solved_in_contest` (from `author.participantType`).
4. Upsert into `problems` and `interactions` per `docs/06-data-model.md`.
5. Report: # new episodes, # new solves, per-tag and per-rating coverage deltas.

Do not online-update the model here — that is `make train` / the nightly refit. Just land clean data.
