# 06 — Data Model

## Codeforces API contract

Public endpoints, no auth. Be polite: **≤1 request / 1.5s**, exponential backoff on 503/429, cache the catalog.

| Endpoint | Key fields | Use |
|---|---|---|
| `problemset.problems` | `problems[].{contestId,index,name,rating,tags}`, `problemStatistics[].solvedCount` | problem catalog + candidate pool. Refresh nightly. |
| `user.status?handle=&from=&count=` | `result[].{problem,verdict,creationTimeSeconds,programmingLanguage,author.participantType}` | the interaction stream. Paginate; track last-seen. |
| `user.rating?handle=` | `result[].{contestId,newRating,ratingUpdateTimeSeconds}` | predicted-vs-actual ground truth. |
| `user.info?handles=` | `rating,maxRating,rank` | cold-start prior + display. |
| `contest.list` | upcoming/finished rounds | contest-readiness feature. |

### Ingestion rules

- **Incremental:** persist the max `creationTimeSeconds` (or submission id) seen; on each poll fetch only newer submissions.
- **Verdict mapping:** `OK` → solve. `WRONG_ANSWER`/`TIME_LIMIT_EXCEEDED`/`RUNTIME_ERROR`/`MEMORY_LIMIT_EXCEEDED` → failed attempt (keep the specific verdict — TLE vs WA is pedagogically different). Compilation errors and skipped → ignore.
- **In-contest vs upsolve:** derive from `author.participantType` (`CONTESTANT`/`MANAGER` ≈ in-contest, `PRACTICE`/`VIRTUAL` ≈ upsolve). Store the flag; weight first in-contest attempts most.
- **First-attempt:** the model cares most about whether the *first* submission to a problem succeeded; collapse repeated submissions to a per-(user,problem) attempt record with `solved`, `n_attempts`, `first_verdict`, `solved_in_contest`.
- **Missing `rating`:** some problems lack a CF rating; fall back to estimating `b_p` from `solvedCount` (IRT on the population) or exclude from the candidate pool until estimated.

## Database schema (Postgres)

```sql
-- cached CF catalog
problems (
  id           text primary key,        -- "{contestId}{index}", e.g. "1850A"
  contest_id   int,  idx text,  name text,
  rating       int,                      -- CF difficulty b_p (nullable → estimate)
  tags         text[],
  solved_count int,
  source       text default 'codeforces',
  updated_at   timestamptz
);

-- one row per (user, problem) attempt episode
interactions (
  id              bigserial primary key,
  user_id         text,
  problem_id      text references problems(id),
  solved          bool,
  n_attempts      int,
  first_verdict   text,
  solved_in_contest bool,
  first_seen_at   timestamptz,
  solved_at       timestamptz
);

-- θ posterior snapshots (history kept for learning-curve fits)
topic_skill (
  user_id   text, tag text,
  mu        real, sigma real,            -- Gaussian posterior over θ_t
  mastered  bool,
  snapshot_at timestamptz,
  primary key (user_id, tag, snapshot_at)
);

-- FSRS retention state per concept/representative problem
reviews (
  user_id text, concept text,
  stability real, difficulty real, last_review timestamptz,
  due_at timestamptz,
  primary key (user_id, concept)
);

-- the recommender's log (needed for offline policy evaluation)
recommendations (
  id bigserial primary key,
  user_id text, problem_id text references problems(id),
  mode text,                             -- 'assess' | 'train'
  predicted_p real, predicted_info real, propensity real,
  served_at timestamptz,
  outcome_solved bool, outcome_at timestamptz
);

-- learned params (auto-tuned values that override config priors)
learned_params (
  user_id text, key text, value jsonb, fit_at timestamptz,
  primary key (user_id, key)
);
```

The `recommendations.propensity` column is **not optional** — offline policy evaluation (`docs/07`) needs the probability with which each action was taken.

## Cold-start data

For L3/DKT cold-start, optionally ingest a public Codeforces interaction dataset (read-only, offline) to pretrain before fine-tuning on the single user. Keep it out of the user's `interactions` table; load it only in `eval/`/training.
