# Whetstone M0 (CF Ingest) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove Codeforces data flows end to end — pull a user's full submission history + the problem catalog, normalize into per-(user,problem) episodes, and land them in Postgres.

**Architecture:** A single `ingest/` package: a rate-limited httpx CF client (`cf_client.py`) → a pure normalize+merge layer (`normalize.py`) → psycopg3 upserts (`db.py`), orchestrated by `poller.py`. Schema is created by one Alembic migration. Ingest is **incremental + merge**: a per-user cursor (max `creationTimeSeconds` seen) means each run fetches only newer submissions, and because CF submissions are chronological, merging a stored episode with new submissions is correct.

**Tech Stack:** Python 3.12, `uv`, httpx (sync), pydantic + pydantic-settings, psycopg3, Alembic, Postgres 18 (docker-compose), pytest, ruff, black.

## Global Constraints

- **Python 3.12** only (pinned via `.python-version`; the system's 3.14 is not used) — `CLAUDE.md` / spec.
- **CF etiquette:** ≤ 1 request / 1.5 s; exponential backoff on 429/503; public endpoints only, no auth/scraping — `docs/06`, `CLAUDE.md`.
- **Source of truth is `interactions`**, never raw API dumps — `CLAUDE.md`.
- **Fully type-hinted**, `ruff` + `black`, `pytest`; no function > ~50 lines without reason — `CLAUDE.md`.
- **`config.py` is the single home** for every pedagogical constant, each annotated with its `docs/02` rule — `docs/05`.
- **`recommendations.propensity` is NOT optional** in the schema — `docs/06` (created here, used M2).
- Verdict mapping (`docs/06`): `OK`→solve; `WRONG_ANSWER`/`TIME_LIMIT_EXCEEDED`/`RUNTIME_ERROR`/`MEMORY_LIMIT_EXCEEDED`→failed attempt (keep specific verdict); `COMPILATION_ERROR`/`SKIPPED`/`TESTING`/anything else → ignored.
- In-contest detection (`docs/06`): `author.participantType` ∈ {`CONTESTANT`,`MANAGER`} → in-contest; {`PRACTICE`,`VIRTUAL`} → upsolve.

### Two deliberate deviations from `docs/06` (honoring its intent)
1. `interactions` gets `UNIQUE (user_id, problem_id)` — required for idempotent upsert of the "one row per (user,problem)" episode the doc describes.
2. A small operational table `ingest_state (user_id, last_creation_time)` holds the incremental cursor. Catalog staleness is derived from `max(problems.updated_at)`, so no extra column is needed.

### Branch
All M0 work on branch `m0-ingest` (cut from `main` at the genesis commit). `docs/08`: one milestone per branch.

```bash
git checkout -b m0-ingest
```

---

### Task 1: Project scaffold + tooling

**Files:**
- Create: `pyproject.toml`
- Create: `.python-version`
- Create: `.gitignore`
- Create: `.env.example`

**Interfaces:**
- Produces: a `uv`-managed env with httpx, pydantic, pydantic-settings, psycopg3, alembic (runtime) and pytest, ruff, black (dev). Consumed by every later task via `uv run ...`.

- [ ] **Step 1: Create `.python-version`**

```
3.12
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[project]
name = "whetstone"
version = "0.1.0"
description = "Adaptive Codeforces trainer"
requires-python = ">=3.12,<3.13"
dependencies = [
    "httpx>=0.27",
    "pydantic>=2.7",
    "pydantic-settings>=2.3",
    "psycopg[binary]>=3.2",
    "alembic>=1.13",
]

[dependency-groups]
dev = ["pytest>=8", "ruff>=0.5", "black>=24"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.black]
line-length = 100

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
```

- [ ] **Step 3: Create `.gitignore`**

```gitignore
__pycache__/
*.py[cod]
.venv/
.env
.pytest_cache/
.ruff_cache/
```

- [ ] **Step 4: Create `.env.example`**

```dotenv
# Codeforces handle to ingest (required to RUN ingest)
CF_HANDLE=

# psycopg-native libpq URL (alembic rewrites it to postgresql+psycopg:// internally)
DATABASE_URL=postgresql://whetstone:whetstone@localhost:5432/whetstone

# Goal triple (docs/01); defaults R=1900 / D~6mo / H=8
TARGET_RATING=1900
TARGET_DATE=2026-12-21
WEEKLY_HOURS=8
```

- [ ] **Step 5: Sync and verify the environment**

Run: `uv sync`
Then: `uv run python -c "import httpx, pydantic, pydantic_settings, psycopg, alembic; print('ok')"`
Expected: prints `ok` (and `uv sync` created `.venv` + `uv.lock`).

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .python-version .gitignore .env.example uv.lock
git commit -m "chore(m0): uv project scaffold + tooling"
```

---

### Task 2: Settings + config

**Files:**
- Create: `config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `config.settings` (a `Settings` instance) with `.cf_handle: str`, `.database_url: str`, `.target_rating: int`, `.target_date: str`, `.weekly_hours: int`. Pedagogical constants (module-level UPPER_CASE) for M1+.
- Consumed by: `ingest/db.py`, `ingest/poller.py`, `db/migrations/env.py`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
import importlib


def test_defaults_when_env_absent(monkeypatch):
    for k in ("CF_HANDLE", "DATABASE_URL", "TARGET_RATING", "TARGET_DATE", "WEEKLY_HOURS"):
        monkeypatch.delenv(k, raising=False)
    import config
    importlib.reload(config)
    s = config.Settings(_env_file=None)
    assert s.target_rating == 1900
    assert s.weekly_hours == 8
    assert s.target_date == "2026-12-21"


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("CF_HANDLE", "tourist")
    monkeypatch.setenv("TARGET_RATING", "2400")
    import config
    importlib.reload(config)
    s = config.Settings(_env_file=None)
    assert s.cf_handle == "tourist"
    assert s.target_rating == 2400


def test_pedagogical_constants_present():
    import config
    importlib.reload(config)
    assert config.MAX_CONSECUTIVE_SAME_TAG == 1
    assert config.TRAIN_TARGET_BAND == (0.55, 0.80)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'config'`.

- [ ] **Step 3: Write minimal implementation**

```python
# config.py
"""Single home for runtime settings and pedagogical constants (docs/05).

Every pedagogical constant traces to a rule in docs/02-learning-science.md.
These are PRIORS — the system estimates better values from the user's data
where it can (docs/02 "the one rule above all"). Used from M1+.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    cf_handle: str = ""
    database_url: str = "postgresql://whetstone:whetstone@localhost:5432/whetstone"
    target_rating: int = 1900        # R (docs/01)
    target_date: str = "2026-12-21"  # D (docs/01, ~6 months)
    weekly_hours: int = 8            # H (docs/01)


settings = Settings()

# --- Pedagogical constants (docs/02). Priors; auto-tuned values override later. ---
MIN_GENERATION_ATTEMPTS_BEFORE_HINT = 1   # P1 retrieval practice
ASSESS_TARGET_P = 0.5                      # P2/P8 max Fisher information
TRAIN_TARGET_BAND = (0.55, 0.80)          # P2 desirable difficulty (auto-tuned)
TARGET_RETRIEVABILITY = 0.90              # P3 spacing: review trigger
MAX_CONSECUTIVE_SAME_TAG = 1              # P4 interleaving
MASTERY_SUSTAINED_REVIEWS = 2            # P5 mastery
FRONTIER_ONLY = True                     # P6 prereq-DAG frontier
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_config.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add config.py tests/test_config.py
git commit -m "feat(m0): settings + pedagogical constants in config.py"
```

---

### Task 3: Postgres (compose) + Alembic schema migration

**Files:**
- Create: `infra/docker-compose.yml`
- Create: `alembic.ini`
- Create: `db/migrations/env.py`
- Create: `db/migrations/script.py.mako`
- Create: `db/migrations/versions/0001_initial_schema.py`
- Test: `tests/test_schema.py`

**Interfaces:**
- Produces: 7 tables in Postgres — `problems`, `interactions` (with `UNIQUE(user_id, problem_id)`), `topic_skill`, `reviews`, `recommendations` (incl. `propensity`), `learned_params`, `ingest_state`.
- Consumed by: `ingest/db.py`.

> **Setup note:** start the DB once for this task and Tasks 4 & 7:
> `docker compose -f infra/docker-compose.yml up -d db` and wait for healthy (`docker compose -f infra/docker-compose.yml ps`).

- [ ] **Step 1: Create `infra/docker-compose.yml`**

```yaml
services:
  db:
    image: postgres:18
    environment:
      POSTGRES_USER: whetstone
      POSTGRES_PASSWORD: whetstone
      POSTGRES_DB: whetstone
    ports:
      - "5432:5432"
    volumes:
      - whetstone_pg:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U whetstone"]
      interval: 5s
      timeout: 3s
      retries: 10

volumes:
  whetstone_pg:
```

- [ ] **Step 2: Create `alembic.ini`** (root; migrations live in `db/migrations`)

```ini
[alembic]
script_location = db/migrations
prepend_sys_path = .

[loggers]
keys = root

[handlers]
keys = console

[formatters]
keys = generic

[logger_root]
level = WARN
handlers = console

[handler_console]
class = StreamHandler
args = (sys.stderr,)
level = NOTSET
formatter = generic

[formatter_generic]
format = %(levelname)-5.5s [%(name)s] %(message)s
```

- [ ] **Step 3: Create `db/migrations/script.py.mako`**

```mako
"""${message}

Revision ID: ${up_revision}
Revises: ${down_revision | comma,n}
"""
from alembic import op
import sqlalchemy as sa

revision = ${repr(up_revision)}
down_revision = ${repr(down_revision)}
branch_labels = ${repr(branch_labels)}
depends_on = ${repr(depends_on)}


def upgrade() -> None:
    ${upgrades if upgrades else "pass"}


def downgrade() -> None:
    ${downgrades if downgrades else "pass"}
```

- [ ] **Step 4: Create `db/migrations/env.py`** (raw-SQL migrations; reads `DATABASE_URL`, rewrites to the psycopg3 driver for SQLAlchemy)

```python
from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from config import settings

config = context.config
# psycopg.connect() wants postgresql://; SQLAlchemy needs the explicit driver.
config.set_main_option("sqlalchemy.url", settings.database_url.replace("postgresql://", "postgresql+psycopg://", 1))
if config.config_file_name:
    fileConfig(config.config_file_name)

target_metadata = None  # raw-SQL migrations; no ORM models (ponytail)


def run_migrations_offline() -> None:
    context.configure(url=config.get_main_option("sqlalchemy.url"), literal_binds=True)
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section), prefix="sqlalchemy.", poolclass=pool.NullPool
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
```

- [ ] **Step 5: Create the migration `db/migrations/versions/0001_initial_schema.py`**

```python
"""initial schema (docs/06)

Revision ID: 0001
Revises:
"""
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE problems (
            id           text PRIMARY KEY,
            contest_id   int,
            idx          text,
            name         text,
            rating       int,
            tags         text[],
            solved_count int,
            source       text DEFAULT 'codeforces',
            updated_at   timestamptz
        );
        CREATE TABLE interactions (
            id                bigserial PRIMARY KEY,
            user_id           text,
            problem_id        text REFERENCES problems(id),
            solved            bool,
            n_attempts        int,
            first_verdict     text,
            solved_in_contest bool,
            first_seen_at     timestamptz,
            solved_at         timestamptz,
            CONSTRAINT interactions_user_problem_uniq UNIQUE (user_id, problem_id)
        );
        CREATE TABLE topic_skill (
            user_id     text,
            tag         text,
            mu          real,
            sigma       real,
            mastered    bool,
            snapshot_at timestamptz,
            PRIMARY KEY (user_id, tag, snapshot_at)
        );
        CREATE TABLE reviews (
            user_id     text,
            concept     text,
            stability   real,
            difficulty  real,
            last_review timestamptz,
            due_at      timestamptz,
            PRIMARY KEY (user_id, concept)
        );
        CREATE TABLE recommendations (
            id             bigserial PRIMARY KEY,
            user_id        text,
            problem_id     text REFERENCES problems(id),
            mode           text,
            predicted_p    real,
            predicted_info real,
            propensity     real,
            served_at      timestamptz,
            outcome_solved bool,
            outcome_at     timestamptz
        );
        CREATE TABLE learned_params (
            user_id text,
            key     text,
            value   jsonb,
            fit_at  timestamptz,
            PRIMARY KEY (user_id, key)
        );
        CREATE TABLE ingest_state (
            user_id            text PRIMARY KEY,
            last_creation_time bigint NOT NULL DEFAULT 0
        );
        """
    )


def downgrade() -> None:
    op.execute(
        """
        DROP TABLE IF EXISTS ingest_state, learned_params, recommendations,
            reviews, topic_skill, interactions, problems CASCADE;
        """
    )
```

- [ ] **Step 6: Run the migration against the running DB**

Run: `docker compose -f infra/docker-compose.yml up -d db` (if not already), then `uv run alembic upgrade head`
Expected: `Running upgrade  -> 0001, initial schema (docs/06)`.

- [ ] **Step 7: Write the schema test**

```python
# tests/test_schema.py
import os
import psycopg
import pytest

DB = os.environ.get("DATABASE_URL", "postgresql://whetstone:whetstone@localhost:5432/whetstone")

EXPECTED = {
    "problems", "interactions", "topic_skill", "reviews",
    "recommendations", "learned_params", "ingest_state",
}


def _db_up() -> bool:
    try:
        psycopg.connect(DB, connect_timeout=2).close()
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _db_up(), reason="Postgres not running (docker compose up -d db)")
def test_all_tables_exist():
    with psycopg.connect(DB) as conn:
        rows = conn.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
        ).fetchall()
    names = {r[0] for r in rows}
    assert EXPECTED <= names


@pytest.mark.skipif(not _db_up(), reason="Postgres not running")
def test_propensity_and_unique_constraint():
    with psycopg.connect(DB) as conn:
        cols = conn.execute(
            "SELECT column_name FROM information_schema.columns WHERE table_name='recommendations'"
        ).fetchall()
        assert "propensity" in {c[0] for c in cols}
        uniq = conn.execute(
            "SELECT conname FROM pg_constraint WHERE conname='interactions_user_problem_uniq'"
        ).fetchall()
        assert uniq, "missing UNIQUE(user_id, problem_id) on interactions"
```

- [ ] **Step 8: Run the schema test**

Run: `uv run pytest tests/test_schema.py -v`
Expected: PASS (2 passed) — assuming the DB is up; otherwise SKIPPED (acceptable in CI without DB, but must PASS locally before the M0 gate).

- [ ] **Step 9: Commit**

```bash
git add infra/docker-compose.yml alembic.ini db/migrations tests/test_schema.py
git commit -m "feat(m0): postgres compose + alembic initial schema (docs/06)"
```

---

### Task 4: DB helpers (psycopg3)

**Files:**
- Create: `ingest/__init__.py` (empty)
- Create: `ingest/db.py`
- Test: `tests/test_db.py`

**Interfaces:**
- Consumes: `config.settings.database_url`; `Episode` type from Task 6 is NOT needed here (db.py takes plain dicts/rows to stay decoupled).
- Produces:
  - `connect() -> psycopg.Connection`
  - `upsert_problems(conn, rows: list[dict]) -> int` — rows have keys `id, contest_id, idx, name, rating, tags, solved_count`. Sets `updated_at=now()`. ON CONFLICT(id) DO UPDATE.
  - `upsert_interaction(conn, ep: dict) -> None` — `ep` keys: `user_id, problem_id, solved, n_attempts, first_verdict, solved_in_contest, first_seen_at, solved_at` (timestamps are `datetime`). ON CONFLICT(user_id, problem_id) DO UPDATE.
  - `get_interaction(conn, user_id: str, problem_id: str) -> dict | None`
  - `get_cursor(conn, user_id: str) -> int` — 0 if none.
  - `set_cursor(conn, user_id: str, last_creation_time: int) -> None` — upsert into `ingest_state`.
  - `catalog_age_seconds(conn) -> float | None` — `now() - max(problems.updated_at)`; `None` if empty.

- [ ] **Step 1: Write the failing test** (integration; skips if DB down)

```python
# tests/test_db.py
import os
from datetime import datetime, timezone
import psycopg
import pytest
from ingest import db

DB = os.environ.get("DATABASE_URL", "postgresql://whetstone:whetstone@localhost:5432/whetstone")


def _db_up() -> bool:
    try:
        psycopg.connect(DB, connect_timeout=2).close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _db_up(), reason="Postgres not running")


@pytest.fixture()
def conn():
    c = db.connect()
    # clean slate for the test user/problem
    c.execute("DELETE FROM interactions WHERE user_id='_test'")
    c.execute("DELETE FROM problems WHERE id='9999Z'")
    c.execute("DELETE FROM ingest_state WHERE user_id='_test'")
    c.commit()
    yield c
    c.close()


def test_upsert_problem_is_idempotent(conn):
    row = dict(id="9999Z", contest_id=9999, idx="Z", name="t", rating=800,
               tags=["math"], solved_count=10)
    assert db.upsert_problems(conn, [row]) == 1
    row["solved_count"] = 20
    db.upsert_problems(conn, [row])
    conn.commit()
    got = conn.execute("SELECT solved_count FROM problems WHERE id='9999Z'").fetchone()
    assert got[0] == 20


def test_upsert_interaction_and_get(conn):
    db.upsert_problems(conn, [dict(id="9999Z", contest_id=9999, idx="Z", name="t",
                                   rating=800, tags=["math"], solved_count=10)])
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    ep = dict(user_id="_test", problem_id="9999Z", solved=True, n_attempts=2,
              first_verdict="WRONG_ANSWER", solved_in_contest=False,
              first_seen_at=ts, solved_at=ts)
    db.upsert_interaction(conn, ep)
    conn.commit()
    got = db.get_interaction(conn, "_test", "9999Z")
    assert got["solved"] is True and got["n_attempts"] == 2


def test_cursor_roundtrip(conn):
    assert db.get_cursor(conn, "_test") == 0
    db.set_cursor(conn, "_test", 1700000000)
    conn.commit()
    assert db.get_cursor(conn, "_test") == 1700000000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_db.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ingest'` (or `AttributeError` on `db`).

- [ ] **Step 3: Write minimal implementation**

```python
# ingest/__init__.py
```

```python
# ingest/db.py
"""Postgres access for ingest (psycopg3, raw SQL — no ORM, ponytail)."""
from datetime import datetime, timezone

import psycopg
from psycopg.rows import dict_row

from config import settings


def connect() -> psycopg.Connection:
    return psycopg.connect(settings.database_url, row_factory=dict_row)


def upsert_problems(conn: psycopg.Connection, rows: list[dict]) -> int:
    sql = """
        INSERT INTO problems (id, contest_id, idx, name, rating, tags, solved_count, updated_at)
        VALUES (%(id)s, %(contest_id)s, %(idx)s, %(name)s, %(rating)s, %(tags)s,
                %(solved_count)s, now())
        ON CONFLICT (id) DO UPDATE SET
            contest_id=EXCLUDED.contest_id, idx=EXCLUDED.idx, name=EXCLUDED.name,
            rating=EXCLUDED.rating, tags=EXCLUDED.tags,
            solved_count=EXCLUDED.solved_count, updated_at=now()
    """
    with conn.cursor() as cur:
        cur.executemany(sql, rows)
    return len(rows)


def upsert_interaction(conn: psycopg.Connection, ep: dict) -> None:
    sql = """
        INSERT INTO interactions
            (user_id, problem_id, solved, n_attempts, first_verdict,
             solved_in_contest, first_seen_at, solved_at)
        VALUES (%(user_id)s, %(problem_id)s, %(solved)s, %(n_attempts)s, %(first_verdict)s,
                %(solved_in_contest)s, %(first_seen_at)s, %(solved_at)s)
        ON CONFLICT (user_id, problem_id) DO UPDATE SET
            solved=EXCLUDED.solved, n_attempts=EXCLUDED.n_attempts,
            first_verdict=EXCLUDED.first_verdict,
            solved_in_contest=EXCLUDED.solved_in_contest,
            first_seen_at=EXCLUDED.first_seen_at, solved_at=EXCLUDED.solved_at
    """
    conn.execute(sql, ep)


def get_interaction(conn: psycopg.Connection, user_id: str, problem_id: str) -> dict | None:
    return conn.execute(
        "SELECT * FROM interactions WHERE user_id=%s AND problem_id=%s",
        (user_id, problem_id),
    ).fetchone()


def get_cursor(conn: psycopg.Connection, user_id: str) -> int:
    row = conn.execute(
        "SELECT last_creation_time FROM ingest_state WHERE user_id=%s", (user_id,)
    ).fetchone()
    return int(row["last_creation_time"]) if row else 0


def set_cursor(conn: psycopg.Connection, user_id: str, last_creation_time: int) -> None:
    conn.execute(
        """
        INSERT INTO ingest_state (user_id, last_creation_time) VALUES (%s, %s)
        ON CONFLICT (user_id) DO UPDATE SET last_creation_time=EXCLUDED.last_creation_time
        """,
        (user_id, last_creation_time),
    )


def catalog_age_seconds(conn: psycopg.Connection) -> float | None:
    row = conn.execute("SELECT max(updated_at) AS m FROM problems").fetchone()
    if not row or row["m"] is None:
        return None
    return (datetime.now(timezone.utc) - row["m"]).total_seconds()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_db.py -v`
Expected: PASS (3 passed) with the DB up.

- [ ] **Step 5: Commit**

```bash
git add ingest/__init__.py ingest/db.py tests/test_db.py
git commit -m "feat(m0): psycopg3 db helpers (upserts, cursor, catalog age)"
```

---

### Task 5: Codeforces API client

**Files:**
- Create: `ingest/cf_client.py`
- Test: `tests/test_cf_client.py`

**Interfaces:**
- Produces (pydantic models + a client):
  - `CFProblem(contest_id: int | None, index: str, name: str, rating: int | None, tags: list[str])` with alias `contestId`→`contest_id`; property `pid -> str` = `f"{contest_id}{index}"`.
  - `CFSubmission(id: int, creation_time: int, problem: CFProblem, verdict: str | None, participant_type: str)` (aliases `creationTimeSeconds`, and `author.participantType` flattened).
  - `CFRatingChange(contest_id: int, new_rating: int, update_time: int)`.
  - `CFUserInfo(handle: str, rating: int | None, max_rating: int | None, rank: str | None)`.
  - `CFClient(transport=None, min_interval=1.5, max_retries=4)` with methods:
    - `problemset_problems() -> list[CFProblem]` (merges `solvedCount` from `problemStatistics`; exposed via parallel `solved_counts: dict[str,int]` return — see signature below).
    - `user_status(handle: str) -> list[CFSubmission]` (auto-paginates all).
    - `user_rating(handle: str) -> list[CFRatingChange]`.
    - `user_info(handle: str) -> CFUserInfo`.
  - To keep `solvedCount` with the catalog, `problemset_problems()` returns `tuple[list[CFProblem], dict[str, int]]` (problems, pid→solved_count).
- Consumed by: `ingest/poller.py` (Task 7).

- [ ] **Step 1: Write the failing test** (httpx MockTransport — no network)

```python
# tests/test_cf_client.py
import json
import httpx
import pytest
from ingest import cf_client


def _resp(payload, status=200):
    return httpx.Response(status, content=json.dumps(payload))


def test_problemset_parsing():
    def handler(request):
        return _resp({"status": "OK", "result": {
            "problems": [{"contestId": 1850, "index": "A", "name": "P", "rating": 800,
                          "tags": ["math"]}],
            "problemStatistics": [{"contestId": 1850, "index": "A", "solvedCount": 1234}],
        }})

    c = cf_client.CFClient(transport=httpx.MockTransport(handler), min_interval=0)
    problems, solved = c.problemset_problems()
    assert problems[0].pid == "1850A"
    assert solved["1850A"] == 1234


def test_user_status_paginates_until_short_page():
    # page of 2 then page of 1 (< count) -> stop
    pages = [
        [{"id": 3, "creationTimeSeconds": 30, "verdict": "OK",
          "author": {"participantType": "PRACTICE"},
          "problem": {"contestId": 1, "index": "A", "name": "a", "rating": 800, "tags": []}},
         {"id": 2, "creationTimeSeconds": 20, "verdict": "WRONG_ANSWER",
          "author": {"participantType": "CONTESTANT"},
          "problem": {"contestId": 1, "index": "A", "name": "a", "rating": 800, "tags": []}}],
        [{"id": 1, "creationTimeSeconds": 10, "verdict": "OK",
          "author": {"participantType": "CONTESTANT"},
          "problem": {"contestId": 2, "index": "B", "name": "b", "rating": 900, "tags": ["dp"]}}],
    ]

    def handler(request):
        frm = int(request.url.params["from"])
        page = pages[0] if frm == 1 else pages[1]
        return _resp({"status": "OK", "result": page})

    c = cf_client.CFClient(transport=httpx.MockTransport(handler), min_interval=0, page_size=2)
    subs = c.user_status("x")
    assert [s.id for s in subs] == [3, 2, 1]
    assert subs[1].participant_type == "CONTESTANT"
    assert subs[2].problem.pid == "2B"


def test_backoff_then_success(monkeypatch):
    monkeypatch.setattr(cf_client.time, "sleep", lambda *_: None)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return _resp({"status": "FAILED", "comment": "limit exceeded"}, status=503)
        return _resp({"status": "OK", "result": []})

    c = cf_client.CFClient(transport=httpx.MockTransport(handler), min_interval=0)
    assert c.user_status("x") == []
    assert calls["n"] == 2  # retried once


def test_failed_status_raises(monkeypatch):
    monkeypatch.setattr(cf_client.time, "sleep", lambda *_: None)

    def handler(request):
        return _resp({"status": "FAILED", "comment": "handle not found"}, status=400)

    c = cf_client.CFClient(transport=httpx.MockTransport(handler), min_interval=0, max_retries=2)
    with pytest.raises(cf_client.CFError):
        c.user_info("nope")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_cf_client.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ingest.cf_client'`.

- [ ] **Step 3: Write minimal implementation**

```python
# ingest/cf_client.py
"""Codeforces public API client: rate-limited, backed off, paginated (docs/06)."""
import time

import httpx
from pydantic import BaseModel, Field

BASE_URL = "https://codeforces.com/api/"
RETRY_STATUS = {429, 503}


class CFError(RuntimeError):
    pass


class CFProblem(BaseModel):
    contest_id: int | None = Field(default=None, alias="contestId")
    index: str = ""
    name: str = ""
    rating: int | None = None
    tags: list[str] = []

    @property
    def pid(self) -> str:
        return f"{self.contest_id}{self.index}"


class CFSubmission(BaseModel):
    id: int
    creation_time: int = Field(alias="creationTimeSeconds")
    problem: CFProblem
    verdict: str | None = None
    author: dict = {}

    @property
    def participant_type(self) -> str:
        return self.author.get("participantType", "")


class CFRatingChange(BaseModel):
    contest_id: int = Field(alias="contestId")
    new_rating: int = Field(alias="newRating")
    update_time: int = Field(alias="ratingUpdateTimeSeconds")


class CFUserInfo(BaseModel):
    handle: str
    rating: int | None = None
    max_rating: int | None = Field(default=None, alias="maxRating")
    rank: str | None = None


class CFClient:
    def __init__(self, transport=None, min_interval: float = 1.5, max_retries: int = 4,
                 page_size: int = 1000) -> None:
        self._client = httpx.Client(base_url=BASE_URL, transport=transport, timeout=30.0)
        self._min_interval = min_interval
        self._max_retries = max_retries
        self._page_size = page_size
        self._last_request = 0.0

    def close(self) -> None:
        self._client.close()

    def _throttle(self) -> None:
        wait = self._min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

    def _get(self, method: str, **params):
        backoff = 2.0
        for attempt in range(self._max_retries):
            self._throttle()
            resp = self._client.get(method, params=params)
            try:
                body = resp.json()
            except ValueError:
                body = {}
            if resp.status_code in RETRY_STATUS:
                if attempt < self._max_retries - 1:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
            if resp.status_code != 200 or body.get("status") != "OK":
                raise CFError(f"{method}: HTTP {resp.status_code} {body.get('comment', resp.text[:200])}")
            return body["result"]
        raise CFError(f"{method}: exhausted retries")

    def problemset_problems(self) -> tuple[list[CFProblem], dict[str, int]]:
        result = self._get("problemset.problems")
        problems = [CFProblem.model_validate(p) for p in result.get("problems", [])]
        solved: dict[str, int] = {}
        for stat in result.get("problemStatistics", []):
            pid = f"{stat.get('contestId')}{stat.get('index')}"
            solved[pid] = stat.get("solvedCount", 0)
        return problems, solved

    def user_status(self, handle: str) -> list[CFSubmission]:
        out: list[CFSubmission] = []
        frm = 1
        while True:
            page = self._get("user.status", handle=handle, **{"from": frm, "count": self._page_size})
            out.extend(CFSubmission.model_validate(s) for s in page)
            if len(page) < self._page_size:
                break
            frm += self._page_size
        return out

    def user_rating(self, handle: str) -> list[CFRatingChange]:
        return [CFRatingChange.model_validate(r) for r in self._get("user.rating", handle=handle)]

    def user_info(self, handle: str) -> CFUserInfo:
        result = self._get("user.info", handles=handle)
        return CFUserInfo.model_validate(result[0])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_cf_client.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add ingest/cf_client.py tests/test_cf_client.py
git commit -m "feat(m0): rate-limited Codeforces API client + models"
```

---

### Task 6: Normalize + merge (the pedagogical core)

**Files:**
- Create: `ingest/normalize.py`
- Test: `tests/test_normalize.py`

**Interfaces:**
- Consumes: `CFSubmission` from Task 5.
- Produces:
  - `Episode` dataclass: `user_id, problem_id, solved, n_attempts, first_verdict, solved_in_contest, first_seen_at (int), solved_at (int | None)`.
  - `normalize(user_id: str, submissions: list[CFSubmission]) -> list[Episode]` — collapses to one episode per problem. Skips submissions whose verdict is not graded (only `OK` + the four fail verdicts count) and whose problem has no `contest_id`.
  - `merge(old: Episode | None, new: Episode) -> Episode` — combines a stored episode with a new-only episode (chronological guarantee: every `new` submission is later than every `old` one).
- Consumed by: `ingest/poller.py` (Task 7).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_normalize.py
from ingest.cf_client import CFSubmission
from ingest import normalize


def sub(id, t, verdict, ptype, cid=1, idx="A"):
    return CFSubmission.model_validate({
        "id": id, "creationTimeSeconds": t, "verdict": verdict,
        "author": {"participantType": ptype},
        "problem": {"contestId": cid, "index": idx, "name": "p", "rating": 800, "tags": ["math"]},
    })


def test_solved_after_wa_practice():
    subs = [sub(1, 10, "WRONG_ANSWER", "PRACTICE"), sub(2, 20, "OK", "PRACTICE")]
    eps = normalize.normalize("u", subs)
    assert len(eps) == 1
    e = eps[0]
    assert e.problem_id == "1A"
    assert e.solved is True
    assert e.n_attempts == 2
    assert e.first_verdict == "WRONG_ANSWER"
    assert e.solved_in_contest is False
    assert e.first_seen_at == 10
    assert e.solved_at == 20


def test_solved_first_try_in_contest():
    eps = normalize.normalize("u", [sub(1, 5, "OK", "CONTESTANT")])
    e = eps[0]
    assert e.n_attempts == 1 and e.first_verdict == "OK"
    assert e.solved_in_contest is True and e.solved_at == 5


def test_unsolved_only_failures():
    eps = normalize.normalize("u", [sub(1, 1, "WRONG_ANSWER", "PRACTICE"),
                                    sub(2, 2, "TIME_LIMIT_EXCEEDED", "PRACTICE")])
    e = eps[0]
    assert e.solved is False and e.solved_at is None
    assert e.n_attempts == 2 and e.solved_in_contest is False


def test_compilation_error_is_ignored():
    eps = normalize.normalize("u", [sub(1, 1, "COMPILATION_ERROR", "PRACTICE")])
    assert eps == []


def test_two_problems_two_episodes():
    eps = normalize.normalize("u", [sub(1, 1, "OK", "PRACTICE", cid=1, idx="A"),
                                    sub(2, 2, "OK", "PRACTICE", cid=2, idx="B")])
    assert {e.problem_id for e in eps} == {"1A", "2B"}


def test_problem_without_contest_id_skipped():
    s = CFSubmission.model_validate({
        "id": 1, "creationTimeSeconds": 1, "verdict": "OK",
        "author": {"participantType": "PRACTICE"},
        "problem": {"index": "A", "name": "acmsguru", "tags": []},
    })
    assert normalize.normalize("u", [s]) == []


def test_merge_unsolved_then_solved():
    old = normalize.normalize("u", [sub(1, 10, "WRONG_ANSWER", "PRACTICE")])[0]
    new = normalize.normalize("u", [sub(2, 20, "OK", "CONTESTANT")])[0]
    m = normalize.merge(old, new)
    assert m.solved is True and m.solved_at == 20
    assert m.n_attempts == 2
    assert m.first_verdict == "WRONG_ANSWER"   # earliest submission overall
    assert m.first_seen_at == 10
    assert m.solved_in_contest is True          # the solving submission's flag


def test_merge_none_old_returns_new():
    new = normalize.normalize("u", [sub(1, 1, "OK", "PRACTICE")])[0]
    assert normalize.merge(None, new) is new
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_normalize.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ingest.normalize'`.

- [ ] **Step 3: Write minimal implementation**

```python
# ingest/normalize.py
"""Collapse CF submissions into per-(user,problem) episodes (docs/06).

Incremental-merge correctness: CF submissions are chronological and the
ingest cursor guarantees every NEW submission is later than every stored
one, so merge(old, new) needs no re-fetch of history.
"""
from dataclasses import dataclass

from ingest.cf_client import CFSubmission

SOLVE_VERDICT = "OK"
FAIL_VERDICTS = {"WRONG_ANSWER", "TIME_LIMIT_EXCEEDED", "RUNTIME_ERROR", "MEMORY_LIMIT_EXCEEDED"}
GRADED = {SOLVE_VERDICT} | FAIL_VERDICTS
IN_CONTEST = {"CONTESTANT", "MANAGER"}


@dataclass
class Episode:
    user_id: str
    problem_id: str
    solved: bool
    n_attempts: int
    first_verdict: str
    solved_in_contest: bool
    first_seen_at: int
    solved_at: int | None


def normalize(user_id: str, submissions: list[CFSubmission]) -> list[Episode]:
    by_problem: dict[str, list[CFSubmission]] = {}
    for s in submissions:
        if s.verdict not in GRADED or s.problem.contest_id is None:
            continue
        by_problem.setdefault(s.problem.pid, []).append(s)

    episodes: list[Episode] = []
    for pid, subs in by_problem.items():
        subs.sort(key=lambda s: s.creation_time)
        first = subs[0]
        solve = next((s for s in subs if s.verdict == SOLVE_VERDICT), None)
        episodes.append(Episode(
            user_id=user_id,
            problem_id=pid,
            solved=solve is not None,
            n_attempts=len(subs),
            first_verdict=first.verdict or "",
            solved_in_contest=bool(solve and solve.participant_type in IN_CONTEST),
            first_seen_at=first.creation_time,
            solved_at=solve.creation_time if solve else None,
        ))
    return episodes


def merge(old: Episode | None, new: Episode) -> Episode:
    if old is None:
        return new
    return Episode(
        user_id=old.user_id,
        problem_id=old.problem_id,
        solved=old.solved or new.solved,
        n_attempts=old.n_attempts + new.n_attempts,
        first_verdict=old.first_verdict,                 # earliest overall
        solved_in_contest=old.solved_in_contest if old.solved else new.solved_in_contest,
        first_seen_at=old.first_seen_at,
        solved_at=old.solved_at if old.solved else new.solved_at,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_normalize.py -v`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
git add ingest/normalize.py tests/test_normalize.py
git commit -m "feat(m0): normalize submissions -> episodes + incremental merge"
```

---

### Task 7: Poller orchestration + Makefile + end-to-end test + M0 gate

**Files:**
- Create: `ingest/poller.py`
- Create: `Makefile`
- Test: `tests/test_poller.py`

**Interfaces:**
- Consumes: `CFClient` (Task 5), `normalize`/`merge`/`Episode` (Task 6), `db` helpers (Task 4), `config.settings`.
- Produces:
  - `run(handle: str | None = None, client: CFClient | None = None) -> dict` — the `/ingest` entry point. Returns a report dict: `{"new_episodes": int, "new_solves": int, "catalog_refreshed": bool, "cursor": int}`. `client` is injectable for tests.
  - `_to_db_row(ep: Episode) -> dict` — converts an `Episode` (unix-seconds timestamps) to the dict `db.upsert_interaction` expects (tz-aware datetimes).
- Behaviour:
  1. handle = arg or `settings.cf_handle`; raise `ValueError` if empty.
  2. refresh catalog if `db.catalog_age_seconds(conn)` is `None` or > 86400; upsert all problems with their `solved_count`.
  3. cursor = `db.get_cursor(conn, handle)`; fetch `user_status`; keep submissions with `creation_time > cursor`.
  4. also upsert the `problems` referenced by those submissions (FK safety; rating/tags from the submission).
  5. `normalize` the new submissions → per-problem new episodes; for each, `merge` with `db.get_interaction(...)` and `db.upsert_interaction(...)`.
  6. `set_cursor` to the max `creation_time` seen (only if newer); commit.

- [ ] **Step 1: Write the failing test** (fake client; real DB; skips if DB down)

```python
# tests/test_poller.py
import os
import psycopg
import pytest
from ingest import poller, db
from ingest.cf_client import CFSubmission, CFProblem

DB = os.environ.get("DATABASE_URL", "postgresql://whetstone:whetstone@localhost:5432/whetstone")


def _db_up() -> bool:
    try:
        psycopg.connect(DB, connect_timeout=2).close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _db_up(), reason="Postgres not running")


class FakeClient:
    """Stand-in for CFClient with a scripted submission stream."""
    def __init__(self, subs, solved_counts=None):
        self._subs = subs
        self._solved = solved_counts or {}

    def problemset_problems(self):
        # one catalog problem so the FK + catalog path exercise
        return [CFProblem.model_validate(
            {"contestId": 1, "index": "A", "name": "p", "rating": 800, "tags": ["math"]}
        )], {"1A": 5000}

    def user_status(self, handle):
        return self._subs


def _sub(id, t, verdict, ptype, cid=1, idx="A"):
    return CFSubmission.model_validate({
        "id": id, "creationTimeSeconds": t, "verdict": verdict,
        "author": {"participantType": ptype},
        "problem": {"contestId": cid, "index": idx, "name": "p", "rating": 800, "tags": ["math"]},
    })


@pytest.fixture(autouse=True)
def clean():
    c = db.connect()
    c.execute("DELETE FROM interactions WHERE user_id='_pt'")
    c.execute("DELETE FROM ingest_state WHERE user_id='_pt'")
    c.commit()
    c.close()


def test_first_run_lands_episodes():
    client = FakeClient([_sub(1, 10, "WRONG_ANSWER", "PRACTICE"),
                         _sub(2, 20, "OK", "PRACTICE")])
    report = poller.run(handle="_pt", client=client)
    assert report["new_episodes"] == 1
    assert report["new_solves"] == 1
    assert report["cursor"] == 20
    c = db.connect()
    ep = db.get_interaction(c, "_pt", "1A")
    assert ep["solved"] is True and ep["n_attempts"] == 2
    c.close()


def test_second_run_is_incremental_and_merges():
    poller.run(handle="_pt", client=FakeClient([_sub(1, 10, "WRONG_ANSWER", "PRACTICE")]))
    # second run delivers only a later submission; poller filters by cursor anyway
    report = poller.run(handle="_pt", client=FakeClient([
        _sub(1, 10, "WRONG_ANSWER", "PRACTICE"),  # <= cursor, skipped
        _sub(2, 30, "OK", "CONTESTANT"),          # > cursor, merged
    ]))
    c = db.connect()
    ep = db.get_interaction(c, "_pt", "1A")
    assert ep["solved"] is True
    assert ep["n_attempts"] == 2          # 1 (old) + 1 (new), not double-counted
    assert ep["solved_in_contest"] is True
    assert report["cursor"] == 30
    c.close()


def test_empty_handle_raises():
    with pytest.raises(ValueError):
        poller.run(handle="", client=FakeClient([]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_poller.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'ingest.poller'`.

- [ ] **Step 3: Write minimal implementation**

```python
# ingest/poller.py
"""Orchestrate one ingest run: CF -> normalize/merge -> Postgres (docs/05 step 1).

ponytail: fetches user.status and filters by the stored cursor each run, then
merges new-only episodes with stored ones (correct because submissions are
chronological). Cursor + merge keep re-runs cheap and idempotent.
"""
from datetime import datetime, timezone

from config import settings
from ingest import db, normalize
from ingest.cf_client import CFClient, CFProblem, CFSubmission

CATALOG_MAX_AGE_S = 86400


def _ts(seconds: int | None) -> datetime | None:
    return datetime.fromtimestamp(seconds, tz=timezone.utc) if seconds is not None else None


def _to_db_row(ep: normalize.Episode) -> dict:
    return {
        "user_id": ep.user_id,
        "problem_id": ep.problem_id,
        "solved": ep.solved,
        "n_attempts": ep.n_attempts,
        "first_verdict": ep.first_verdict,
        "solved_in_contest": ep.solved_in_contest,
        "first_seen_at": _ts(ep.first_seen_at),
        "solved_at": _ts(ep.solved_at),
    }


def _problem_row(p: CFProblem, solved_count: int | None) -> dict:
    return {
        "id": p.pid, "contest_id": p.contest_id, "idx": p.index, "name": p.name,
        "rating": p.rating, "tags": p.tags, "solved_count": solved_count,
    }


def _refresh_catalog(conn, client) -> bool:
    age = db.catalog_age_seconds(conn)
    if age is not None and age <= CATALOG_MAX_AGE_S:
        return False
    problems, solved = client.problemset_problems()
    rows = [_problem_row(p, solved.get(p.pid)) for p in problems if p.contest_id is not None]
    db.upsert_problems(conn, rows)
    return True


def run(handle: str | None = None, client: CFClient | None = None) -> dict:
    handle = handle or settings.cf_handle
    if not handle:
        raise ValueError("CF handle required (set CF_HANDLE or pass handle=)")
    owns_client = client is None
    client = client or CFClient()
    conn = db.connect()
    try:
        refreshed = _refresh_catalog(conn, client)

        cursor = db.get_cursor(conn, handle)
        all_subs: list[CFSubmission] = client.user_status(handle)
        new_subs = [s for s in all_subs if s.creation_time > cursor]

        # FK safety: ensure problems referenced by new submissions exist.
        seen: dict[str, CFProblem] = {}
        for s in new_subs:
            if s.problem.contest_id is not None:
                seen[s.problem.pid] = s.problem
        if seen:
            db.upsert_problems(conn, [_problem_row(p, None) for p in seen.values()])

        new_eps = normalize.normalize(handle, new_subs)
        new_solves = 0
        for ep in new_eps:
            existing = db.get_interaction(conn, handle, ep.problem_id)
            old = _existing_to_episode(existing) if existing else None
            merged = normalize.merge(old, ep)
            db.upsert_interaction(conn, _to_db_row(merged))
            if merged.solved and not (old and old.solved):
                new_solves += 1

        max_ct = max((s.creation_time for s in new_subs), default=cursor)
        if max_ct > cursor:
            db.set_cursor(conn, handle, max_ct)
        conn.commit()
        return {"new_episodes": len(new_eps), "new_solves": new_solves,
                "catalog_refreshed": refreshed, "cursor": max_ct}
    finally:
        conn.close()
        if owns_client:
            client.close()


def _existing_to_episode(row: dict) -> normalize.Episode:
    def secs(dt):
        return int(dt.timestamp()) if dt else None
    return normalize.Episode(
        user_id=row["user_id"], problem_id=row["problem_id"], solved=row["solved"],
        n_attempts=row["n_attempts"], first_verdict=row["first_verdict"],
        solved_in_contest=row["solved_in_contest"],
        first_seen_at=secs(row["first_seen_at"]), solved_at=secs(row["solved_at"]),
    )


if __name__ == "__main__":
    print(run())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/test_poller.py -v`
Expected: PASS (3 passed) with the DB up.

- [ ] **Step 5: Create the `Makefile`**

```makefile
.PHONY: dbup dbdown migrate ingest test fmt lint

dbup:
	docker compose -f infra/docker-compose.yml up -d db

dbdown:
	docker compose -f infra/docker-compose.yml down

migrate:
	uv run alembic upgrade head

ingest:
	uv run python -m ingest.poller

test:
	uv run pytest -q

fmt:
	uv run black . && uv run ruff check --fix .

lint:
	uv run ruff check . && uv run black --check .
```

- [ ] **Step 6: Full test sweep + lint**

Run: `uv run pytest -q` then `make lint`
Expected: all tests pass (DB-backed ones require `make dbup`); lint clean.

- [ ] **Step 7: Commit**

```bash
git add ingest/poller.py Makefile tests/test_poller.py
git commit -m "feat(m0): ingest poller orchestration + Makefile + e2e test"
```

- [ ] **Step 8: M0 ACCEPTANCE GATE (live, needs the user's CF handle)**

> This is the `docs/08` M0 gate. Requires the real handle — **prompt the user for it here** and set `CF_HANDLE` in `.env`.

```bash
cp -n .env.example .env          # then edit .env: set CF_HANDLE=<your handle>
make dbup
make migrate
make ingest                      # patient: rate-limited; full history pull on first run
```

Then sanity-check against the user's actual CF profile:

```bash
uv run python - <<'PY'
from ingest import db
c = db.connect()
print("episodes:", c.execute("SELECT count(*) FROM interactions").fetchone())
print("solved:", c.execute("SELECT count(*) FROM interactions WHERE solved").fetchone())
print("by rating bucket:")
for r in c.execute("""
    SELECT (p.rating/200)*200 AS bucket, count(*)
    FROM interactions i JOIN problems p ON p.id=i.problem_id
    WHERE i.solved AND p.rating IS NOT NULL
    GROUP BY 1 ORDER BY 1""").fetchall():
    print("  ", r)
print("top tags:")
for r in c.execute("""
    SELECT tag, count(*) FROM interactions i
    JOIN problems p ON p.id=i.problem_id, unnest(p.tags) tag
    WHERE i.solved GROUP BY tag ORDER BY 2 DESC LIMIT 10""").fetchall():
    print("  ", r)
c.close()
PY
```

**Gate PASS criteria:** total solved count and the per-rating / per-tag distributions match the user's Codeforces profile (eyeball against codeforces.com). If counts look wrong → stop, diagnose (do not loosen). On PASS, M0 is complete; open the M1 plan next.

---

## Self-Review

**1. Spec coverage** (against the bootstrap+M0 spec and `docs/08` M0 checklist):
- `cf_client.py` typed wrappers for `problemset.problems`, `user.status`, `user.rating`, `user.info` + rate-limit + backoff → **Task 5**. ✓
- Pull full `user.status` + catalog; sanity-check counts/coverage → **Task 7 Step 8 (gate)**. ✓
- `normalize.py` collapse → episodes, in-contest vs upsolve flag → **Task 6**. ✓
- Load into Postgres via `docs/06` schema + Alembic migration → **Task 3** (+ **Task 4** access). ✓
- All 6 docs/06 tables incl. non-optional `propensity` → **Task 3**. ✓
- `config.py` single home for constants, annotated → **Task 2**. ✓
- uv/3.12, compose Postgres, ruff/black/pytest → **Tasks 1, 3, 7**. ✓
- `/ingest` command behaviour (catalog refresh-if-stale, incremental, normalize, upsert, report) → **Task 7**. ✓

**2. Placeholder scan:** No "TBD"/"add error handling"/"similar to Task N". Every code step is complete. The only intentional stub is `config.py`'s pedagogical constants (real values from docs/02, used M1+) — documented as such, not a placeholder.

**3. Type consistency:** `Episode` fields are identical across Tasks 6 & 7. `CFProblem.pid`, `CFSubmission.participant_type/creation_time`, `db.upsert_interaction(ep: dict)` keys, and `poller._to_db_row` output keys all match. `problemset_problems` returns `tuple[list[CFProblem], dict]` consistently in Tasks 5 & 7. Cursor is `int` everywhere.

**No outstanding risks.** (The `_get` retry condition was simplified to `if resp.status_code in RETRY_STATUS:` — CF signals throttling via 429/503; the status-string check was redundant.)
