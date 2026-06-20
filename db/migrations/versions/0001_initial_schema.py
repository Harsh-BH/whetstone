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
