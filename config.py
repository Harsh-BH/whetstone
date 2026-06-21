"""Single home for runtime settings and pedagogical constants (docs/05).

Every pedagogical constant traces to a rule in docs/02-learning-science.md.
These are PRIORS — the system estimates better values from the user's data
where it can (docs/02 "the one rule above all"). Used from M1+.
"""

import math

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    cf_handle: str = ""
    database_url: str = "postgresql://whetstone:whetstone@localhost:5432/whetstone"
    target_rating: int = 1900  # R (docs/01)
    target_date: str = "2026-12-21"  # D (docs/01, ~6 months)
    weekly_hours: int = 8  # H (docs/01)


settings = Settings()

# --- Pedagogical constants (docs/02). Priors; auto-tuned values override later. ---
MIN_GENERATION_ATTEMPTS_BEFORE_HINT = 1  # P1 retrieval practice
ASSESS_TARGET_P = 0.5  # P2/P8 max Fisher information
TRAIN_TARGET_BAND = (0.55, 0.80)  # P2 desirable difficulty (auto-tuned)
TARGET_RETRIEVABILITY = 0.90  # P3 spacing: review trigger
MAX_CONSECUTIVE_SAME_TAG = 1  # P4 interleaving
MASTERY_SUSTAINED_REVIEWS = 2  # P5 mastery
FRONTIER_ONLY = True  # P6 prereq-DAG frontier

# --- IRT / knowledge-model constants (docs/03). Scale shared with CF ratings. ---
IRT_S = 400.0 / math.log(10)  # logistic scale ~ Elo (a 400-pt gap ≈ CF win prob); fit per user
PRIOR_MU = 1500.0  # cold-start θ prior when CF rating unknown (docs/03 cold-start)
PRIOR_SIGMA = 350.0  # high initial per-tag uncertainty (drives Assess in M2)

# --- Recommender constants (docs/04, docs/02). ---
ASSESS_SIGMA_THRESHOLD = 120.0  # P8: σ above this routes a topic to Assess (CAT)
FRONTIER_MARGIN = 200.0  # P6: prereq "satisfied" when μ ≥ R_band − margin (M2 proxy; M3=mastery)
STRETCH_TARGET_P = 0.40  # docs/04: stretch problems for growth/exploration
DAILY_BLEND = {"train": 0.8, "stretch": 0.2}  # M2 (no FSRS reviews yet); M3 -> 65/20/15
MINUTES_PER_PROBLEM = 30.0  # rough cost to size the daily set from the H-hour budget
