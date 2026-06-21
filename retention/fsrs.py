"""FSRS spaced-repetition core (docs/02 P3): power forgetting curve + stability update.

Simplified FSRS-5: retrievability R(t) = (1 + FACTOR·t/S)^DECAY, calibrated so a concept
hits R=0.90 exactly when elapsed == stability S. Stability grows on a successful review
(more when reviewed late, i.e. at low R — the spacing effect) and shrinks on a lapse.
The full 17-weight FSRS fit to the user's own recall data is a later refinement (docs/02).
"""

from config import (
    FSRS_DECAY,
    FSRS_FACTOR,
    FSRS_FORGET,
    FSRS_GROWTH,
    FSRS_INIT_STABILITY,
    TARGET_RETRIEVABILITY,
)


def retrievability(stability_days: float, elapsed_days: float) -> float:
    """Probability of recall after `elapsed_days`, given current stability."""
    s = max(stability_days, 1e-6)
    return (1.0 + FSRS_FACTOR * elapsed_days / s) ** FSRS_DECAY


def days_until_due(stability_days: float, target: float = TARGET_RETRIEVABILITY) -> float:
    """Elapsed days at which retrievability falls to `target` (review trigger)."""
    s = max(stability_days, 1e-6)
    return s * (target ** (1.0 / FSRS_DECAY) - 1.0) / FSRS_FACTOR


def next_stability(stability_days: float, success: bool, elapsed_days: float) -> float:
    """Updated stability after a review. Success grows S (more when reviewed at low R);
    a lapse shrinks it toward the initial stability."""
    if success:
        r = retrievability(stability_days, elapsed_days)
        return stability_days * (1.0 + FSRS_GROWTH * (1.0 - r))
    return max(FSRS_INIT_STABILITY, stability_days * FSRS_FORGET)
