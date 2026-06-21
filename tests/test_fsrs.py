from config import FSRS_INIT_STABILITY, TARGET_RETRIEVABILITY
from retention import fsrs


def test_retrievability_decreases_with_elapsed():
    s = 10.0
    assert fsrs.retrievability(s, 0.0) == 1.0
    assert fsrs.retrievability(s, 5.0) > fsrs.retrievability(s, 20.0)


def test_retrievability_at_stability_equals_target():
    for s in (1.0, 10.0, 50.0):
        assert abs(fsrs.retrievability(s, s) - TARGET_RETRIEVABILITY) < 1e-3


def test_days_until_due_equals_stability_at_default_target():
    for s in (1.0, 10.0, 50.0):
        assert abs(fsrs.days_until_due(s) - s) < 1e-6


def test_success_grows_stability_more_when_reviewed_late():
    s = 10.0
    immediate = fsrs.next_stability(s, success=True, elapsed_days=0.0)
    early = fsrs.next_stability(s, success=True, elapsed_days=5.0)
    late = fsrs.next_stability(s, success=True, elapsed_days=20.0)
    assert immediate == s  # reviewing at R=1 (no forgetting) yields no stability gain
    assert late > early > s  # reviewing later (lower R) grows stability more


def test_lapse_shrinks_stability_but_not_below_init():
    assert fsrs.next_stability(10.0, success=False, elapsed_days=5.0) < 10.0
    assert fsrs.next_stability(1.0, success=False, elapsed_days=5.0) >= FSRS_INIT_STABILITY
