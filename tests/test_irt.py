import math
import random

from model.irt import SkillModel, TagSkill


def test_p_half_when_theta_equals_b():
    m = SkillModel()
    m.tags["dp"] = TagSkill(mu=1500, sigma=100)
    p, info = m.predict_solve(b=1500, tags=["dp"])
    assert abs(p - 0.5) < 1e-9
    assert abs(info - 0.25 / m.s**2) < 1e-12


def test_monotonic_in_skill():
    m = SkillModel()
    m.tags["dp"] = TagSkill(mu=1800, sigma=100)
    p_easy, _ = m.predict_solve(b=1200, tags=["dp"])
    p_hard, _ = m.predict_solve(b=2400, tags=["dp"])
    assert p_easy > 0.5 > p_hard


def test_theta_eff_is_min_over_tags():
    m = SkillModel()
    m.tags["dp"] = TagSkill(mu=2000, sigma=100)
    m.tags["math"] = TagSkill(mu=1400, sigma=100)
    assert m.theta_eff(["dp", "math"]) == 1400


def test_update_moves_mu_toward_outcome_and_shrinks_sigma():
    m = SkillModel()
    m.tags["dp"] = TagSkill(mu=1500, sigma=200)
    s0 = m.tags["dp"].sigma
    m.update(b=1700, tags=["dp"], y=1)
    assert m.tags["dp"].mu > 1500
    assert m.tags["dp"].sigma < s0


def test_unknown_tag_starts_at_prior():
    m = SkillModel(prior_mu=1500, prior_sigma=350)
    p, _ = m.predict_solve(b=1500, tags=["never_seen"])
    assert abs(p - 0.5) < 1e-9
    assert m.tags["never_seen"].mu == 1500 and m.tags["never_seen"].sigma == 350


def test_recovers_true_skill_on_synthetic_stream():
    rng = random.Random(0)
    true_theta, s = 1900.0, 173.7
    m = SkillModel(s=s, prior_mu=1500, prior_sigma=350)
    for _ in range(4000):
        b = rng.uniform(1000, 2800)
        p_true = 1 / (1 + math.exp(-(true_theta - b) / s))
        y = 1 if rng.random() < p_true else 0
        m.update(b=b, tags=["dp"], y=y)
    assert abs(m.tags["dp"].mu - true_theta) < 120
