import random

from eval import sim_m2
from model.irt import SkillModel, TagSkill
from recommender.candidates import Candidate


def _model():
    m = SkillModel()  # s≈173.7
    m.tags["implementation"] = TagSkill(mu=1900, sigma=80)  # prereq enabler
    m.tags["brute force"] = TagSkill(mu=1900, sigma=80)
    m.tags["dp"] = TagSkill(mu=1500, sigma=80)  # weak frontier topic (gap 400)
    m.tags["math"] = TagSkill(mu=1500, sigma=80)  # weak root topic
    m.tags["greedy"] = TagSkill(mu=1900, sigma=80)  # already at target (gap 0)
    return m


def _cands():
    out = []
    for tag in ("dp", "math", "greedy"):
        for b in (900, 1300, 1400, 1900, 2400):  # 900/2400 out of band; 1300/1400 in band
            out.append(Candidate(pid=f"{tag}{b}", b=b, tags=[tag], solved_count=100))
    return out


def test_assess_converges_faster_than_random():
    bs = [float(b) for b in range(800, 3001, 100)]
    a = sim_m2.items_to_converge(1900, 173.7, bs, 120, 100, "assess", random.Random(1))
    r = sim_m2.items_to_converge(1900, 173.7, bs, 120, 100, "random", random.Random(1))
    assert a <= r


def test_greedy_value_beats_random_and_fixed():
    m, cands = _model(), _cands()
    g = sim_m2.greedy_select(m, cands, 6, random.Random(0), r_band=1900)
    gv = sim_m2.pedagogical_value(m, g, 1900)
    rv = sim_m2.pedagogical_value(
        m, sim_m2.random_select(m, cands, 6, random.Random(0), 1900), 1900
    )
    fv = sim_m2.pedagogical_value(
        m, sim_m2.fixed_curriculum_select(m, cands, 6, random.Random(0), 1900), 1900
    )
    assert gv > rv and gv > fv


def test_greedy_stays_in_band_and_avoids_strong_topic():
    m, cands = _model(), _cands()
    sel = sim_m2.greedy_select(m, cands, 6, random.Random(0), r_band=1900)
    assert sim_m2.in_band_fraction(m, sel) == 1.0  # only in-band problems
    # 'greedy' tag has gap 0 -> greedy policy should not spend value there
    assert all(
        "greedy" not in c.tags or sim_m2._cand_gap(m, c, {"dp", "math"}, 1900) == 0 for c in sel
    )


def test_fixed_curriculum_serves_easy_out_of_band():
    m, cands = _model(), _cands()
    sel = sim_m2.fixed_curriculum_select(m, cands, 3, random.Random(0), 1900)
    # ascending ladder starts at b=900 (too easy) -> low in-band fraction
    assert sim_m2.in_band_fraction(m, sel) < 1.0
