from model.irt import SkillModel, TagSkill
from recommender import assess
from recommender.candidates import Candidate


def _c(pid, b, tags):
    return Candidate(pid=pid, b=b, tags=tags, solved_count=100)


def test_assess_prefers_p_half_for_target_topic():
    m = SkillModel()
    m.tags["dp"] = TagSkill(mu=1700, sigma=300)
    cands = [_c("A", 1200, ["dp"]), _c("B", 1700, ["dp"]), _c("C", 2400, ["dp"])]
    out = assess.assess_select(m, cands, topics={"dp"}, k=3)
    assert out[0]["candidate"].pid == "B"  # b == mu -> P ~ 0.5 -> max info


def test_assess_ignores_non_target_topics():
    m = SkillModel()
    m.tags["dp"] = TagSkill(mu=1700, sigma=300)
    cands = [_c("X", 1700, ["math"])]
    assert assess.assess_select(m, cands, topics={"dp"}, k=3) == []


def test_assess_prefers_higher_sigma_topic_at_equal_info():
    m = SkillModel()
    m.tags["dp"] = TagSkill(mu=1700, sigma=300)
    m.tags["greedy"] = TagSkill(mu=1700, sigma=120)
    cands = [_c("D", 1700, ["dp"]), _c("G", 1700, ["greedy"])]
    out = assess.assess_select(m, cands, topics={"dp", "greedy"}, k=2)
    assert out[0]["candidate"].pid == "D"  # same b=mu, higher sigma -> more var reduction


def test_assess_respects_k():
    m = SkillModel()
    m.tags["dp"] = TagSkill(mu=1700, sigma=300)
    cands = [_c(str(i), 1700 + i, ["dp"]) for i in range(5)]
    assert len(assess.assess_select(m, cands, topics={"dp"}, k=2)) == 2
