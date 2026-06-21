from model.irt import SkillModel, TagSkill
from recommender import prereq_dag


def test_roots_always_open():
    m = SkillModel()
    fr = prereq_dag.frontier(m, ["math", "implementation"], r_band=1900, margin=200)
    assert "math" in fr and "implementation" in fr


def test_child_gated_by_unmastered_prereq():
    m = SkillModel(prior_mu=1000)  # everything weak
    fr = prereq_dag.frontier(m, ["math", "number theory"], r_band=1900, margin=200)
    assert "math" in fr  # root
    assert "number theory" not in fr  # prereq math (1000) < 1700


def test_child_opens_when_prereq_strong():
    m = SkillModel()
    m.tags["math"] = TagSkill(mu=1800, sigma=100)  # >= 1700
    fr = prereq_dag.frontier(m, ["math", "number theory"], r_band=1900, margin=200)
    assert "number theory" in fr


def test_deep_node_needs_chain():
    m = SkillModel(prior_mu=1000)
    m.tags["dfs and similar"] = TagSkill(mu=1800, sigma=100)
    # 'graphs' needs 'dfs and similar' (1800 ok) -> open; 'trees' needs graphs+dfs,
    # graphs mu is prior (1000) -> trees gated.
    fr = prereq_dag.frontier(m, ["graphs", "trees", "dfs and similar"], r_band=1900, margin=200)
    assert "graphs" in fr
    assert "trees" not in fr


def test_frontier_does_not_mutate_model():
    m = SkillModel()
    prereq_dag.frontier(m, ["number theory"], r_band=1900, margin=200)
    assert m.tags == {}  # reading prereq μ must not create tag entries
