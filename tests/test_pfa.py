import math
import random

from eval.dataset import Record
from model import pfa


def _rec(tag, y, t, b=1500):
    return Record(b=b, tags=[tag], y=y, t=t)


def test_prior_opportunity_counts_exclude_current():
    recs = [_rec("dp", 1, 0), _rec("dp", 0, 1), _rec("dp", 1, 2)]
    x, y, names, vocab = pfa.build_features(recs)
    n = len(vocab)
    j = vocab.index("dp")
    # 3rd attempt: prior_correct=1 (rec0), prior_incorrect=1 (rec1) -> γ+ρ feats sum to 2
    assert x[2][n + j] + x[2][2 * n + j] == 2.0
    # 1st attempt has no prior opportunities
    assert x[0][n + j] == 0.0 and x[0][2 * n + j] == 0.0


def test_positive_learning_rate_for_improving_skill():
    # Learning: P(solve) rises with the number of prior CORRECT attempts on the skill.
    rng = random.Random(0)
    recs = []
    correct = 0
    for i in range(60):
        p = 1.0 / (1.0 + math.exp(-1.0 * (correct - 2)))  # rises with prior correct
        y = 1 if rng.random() < p else 0
        recs.append(_rec("dp", y, i))
        if y:
            correct += 1
    assert len({r.y for r in recs}) == 2  # both classes present
    m = pfa.PFAModel().fit(recs)
    assert m.learning_rates()["dp"] > 0  # prior-correct is the true driver -> positive γ


def test_predict_in_unit_interval():
    recs = [_rec("dp", i % 2, i) for i in range(20)]
    m = pfa.PFAModel().fit(recs)
    ps = m.predict(recs)
    assert all(0.0 < p < 1.0 for p in ps)
