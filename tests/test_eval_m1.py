import math
import random

from eval import run_m1
from eval.dataset import Record, temporal_split


def _synthetic(n=3000, true_theta=1900.0, s=173.7, seed=0):
    rng = random.Random(seed)
    recs = []
    for i in range(n):
        b = rng.uniform(1000, 2800)
        p = 1 / (1 + math.exp(-(true_theta - b) / s))
        y = 1 if rng.random() < p else 0
        recs.append(Record(b=b, tags=["dp"], y=y, t=i))
    return recs


def test_temporal_split_is_chronological():
    recs = [Record(b=1, tags=["x"], y=0, t=i) for i in range(10)]
    tr, te = temporal_split(recs, frac=0.8)
    assert len(tr) == 8 and len(te) == 2
    assert te[0].t == 8


def test_evaluate_on_synthetic_beats_baseline_and_calibrates():
    recs = _synthetic()
    res = run_m1.evaluate_records(recs, cf_rating=1500)
    assert res["auc"] >= 0.70
    assert res["auc"] >= res["per_tag_auc"] + 0.05
    assert res["ece"] <= 0.05
