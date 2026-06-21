from eval.dataset import Record
from eval.metrics import auc
from model import dkt


def test_dkt_learns_separable_pattern():
    # skill A always solved, skill B always failed -> easily separable
    recs = []
    for i in range(60):
        tag = "A" if i % 2 == 0 else "B"
        recs.append(Record(b=1500, tags=[tag], y=(1 if tag == "A" else 0), t=i))
    m = dkt.train_dkt(recs[:40], epochs=120, seed=0)
    probs = dkt.predict(m, recs)  # full sequence; eval the held-out tail
    test_y = [r.y for r in recs[40:]]
    assert auc(test_y, probs[40:]) > 0.6


def test_dkt_deterministic_given_seed():
    recs = [Record(b=1500, tags=["A"], y=i % 2, t=i) for i in range(20)]
    m1 = dkt.train_dkt(recs, epochs=20, seed=0)
    m2 = dkt.train_dkt(recs, epochs=20, seed=0)
    assert dkt.predict(m1, recs) == dkt.predict(m2, recs)
