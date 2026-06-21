from eval.dataset import Record
from eval import learning_curves as lc


def test_fit_power_law_decreasing_is_ok():
    pts = [(1, 0.8), (2, 0.5), (3, 0.35), (4, 0.25), (5, 0.18)]  # decaying
    f = lc.fit_power_law(pts)
    assert f["c"] > 0 and f["ok"] is True


def test_fit_power_law_increasing_not_ok():
    pts = [(1, 0.2), (2, 0.35), (3, 0.5), (4, 0.7)]  # error rising = not learning
    f = lc.fit_power_law(pts)
    assert f["c"] < 0 and f["ok"] is False


def test_opportunity_error_bins_in_time_order():
    recs = [Record(b=1500, tags=["dp"], y=(1 if i >= 5 else 0), t=i) for i in range(10)]
    pts = lc.opportunity_error(recs, "dp", n_bins=2)
    # first half errors (y=0 -> err 1) high, second half (y=1 -> err 0) low
    assert pts[0][1] > pts[-1][1]


def test_curve_report_fraction_in_range_and_skips_sparse():
    recs = [Record(b=1500, tags=["dp"], y=(1 if i >= 4 else 0), t=i) for i in range(12)]
    recs += [Record(b=1500, tags=["rare"], y=0, t=100)]  # only 1 opportunity -> skipped
    rep = lc.curve_report(recs, min_opps=5)
    assert "rare" not in rep["skills"] and "dp" in rep["skills"]
    assert 0.0 <= rep["fraction_fit"] <= 1.0
