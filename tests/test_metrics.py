import math

from eval import metrics


def test_auc_perfect_separation():
    assert metrics.auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0


def test_auc_single_class_is_nan():
    assert math.isnan(metrics.auc([1, 1, 1], [0.5, 0.6, 0.7]))


def test_perfect_calibration_has_zero_ece():
    y = [0, 1] * 50
    p = [0.5] * 100
    assert metrics.ece(y, p, n_bins=10) < 1e-9


def test_ece_detects_miscalibration():
    y = [0] * 100
    p = [0.9] * 100
    assert abs(metrics.ece(y, p, n_bins=10) - 0.9) < 1e-9


def test_accuracy_and_brier():
    y = [1, 0, 1, 0]
    p = [0.9, 0.1, 0.4, 0.2]
    assert metrics.accuracy(y, p) == 0.75
    assert abs(metrics.brier(y, p) - (0.01 + 0.01 + 0.36 + 0.04) / 4) < 1e-9


def test_reliability_bins_sum_to_n():
    y = [0, 1, 1, 0, 1]
    p = [0.2, 0.8, 0.6, 0.3, 0.9]
    bins = metrics.reliability(y, p, n_bins=5)
    assert sum(b["n"] for b in bins) == 5
