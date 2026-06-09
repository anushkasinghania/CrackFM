import numpy as np

from crackfm.metrics import all_metrics, boundary_f, dice, iou, precision_recall_f1


def test_perfect_match():
    m = np.zeros((20, 20), bool)
    m[5:15, 8] = True
    assert iou(m, m) > 0.99
    assert dice(m, m) > 0.99
    assert boundary_f(m, m, tolerance=1) > 0.99


def test_disjoint():
    a = np.zeros((20, 20), bool)
    b = np.zeros((20, 20), bool)
    a[2, 2] = True
    b[18, 18] = True
    assert iou(a, b) < 1e-3
    assert boundary_f(a, b, tolerance=1) < 1e-3


def test_both_empty_is_perfect_boundary():
    z = np.zeros((10, 10), bool)
    assert boundary_f(z, z) == 1.0


def test_boundary_tolerance_helps_thin_offset():
    gt = np.zeros((20, 20), bool)
    gt[:, 10] = True
    pred = np.zeros((20, 20), bool)
    pred[:, 11] = True  # one pixel off
    strict = boundary_f(pred, gt, tolerance=0)
    loose = boundary_f(pred, gt, tolerance=2)
    assert loose > strict


def test_precision_recall_relationship():
    gt = np.zeros((10, 10), bool)
    gt[:, 5] = True
    pred = np.zeros((10, 10), bool)
    pred[:5, 5] = True  # half the crack, no false positives
    p, r, f1 = precision_recall_f1(pred, gt)
    assert p > 0.99
    assert abs(r - 0.5) < 1e-2
    assert "iou" in all_metrics(pred, gt)
