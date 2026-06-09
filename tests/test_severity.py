import numpy as np

from crackfm.severity import quantify, severity_class


def test_empty_mask_is_none():
    m = np.zeros((30, 30), bool)
    res = quantify(m)
    assert res.severity == "none"
    assert res.max_width == 0.0
    assert res.length == 0.0


def test_wider_crack_has_larger_width():
    thin = np.zeros((40, 40), bool)
    thin[:, 20] = True              # 1 px wide
    thick = np.zeros((40, 40), bool)
    thick[:, 18:23] = True          # 5 px wide
    wt = quantify(thin).max_width
    kt = quantify(thick).max_width
    assert kt > wt


def test_severity_class_thresholds():
    assert severity_class(0.0) == "none"
    assert severity_class(0.5, (1.0, 3.0)) == "low"
    assert severity_class(2.0, (1.0, 3.0)) == "medium"
    assert severity_class(5.0, (1.0, 3.0)) == "high"


def test_mm_scale_changes_unit_and_value():
    m = np.zeros((40, 40), bool)
    m[:, 18:23] = True
    px = quantify(m)
    mm = quantify(m, mm_per_pixel=0.5)
    assert px.unit == "px" and mm.unit == "mm"
    assert abs(mm.max_width - 0.5 * px.max_width) < 1e-5


def test_length_tracks_skeleton():
    m = np.zeros((50, 50), bool)
    m[10:40, 25] = True  # ~30 px vertical crack
    res = quantify(m)
    assert res.length >= 25
