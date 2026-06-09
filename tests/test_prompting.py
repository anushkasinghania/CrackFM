import numpy as np

from crackfm.prompting import (
    GeometryAdaptivePrompter,
    connected_component_boxes,
    medial_axis_points,
)


def _diagonal_crack(n=40):
    m = np.zeros((n, n), bool)
    for i in range(5, n - 5):
        m[i, i] = True
        m[i, i + 1] = True
    return m


def test_medial_axis_points_on_skeleton():
    m = _diagonal_crack()
    pts = medial_axis_points(m, n_points=10, strategy="uniform")
    assert pts.shape[1] == 2
    assert 0 < len(pts) <= 10
    # Every sampled (x, y) should fall on a crack pixel.
    for x, y in pts.astype(int):
        assert m[y, x]


def test_empty_mask_yields_no_points():
    m = np.zeros((30, 30), bool)
    assert len(medial_axis_points(m, 10)) == 0
    assert connected_component_boxes(m).shape == (0, 4)


def test_two_components_two_boxes():
    m = np.zeros((40, 40), bool)
    m[5:15, 10] = True
    m[25:35, 30] = True
    boxes = connected_component_boxes(m, min_area=3)
    assert boxes.shape[0] == 2


def test_prompter_produces_pos_and_neg():
    m = _diagonal_crack()
    prompter = GeometryAdaptivePrompter(n_points=12, neg_points=5, seed=1)
    ps = prompter(m)
    assert ps.num_points > 0
    assert set(np.unique(ps.point_labels)).issubset({0, 1})
    assert (ps.point_labels == 1).sum() > 0  # has positives
    assert ps.num_boxes >= 1


def test_curvature_strategy_runs():
    m = _diagonal_crack()
    pts = medial_axis_points(m, n_points=8, strategy="curvature",
                             rng=np.random.default_rng(0))
    assert len(pts) > 0
