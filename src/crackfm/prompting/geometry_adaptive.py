"""Geometry-adaptive prompting — the core methodological contribution.

A promptable segmenter (SAM 2) expects point and/or box prompts. For blob-like
objects a single box prompt is fine, but cracks are thin, elongated and
branching: a box covers mostly background and a single centre point misses most
of the structure. CrackFM instead derives prompts from the *geometry* of a
coarse crack proposal:

* **Medial-axis point prompts** — positive points sampled along the skeleton of
  the crack region, so prompt coverage follows the crack itself.
* **Curvature-adaptive density** — more points are placed where the skeleton
  bends or branches (high information) and fewer along straight runs.
* **Per-component boxes** — a tight box per connected component, giving the
  decoder a coarse spatial anchor for each separate crack.
* **Background negatives** — negative points sampled in the dilation margin
  just outside the crack, discouraging leakage into shadows/texture.

Everything here is pure NumPy + scikit-image so it is testable and usable
independently of any deep-learning framework. The output ``PromptSet`` maps
directly onto SAM 2's ``point_coords`` / ``point_labels`` / ``box`` API.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize


@dataclass
class PromptSet:
    """Prompts in image (x, y) pixel coordinates.

    point_coords: (N, 2) float array of (x, y).
    point_labels: (N,) int array, 1 = foreground, 0 = background.
    boxes:        (M, 4) float array of (x0, y0, x1, y1).
    """

    point_coords: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), np.float32))
    point_labels: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.int64))
    boxes: np.ndarray = field(default_factory=lambda: np.zeros((0, 4), np.float32))

    @property
    def num_points(self) -> int:
        return int(self.point_coords.shape[0])

    @property
    def num_boxes(self) -> int:
        return int(self.boxes.shape[0])


def connected_component_boxes(mask: np.ndarray, min_area: int = 8) -> np.ndarray:
    """Tight (x0, y0, x1, y1) box per connected component above ``min_area``."""
    mask = np.asarray(mask).astype(bool)
    labels, n = ndi.label(mask)
    boxes = []
    for sl_y, sl_x in ndi.find_objects(labels) or []:
        comp = labels[sl_y, sl_x]
        if comp.size == 0:
            continue
        area = int((comp > 0).sum())
        if area < min_area:
            continue
        boxes.append([sl_x.start, sl_y.start, sl_x.stop - 1, sl_y.stop - 1])
    if not boxes:
        return np.zeros((0, 4), np.float32)
    return np.asarray(boxes, np.float32)


def _skeleton_curvature(skel: np.ndarray) -> np.ndarray:
    """Per-skeleton-pixel score ~ local neighbour count.

    Endpoints (1 neighbour) and junctions (>=3 neighbours) are the geometrically
    informative locations; straight runs have exactly 2 neighbours. We weight by
    deviation from 2 so prompts concentrate at bends, tips and branch points.
    """
    skel = skel.astype(np.uint8)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)
    neighbours = ndi.convolve(skel, kernel, mode="constant", cval=0)
    score = np.abs(neighbours.astype(np.int32) - 2) + 1  # >=1 everywhere on skel
    return (score * skel).astype(np.float32)


def medial_axis_points(
    mask: np.ndarray,
    n_points: int = 24,
    strategy: str = "curvature",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Sample (x, y) positive points along the skeleton of ``mask``.

    strategy='uniform'  : even sampling over skeleton pixels.
    strategy='curvature': sampling probability ∝ local curvature score, so more
                          points land on bends/branches/tips.
    """
    rng = rng or np.random.default_rng(0)
    mask = np.asarray(mask).astype(bool)
    if mask.sum() == 0:
        return np.zeros((0, 2), np.float32)

    skel = skeletonize(mask)
    ys, xs = np.nonzero(skel)
    if len(xs) == 0:  # mask too thin to skeletonize; fall back to mask pixels
        ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros((0, 2), np.float32)

    n = min(n_points, len(xs))
    if strategy == "curvature":
        score = _skeleton_curvature(skel)
        w = score[ys, xs]
        if w.sum() <= 0:
            w = np.ones_like(w, dtype=np.float32)
        p = w / w.sum()
        idx = rng.choice(len(xs), size=n, replace=False, p=p)
    else:
        idx = np.linspace(0, len(xs) - 1, n).round().astype(int)

    pts = np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)
    return pts


def _background_points(
    mask: np.ndarray, n: int, margin: int = 6, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Negative points in the dilation margin just outside the crack."""
    rng = rng or np.random.default_rng(0)
    mask = np.asarray(mask).astype(bool)
    if n <= 0 or mask.sum() == 0:
        return np.zeros((0, 2), np.float32)
    dil = ndi.binary_dilation(mask, iterations=margin)
    ring = dil & ~mask
    ys, xs = np.nonzero(ring)
    if len(xs) == 0:
        return np.zeros((0, 2), np.float32)
    idx = rng.choice(len(xs), size=min(n, len(xs)), replace=False)
    return np.stack([xs[idx], ys[idx]], axis=1).astype(np.float32)


class GeometryAdaptivePrompter:
    """Turn a coarse binary crack mask into a SAM-2-ready ``PromptSet``."""

    def __init__(
        self,
        n_points: int = 24,
        strategy: str = "curvature",
        neg_points: int = 8,
        per_component_box: bool = True,
        min_component_area: int = 8,
        seed: int = 0,
    ) -> None:
        self.n_points = n_points
        self.strategy = strategy
        self.neg_points = neg_points
        self.per_component_box = per_component_box
        self.min_component_area = min_component_area
        self._rng = np.random.default_rng(seed)

    def __call__(self, coarse_mask: np.ndarray) -> PromptSet:
        pos = medial_axis_points(coarse_mask, self.n_points, self.strategy, self._rng)
        neg = _background_points(coarse_mask, self.neg_points, rng=self._rng)

        coords = np.concatenate([pos, neg], axis=0) if len(neg) else pos
        labels = np.concatenate(
            [np.ones(len(pos), np.int64), np.zeros(len(neg), np.int64)]
        ) if len(neg) else np.ones(len(pos), np.int64)

        boxes = (
            connected_component_boxes(coarse_mask, self.min_component_area)
            if self.per_component_box
            else np.zeros((0, 4), np.float32)
        )
        return PromptSet(point_coords=coords, point_labels=labels, boxes=boxes)
