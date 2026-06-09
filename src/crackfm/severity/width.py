"""Severity quantification: turn a binary crack mask into physical measurements.

The approach is the standard skeleton + distance-transform method:

* The Euclidean distance transform of the mask gives, at every crack pixel, the
  distance to the nearest background pixel — i.e. the local half-width.
* Sampling that transform *on the skeleton* yields the local half-width along
  the crack's centreline; doubling gives the full local width.
* Skeleton pixel count approximates crack length (in pixels).

With a ``mm_per_pixel`` scale these become millimetres. A simple thresholded
severity class (low / medium / high) is derived from the maximum width, which
is what most inspection standards key on.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize


@dataclass
class CrackMeasurement:
    length: float          # skeleton length (px, or mm if scale given)
    mean_width: float      # mean width along skeleton
    max_width: float       # maximum width along skeleton
    p95_width: float       # 95th-percentile width (robust to skeleton spurs)
    area: float            # crack area (px^2 or mm^2)
    severity: str          # "none" | "low" | "medium" | "high"
    unit: str              # "px" or "mm"

    def to_dict(self) -> dict:
        return asdict(self)


def severity_class(max_width: float, thresholds: tuple[float, float] = (1.0, 3.0)) -> str:
    """Map a maximum width to a severity label using two thresholds (lo, hi)."""
    lo, hi = thresholds
    if max_width <= 0:
        return "none"
    if max_width < lo:
        return "low"
    if max_width < hi:
        return "medium"
    return "high"


def quantify(
    mask: np.ndarray,
    mm_per_pixel: float | None = None,
    width_thresholds: tuple[float, float] = (1.0, 3.0),
) -> CrackMeasurement:
    """Measure length / width / area / severity from a binary crack mask.

    If ``mm_per_pixel`` is given, all lengths are in mm (and area in mm^2);
    otherwise pixels. ``width_thresholds`` are interpreted in the same unit.
    """
    mask = np.asarray(mask).astype(bool)
    scale = float(mm_per_pixel) if mm_per_pixel else 1.0
    unit = "mm" if mm_per_pixel else "px"

    if mask.sum() == 0:
        return CrackMeasurement(0.0, 0.0, 0.0, 0.0, 0.0, "none", unit)

    skel = skeletonize(mask)
    # distance_transform_edt: distance from each True pixel to nearest False.
    dt = ndi.distance_transform_edt(mask)
    widths = 2.0 * dt[skel]  # full local width at each centreline pixel
    if widths.size == 0:
        widths = np.array([0.0])

    length = float(skel.sum()) * scale
    area = float(mask.sum()) * (scale ** 2)
    mean_w = float(widths.mean()) * scale
    max_w = float(widths.max()) * scale
    p95_w = float(np.percentile(widths, 95)) * scale

    sev = severity_class(max_w, width_thresholds)
    return CrackMeasurement(length, mean_w, max_w, p95_w, area, sev, unit)
