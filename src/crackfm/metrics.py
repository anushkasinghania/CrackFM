"""Segmentation metrics for crack masks.

Pure NumPy so they run without a GPU or deep-learning framework. All functions
accept binary masks (bool or {0,1}) of identical shape and return floats.

Includes region metrics (IoU, Dice, precision/recall/F1) and a tolerance-based
boundary-F score, the building block behind the ODS/OIS protocol used in the
crack/edge-detection literature.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

EPS = 1e-7


def _binarize(mask: np.ndarray) -> np.ndarray:
    return np.asarray(mask).astype(bool)


def confusion(pred: np.ndarray, gt: np.ndarray) -> tuple[int, int, int, int]:
    """Return (tp, fp, fn, tn) for two binary masks."""
    p, g = _binarize(pred), _binarize(gt)
    tp = int(np.logical_and(p, g).sum())
    fp = int(np.logical_and(p, ~g).sum())
    fn = int(np.logical_and(~p, g).sum())
    tn = int(np.logical_and(~p, ~g).sum())
    return tp, fp, fn, tn


def iou(pred: np.ndarray, gt: np.ndarray) -> float:
    tp, fp, fn, _ = confusion(pred, gt)
    denom = tp + fp + fn
    return (tp + EPS) / (denom + EPS)


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    tp, fp, fn, _ = confusion(pred, gt)
    return (2 * tp + EPS) / (2 * tp + fp + fn + EPS)


def precision_recall_f1(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float, float]:
    tp, fp, fn, _ = confusion(pred, gt)
    prec = (tp + EPS) / (tp + fp + EPS)
    rec = (tp + EPS) / (tp + fn + EPS)
    f1 = 2 * prec * rec / (prec + rec + EPS)
    return prec, rec, f1


def boundary_f(pred: np.ndarray, gt: np.ndarray, tolerance: int = 2) -> float:
    """Tolerance-based boundary F-measure.

    A predicted crack pixel counts as a true positive if a ground-truth crack
    pixel lies within ``tolerance`` pixels (Euclidean), and vice-versa. This is
    the matching rule behind ODS/OIS scoring for thin structures, where exact
    pixel overlap is too strict.
    """
    p, g = _binarize(pred), _binarize(gt)
    if p.sum() == 0 and g.sum() == 0:
        return 1.0
    if p.sum() == 0 or g.sum() == 0:
        return 0.0

    # Distance from every pixel to the nearest GT / pred crack pixel.
    dist_to_gt = ndi.distance_transform_edt(~g)
    dist_to_pred = ndi.distance_transform_edt(~p)

    matched_pred = p & (dist_to_gt <= tolerance)
    matched_gt = g & (dist_to_pred <= tolerance)

    precision = matched_pred.sum() / (p.sum() + EPS)
    recall = matched_gt.sum() / (g.sum() + EPS)
    return float(2 * precision * recall / (precision + recall + EPS))


def all_metrics(pred: np.ndarray, gt: np.ndarray, boundary_tolerance: int = 2) -> dict[str, float]:
    prec, rec, f1 = precision_recall_f1(pred, gt)
    return {
        "iou": iou(pred, gt),
        "dice": dice(pred, gt),
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "boundary_f": boundary_f(pred, gt, boundary_tolerance),
    }


class MetricAccumulator:
    """Average a metric dict over a dataset (per-image mean)."""

    def __init__(self) -> None:
        self._sums: dict[str, float] = {}
        self._n = 0

    def update(self, pred: np.ndarray, gt: np.ndarray, boundary_tolerance: int = 2) -> None:
        m = all_metrics(pred, gt, boundary_tolerance)
        for k, v in m.items():
            self._sums[k] = self._sums.get(k, 0.0) + v
        self._n += 1

    def compute(self) -> dict[str, float]:
        if self._n == 0:
            return {}
        return {k: v / self._n for k, v in self._sums.items()}
