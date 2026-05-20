"""Landmark utilities for Task 2 (face alignment).

Lecture provenance: W09_L18 — inter-eye-normalised Euclidean error,
left/right-flip annotation swap, CED.
"""

from __future__ import annotations

import numpy as np


# Landmark order is [left_eye, right_eye, nose, left_mouth, right_mouth].
# On a horizontal flip the *image* swaps left and right, so the annotations must
# also swap (W09_L18: "make sure you flip the annotations!"). Nose (index 2) stays.
LEFT_RIGHT_FLIP_PAIRS = [(0, 1), (3, 4)]


def horizontal_flip_with_landmarks(
    img: np.ndarray,
    landmarks: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Mirror image across the vertical axis and update landmarks accordingly.

    Swaps the index pairs in ``LEFT_RIGHT_FLIP_PAIRS`` — this is the W09_L18
    tripwire (flip the annotations, not just the pixels)."""
    W = img.shape[1]
    flipped_img = img[:, ::-1, ...].copy()

    flipped_lm = landmarks.astype(np.float32, copy=True)
    flipped_lm[..., 0] = (W - 1) - flipped_lm[..., 0]
    for a, b in LEFT_RIGHT_FLIP_PAIRS:
        flipped_lm[[a, b]] = flipped_lm[[b, a]]
    return flipped_img, flipped_lm


def per_landmark_euclidean(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Per-landmark Euclidean errors, returns ``(N, 5)`` in pixel units."""
    diff = pred.astype(np.float64) - true.astype(np.float64)
    return np.sqrt((diff ** 2).sum(axis=-1))


def nme(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """Per-image normalised mean error (report Eq. 2):

        NME_i = mean_k(||p_hat_ik - p_ik||) / d_ie_i

    where ``d_ie`` is the inter-eye distance computed from the ground-truth
    landmarks 0 and 1. Returns ``(N,)``.
    """
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    per_lm = per_landmark_euclidean(pred, true)  # (N, 5)
    d_ie = np.linalg.norm(true[:, 0, :] - true[:, 1, :], axis=-1)  # (N,)
    # Guard against zero (shouldn't happen with real annotations).
    d_ie = np.where(d_ie > 0, d_ie, np.nan)
    return per_lm.mean(axis=1) / d_ie


def ced_curve(
    nmes: np.ndarray,
    max_err: float = 0.5,
    n: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """Cumulative error distribution. Returns ``(thresholds, fractions)``."""
    nmes = np.asarray(nmes, dtype=np.float64)
    nmes = nmes[~np.isnan(nmes)]
    thresholds = np.linspace(0.0, max_err, n)
    fractions = np.array([(nmes < t).mean() for t in thresholds])
    return thresholds, fractions
