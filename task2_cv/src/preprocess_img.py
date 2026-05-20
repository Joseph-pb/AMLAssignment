"""Preprocessing utilities for Task 2 (face alignment).

Lecture provenance: W06_L12 (image basics, grayscale conversion, intensity
normalisation, histogram equalisation). All choices here come from that lecture.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


_LAYOUT_CANDIDATES = [
    Path(__file__).resolve().parent.parent / "data" / "data_layout.json",
    Path.cwd() / "data" / "data_layout.json",
    Path.cwd().parent / "data" / "data_layout.json",
]


def _find_layout() -> Path:
    for p in _LAYOUT_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "data_layout.json not found. Run 01_explore.ipynb first. "
        f"Looked in: {[str(p) for p in _LAYOUT_CANDIDATES]}"
    )


def load_data() -> dict:
    """Read ``data_layout.json`` and return train/test arrays at original
    resolution and dtype.

    Keys returned: ``train_images``, ``train_points``, ``test_images``.
    ``train_points`` is reshaped to ``(N, 5, 2)`` when stored flat as ``(N, 10)``.
    """
    layout_path = _find_layout()
    with open(layout_path) as f:
        layout = json.load(f)

    train_path = layout["train"]["path"]
    train_img_key = layout["train"]["images_key"]
    train_pt_key = layout["train"]["points_key"]
    test_path = layout["test"]["path"]
    test_img_key = layout["test"]["images_key"]

    with np.load(train_path, allow_pickle=True) as z:
        train_images = z[train_img_key].copy()
        train_points = z[train_pt_key].copy()
    with np.load(test_path, allow_pickle=True) as z:
        test_images = z[test_img_key].copy()

    if train_points.ndim == 2 and train_points.shape[1] == 10:
        train_points = train_points.reshape(-1, 5, 2)

    return {
        "train_images": train_images,
        "train_points": train_points,
        "test_images": test_images,
        "layout": layout,
    }


def to_grayscale(img: np.ndarray) -> np.ndarray:
    """Convert (H, W, 3) to (H, W) using ITU-R BT.601 luma coefficients
    0.299R + 0.587G + 0.114B (W06_L12). Pass through if already (H, W)."""
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[-1] == 1:
        return img[..., 0]
    if img.ndim == 3 and img.shape[-1] == 3:
        r, g, b = img[..., 0], img[..., 1], img[..., 2]
        out = 0.299 * r.astype(np.float32) + 0.587 * g.astype(np.float32) + 0.114 * b.astype(np.float32)
        if img.dtype == np.uint8:
            return np.clip(out, 0, 255).astype(np.uint8)
        return out.astype(img.dtype, copy=False)
    raise ValueError(f"Unexpected image shape: {img.shape}")


def resize_image_and_landmarks(
    img: np.ndarray,
    landmarks: np.ndarray | None,
    out_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray | None]:
    """Resize image and rescale landmark coordinates by ``(W_out / W_in, H_out / H_in)``.

    ``out_size = (H_out, W_out)``. Uses ``cv2.INTER_AREA`` for downsampling
    (the OpenCV-recommended antialiased choice) and ``cv2.INTER_LINEAR`` when
    upsampling. Landmarks may be ``None`` (test images).
    """
    H_out, W_out = out_size
    H_in, W_in = img.shape[:2]
    interp = cv2.INTER_AREA if (H_out <= H_in and W_out <= W_in) else cv2.INTER_LINEAR

    resized = cv2.resize(img, (W_out, H_out), interpolation=interp)

    if landmarks is None:
        return resized, None

    sx = W_out / W_in
    sy = H_out / H_in
    scaled = landmarks.astype(np.float32, copy=True)
    scaled[..., 0] *= sx
    scaled[..., 1] *= sy
    return resized, scaled


def normalise_intensity(img: np.ndarray) -> np.ndarray:
    """Scale to float32 in [0, 1] (W06_L12). If dtype is uint8, divide by 255;
    otherwise rescale by the observed max (treating max <= 1 as already-normalised)."""
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0
    arr = img.astype(np.float32, copy=False)
    mx = float(arr.max()) if arr.size else 1.0
    if mx > 1.0:
        return arr / 255.0
    return arr


def hist_equalise(img: np.ndarray) -> np.ndarray:
    """OpenCV ``equalizeHist`` on a uint8 grayscale image. W06_L12 flags that
    this can amplify noise — the caller decides whether to apply it."""
    if img.ndim != 2:
        raise ValueError(f"hist_equalise expects a (H, W) grayscale image, got {img.shape}")
    if img.dtype != np.uint8:
        raise ValueError(f"hist_equalise expects uint8, got {img.dtype}")
    return cv2.equalizeHist(img)


def preprocess_image(
    img: np.ndarray,
    landmarks: np.ndarray | None,
    config: dict,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Full preprocessing pipeline.

    Order: grayscale -> resize (and scale landmarks) -> optional hist_eq -> normalise to [0, 1].
    ``config`` keys: ``out_size``, ``grayscale``, ``hist_eq``, ``normalise``.
    """
    out_size = tuple(config["out_size"])
    do_gray = bool(config.get("grayscale", True))
    do_eq = bool(config.get("hist_eq", False))
    do_norm = bool(config.get("normalise", True))

    out = img
    if do_gray:
        out = to_grayscale(out)
    out, scaled_landmarks = resize_image_and_landmarks(out, landmarks, out_size)
    if do_eq:
        if out.dtype != np.uint8:
            # equalizeHist needs uint8 input; rescale temporarily.
            tmp = np.clip(out, 0, 255).astype(np.uint8) if out.max() > 1 else (out * 255).astype(np.uint8)
            out = hist_equalise(tmp)
        else:
            out = hist_equalise(out)
    if do_norm:
        out = normalise_intensity(out)
    return out, scaled_landmarks


DEFAULT_CONFIG = {
    "out_size": (64, 64),
    "grayscale": True,
    "hist_eq": False,
    "normalise": True,
}


if __name__ == "__main__":
    # Smoke tests on 3 training images.
    data = load_data()
    images = data["train_images"]
    points = data["train_points"]

    print(f"loaded train_images shape={images.shape} dtype={images.dtype}")
    print(f"loaded train_points shape={points.shape} dtype={points.dtype}")

    for i in range(3):
        img = images[i]
        lm = points[i]
        out, scaled_lm = preprocess_image(img, lm, DEFAULT_CONFIG)
        H_out, W_out = DEFAULT_CONFIG["out_size"]
        inside_x = (scaled_lm[:, 0] >= 0) & (scaled_lm[:, 0] <= W_out - 1)
        inside_y = (scaled_lm[:, 1] >= 0) & (scaled_lm[:, 1] <= H_out - 1)
        print(
            f"img#{i}: in={img.shape}/{img.dtype} range=[{img.min()},{img.max()}] "
            f"-> out={out.shape}/{out.dtype} range=[{out.min():.3f},{out.max():.3f}] "
            f"landmarks_in_bounds={inside_x.all() and inside_y.all()}"
        )
