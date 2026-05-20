"""CNN for facial landmark regression (Task 2, Approach 3).

Lecture provenance:
- W10_L19: 2D convolutions, channels, padding, receptive fields, equivariance.
- W10_L20: data augmentation as teaching transformations.
- W09_L18: regression-to-coordinates framing; flip swaps landmark indices.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

from landmarks import horizontal_flip_with_landmarks


# ----------------------------- model ----------------------------------------


class FaceCNN(nn.Module):
    """Small CNN: 4 conv blocks (32->64->128->128) with BN+ReLU+MaxPool,
    then a 256-d FC head with dropout, then 10 outputs reshaped to (B, 5, 2).

    Each 3x3 conv with padding=1 preserves spatial size before pooling
    (W10_L19); depth grows the effective receptive field.
    """

    def __init__(self, in_channels: int = 1, n_landmarks: int = 5, dropout: float = 0.3):
        super().__init__()
        self.n_landmarks = n_landmarks

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(in_channels, 32),   # 64 -> 32
            block(32, 64),            # 32 -> 16
            block(64, 128),           # 16 -> 8
            block(128, 128),          # 8  -> 4
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, n_landmarks * 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.features(x)
        out = self.head(z)
        return out.view(-1, self.n_landmarks, 2)


MODEL_CONFIG = {
    "in_channels": 1,
    "n_landmarks": 5,
    "dropout": 0.3,
    "conv_channels": [32, 64, 128, 128],
    "kernel_size": 3,
    "padding": 1,
    "pool": 2,
    "fc_hidden": 256,
    "input_hw": [64, 64],
}


# ----------------------------- augmentation ---------------------------------


@dataclass
class AugConfig:
    rotate_deg: float = 15.0    # uniform +/- this
    translate_px: float = 4.0   # uniform +/- this in x and y
    scale_pct: float = 0.10     # uniform +/- this (e.g. 0.10 -> [0.9, 1.1])
    brightness: tuple = (0.8, 1.2)
    hflip_p: float = 0.5


def _affine_image_and_landmarks(
    img: np.ndarray,
    landmarks: np.ndarray,
    angle_deg: float,
    tx: float,
    ty: float,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply combined rotate/scale/translate about image centre to both image
    and landmark coordinates. Image is warped with cv2.warpAffine using
    border replication; landmarks transformed by the same 2x3 affine matrix."""
    H, W = img.shape[:2]
    cx, cy = (W - 1) * 0.5, (H - 1) * 0.5
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, scale)
    M[0, 2] += tx
    M[1, 2] += ty

    warped = cv2.warpAffine(
        img, M, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    lm = landmarks.astype(np.float32, copy=False)
    ones = np.ones((lm.shape[0], 1), dtype=np.float32)
    homo = np.concatenate([lm, ones], axis=1)            # (5, 3)
    new_lm = homo @ M.T.astype(np.float32)               # (5, 2)
    return warped.astype(img.dtype, copy=False), new_lm


def augment_sample(
    img: np.ndarray,
    landmarks: np.ndarray,
    rng: np.random.Generator,
    cfg: AugConfig = AugConfig(),
) -> tuple[np.ndarray, np.ndarray]:
    """Online augmentation following W10_L20 (teach the network the
    transformations under which the label is invariant/equivariant).

    Image is expected float32 in [0, 1], shape (H, W). Landmarks (5, 2)
    in pixel coordinates of the same frame.
    """
    angle = float(rng.uniform(-cfg.rotate_deg, cfg.rotate_deg))
    tx = float(rng.uniform(-cfg.translate_px, cfg.translate_px))
    ty = float(rng.uniform(-cfg.translate_px, cfg.translate_px))
    scale = float(rng.uniform(1.0 - cfg.scale_pct, 1.0 + cfg.scale_pct))

    img2, lm2 = _affine_image_and_landmarks(img, landmarks, angle, tx, ty, scale)

    b = float(rng.uniform(cfg.brightness[0], cfg.brightness[1]))
    img2 = np.clip(img2 * b, 0.0, 1.0).astype(np.float32, copy=False)

    if rng.random() < cfg.hflip_p:
        img2, lm2 = horizontal_flip_with_landmarks(img2, lm2)

    return img2, lm2.astype(np.float32, copy=False)


# ----------------------------- dataset --------------------------------------


class FaceLandmarksDataset(Dataset):
    """Wrap (images, landmarks) numpy arrays as a torch Dataset.

    Images: (N, H, W) float32 in [0, 1]. Landmarks: (N, 5, 2) float32 in the
    same pixel frame. When ``augment=True`` an online augmentation pipeline is
    applied per sample (W10_L20).
    """

    def __init__(
        self,
        images: np.ndarray,
        landmarks: np.ndarray,
        augment: bool = False,
        aug_cfg: AugConfig | None = None,
        seed: int = 0,
    ):
        assert images.ndim == 3, f"expect (N, H, W), got {images.shape}"
        assert landmarks.shape[1:] == (5, 2), f"expect (N, 5, 2), got {landmarks.shape}"
        self.images = images.astype(np.float32, copy=False)
        self.landmarks = landmarks.astype(np.float32, copy=False)
        self.augment = augment
        self.aug_cfg = aug_cfg or AugConfig()
        # Per-worker seed-safe RNG (one rng per dataset; each __getitem__ draws).
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return self.images.shape[0]

    def __getitem__(self, idx: int):
        img = self.images[idx]
        lm = self.landmarks[idx]
        if self.augment:
            img, lm = augment_sample(img, lm, self._rng, self.aug_cfg)
        # to tensor: (1, H, W)
        t_img = torch.from_numpy(np.ascontiguousarray(img))[None, :, :]
        t_lm = torch.from_numpy(np.ascontiguousarray(lm))
        return t_img, t_lm


# ----------------------------- evaluation -----------------------------------


@torch.no_grad()
def predict_landmarks(model: nn.Module, images: np.ndarray, device, batch_size: int = 128) -> np.ndarray:
    """Run model over (N, H, W) float32 images, return (N, 5, 2) numpy."""
    model.eval()
    outs = []
    N = images.shape[0]
    for i in range(0, N, batch_size):
        chunk = images[i:i + batch_size].astype(np.float32, copy=False)
        x = torch.from_numpy(chunk)[:, None, :, :].to(device)
        y = model(x).detach().cpu().numpy()
        outs.append(y)
    return np.concatenate(outs, axis=0)
