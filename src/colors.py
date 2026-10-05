"""Dominant color of a vehicle crop, using HSV pixel voting.

Averaging BGR values mixes the body color with windows, tires and road, so a
red car on gray asphalt easily comes out "brown" or "gray". Instead we:

1. keep the central part of the box (the body panels, mostly),
2. put every pixel in one color bin (achromatic bins first, then by hue),
3. return the bin with the most votes.
"""
from __future__ import annotations

import cv2
import numpy as np

# OpenCV hue range is 0-179.  (upper bound exclusive, name)
HUE_BINS = [
    (10, "kirmizi"),
    (22, "turuncu"),
    (35, "sari"),
    (85, "yesil"),
    (130, "mavi"),
    (160, "mor"),
    (180, "kirmizi"),   # red wraps around the hue circle
]

DARK_V = 60          # below this brightness a pixel is "siyah"
LOW_SAT = 45         # below this saturation a pixel is white / gray
WHITE_V = 170        # low-saturation pixels brighter than this are "beyaz"


def center_crop(img: np.ndarray, keep: float = 0.6) -> np.ndarray:
    """Central `keep` fraction of the image in both directions."""
    h, w = img.shape[:2]
    dy, dx = int(h * (1 - keep) / 2), int(w * (1 - keep) / 2)
    return img[dy:h - dy, dx:w - dx]


def dominant_color(bgr: np.ndarray | None) -> str | None:
    """Turkish color name of the dominant color, or None for an empty crop."""
    if bgr is None or bgr.size == 0 or min(bgr.shape[:2]) < 4:
        return None

    crop = center_crop(bgr)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.int32)
    h, s, v = hsv[:, 0], hsv[:, 1], hsv[:, 2]

    votes: dict[str, int] = {}
    dark = v < DARK_V
    low_sat = (s < LOW_SAT) & ~dark
    chromatic = ~dark & ~low_sat

    votes["siyah"] = int(dark.sum())
    votes["beyaz"] = int((low_sat & (v >= WHITE_V)).sum())
    votes["gri"] = int((low_sat & (v < WHITE_V)).sum())

    hues = h[chromatic]
    lower = 0
    for upper, name in HUE_BINS:
        votes[name] = votes.get(name, 0) + int(((hues >= lower) & (hues < upper)).sum())
        lower = upper

    return max(votes, key=votes.get)
