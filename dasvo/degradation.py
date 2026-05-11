"""
Deterministic RGB degradations for robustness evaluation (simulated link budget / motion blur).
"""

from __future__ import annotations

import cv2
import numpy as np

from dasvo.settings import SETTINGS

PRESETS = frozenset(SETTINGS.degradations.presets.keys())


def apply_visual_degradation(img_bgr: np.ndarray, preset: str | None, seed: int = 0) -> np.ndarray:
    """
    Apply degradation to a BGR uint8 image. Depth maps are left unchanged upstream.
    """
    if img_bgr is None or preset is None or preset == "none":
        return img_bgr

    if preset not in PRESETS:
        raise ValueError(f"Unknown degradation preset '{preset}'. Expected one of {sorted(PRESETS)}")

    out = img_bgr
    if preset == "blur_mild":
        out = cv2.GaussianBlur(out, (5, 5), 0)
    elif preset == "blur_heavy":
        out = cv2.GaussianBlur(out, (11, 11), 0)
    elif preset == "gaussian_noise":
        rng = np.random.default_rng(seed)
        noise = rng.normal(0.0, 12.0, out.shape)
        noisy = np.clip(out.astype(np.float32) + noise, 0.0, 255.0).astype(np.uint8)
        out = noisy
    elif preset == "jpeg_low":
        ok, buf = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 25])
        if not ok:
            raise RuntimeError("JPEG encode failed in jpeg_low degradation")
        out = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    elif preset == "low_light":
        gamma = 1.8
        adjusted = (out.astype(np.float32) / 255.0) ** gamma
        out = np.clip(adjusted * 255.0, 0.0, 255.0).astype(np.uint8)

    return out
