"""Heuristic guardrail that rejects obvious non-brain-MRI uploads."""

from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np
from PIL import Image


def validate_brain_mri(image: Image.Image) -> Dict[str, Any]:
    """
    Validate whether an image resembles a brain MRI.

    This is a conservative computer-vision gate for obvious non-medical images.
    It checks grayscale dominance, central anatomy, background ratio, texture,
    and approximate left-right symmetry. It is not a diagnostic model.
    """
    arr = np.array(image.convert("RGB").resize((256, 256)))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)

    channel_delta = np.mean(np.max(arr, axis=2) - np.min(arr, axis=2))
    grayscale_score = max(0.0, 1.0 - float(channel_delta) / 55.0)

    _, mask = cv2.threshold(gray, max(8, int(np.percentile(gray, 35))), 255, cv2.THRESH_BINARY)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_area = max((cv2.contourArea(c) for c in contours), default=0.0)
    anatomy_ratio = largest_area / float(mask.shape[0] * mask.shape[1])

    moments = cv2.moments(mask)
    if moments["m00"]:
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        centered_score = 1.0 - min(1.0, (abs(cx - 128) + abs(cy - 128)) / 150.0)
    else:
        centered_score = 0.0

    left = gray[:, :128].astype(np.float32)
    right = cv2.flip(gray[:, 128:], 1).astype(np.float32)
    symmetry_error = np.mean(np.abs(left - right)) / 255.0
    symmetry_score = max(0.0, 1.0 - symmetry_error * 2.4)

    edge_density = float(np.mean(cv2.Canny(gray, 40, 120) > 0))
    texture_score = 1.0 if 0.015 <= edge_density <= 0.22 else 0.35

    background_ratio = float(np.mean(gray < 18))
    background_score = 1.0 if background_ratio >= 0.08 else 0.45

    score = (
        grayscale_score * 0.28
        + centered_score * 0.22
        + symmetry_score * 0.18
        + texture_score * 0.16
        + background_score * 0.10
        + (1.0 if 0.08 <= anatomy_ratio <= 0.82 else 0.3) * 0.06
    )
    accepted = bool(score >= 0.58)

    return {
        "accepted": accepted,
        "message": "Brain MRI validation passed" if accepted else "This is not a Brain MRI",
        "score": round(float(score) * 100, 1),
        "checks": {
            "grayscale_score": round(float(grayscale_score) * 100, 1),
            "centered_anatomy": round(float(centered_score) * 100, 1),
            "symmetry": round(float(symmetry_score) * 100, 1),
            "texture": round(float(texture_score) * 100, 1),
            "dark_background": round(float(background_score) * 100, 1),
            "anatomy_ratio": round(float(anatomy_ratio) * 100, 1),
        },
    }
