"""Image decoding, quality checks, and encoding helpers."""

from __future__ import annotations

import base64
import io
from typing import Any, Dict

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError


ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_RESOLUTION = 128
MIN_SHARPNESS = 60.0
MIN_BRIGHTNESS = 12.0
MAX_BRIGHTNESS = 245.0
MIN_STD_DEV = 8.0


class ImageValidationError(ValueError):
    """Raised when an uploaded file cannot be used as an image."""


def validate_extension(filename: str) -> None:
    """Reject unsupported image extensions before decoding."""
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix not in ALLOWED_EXTENSIONS:
        supported = ", ".join(sorted(ALLOWED_EXTENSIONS))
        raise ImageValidationError(f"Unsupported file type. Please upload one of: {supported}.")


def load_image_from_bytes(raw: bytes) -> Image.Image:
    """Decode image bytes into a normalized RGB PIL image."""
    if not raw:
        raise ImageValidationError("The uploaded file is empty.")

    try:
        image = Image.open(io.BytesIO(raw))
        image.verify()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageValidationError("The uploaded file is corrupted or is not a readable image.") from exc

    return image


def check_image_quality(image: Image.Image) -> Dict[str, Any]:
    """Validate MRI image quality before inference."""
    try:
        img_array = np.array(image.convert("RGB"))
        height, width = img_array.shape[:2]

        if height < MIN_RESOLUTION or width < MIN_RESOLUTION:
            return {
                "suitable": False,
                "reason": "Resolution too low",
                "details": (
                    f"Image is {width}x{height} px. Please upload an MRI image of at "
                    f"least {MIN_RESOLUTION}x{MIN_RESOLUTION} px."
                ),
                "metrics": {"resolution": f"{width}x{height}"},
            }

        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        brightness = float(np.mean(gray))
        contrast = float(np.std(gray))

        if brightness < MIN_BRIGHTNESS:
            return {
                "suitable": False,
                "reason": "Image too dark",
                "details": "The scan is underexposed. Please upload a clearer MRI image.",
                "metrics": _quality_metrics(width, height, sharpness, brightness, contrast),
            }

        if brightness > MAX_BRIGHTNESS:
            return {
                "suitable": False,
                "reason": "Image too bright",
                "details": "The scan is overexposed. Please upload a properly exposed MRI image.",
                "metrics": _quality_metrics(width, height, sharpness, brightness, contrast),
            }

        if contrast < MIN_STD_DEV:
            return {
                "suitable": False,
                "reason": "Image may be corrupted",
                "details": "Very low pixel variation suggests a blank, uniform, or corrupted image.",
                "metrics": _quality_metrics(width, height, sharpness, brightness, contrast),
            }

        if sharpness < MIN_SHARPNESS:
            return {
                "suitable": False,
                "reason": "Image appears blurred",
                "details": "The scan is not sharp enough for reliable analysis. Please upload a clearer MRI image.",
                "metrics": _quality_metrics(width, height, sharpness, brightness, contrast),
            }

        return {
            "suitable": True,
            "reason": "Image quality acceptable",
            "details": f"{width}x{height} px, sharpness {sharpness:.0f}, brightness {brightness:.0f}",
            "metrics": _quality_metrics(width, height, sharpness, brightness, contrast),
        }
    except Exception as exc:
        return {
            "suitable": False,
            "reason": "Quality check error",
            "details": f"The image could not be evaluated safely: {exc}",
        }


def _quality_metrics(width: int, height: int, sharpness: float, brightness: float, contrast: float) -> Dict[str, Any]:
    return {
        "resolution": f"{width}x{height}",
        "sharpness": round(sharpness, 1),
        "brightness": round(brightness, 1),
        "contrast": round(contrast, 1),
    }


def bgr_to_b64(img_bgr: np.ndarray) -> str:
    """Encode an OpenCV BGR image as base64 PNG."""
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise ImageValidationError("Could not encode image output.")
    return base64.b64encode(buf).decode("utf-8")


def rgb_to_b64(image: Image.Image | np.ndarray) -> str:
    """Encode an RGB PIL image or array as base64 PNG."""
    arr = np.array(image.convert("RGB") if isinstance(image, Image.Image) else image)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr_to_b64(bgr)
