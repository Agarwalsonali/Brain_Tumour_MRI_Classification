"""Real Grad-CAM generation using PyTorch forward/backward hooks."""

from __future__ import annotations

from typing import Any, Dict

import cv2
import numpy as np
from PIL import Image

from .image_processing import bgr_to_b64, rgb_to_b64
from .model_loader import IMG_SIZE, ModelLoadError, get_loaded_model_context, preprocess_image

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


class GradCAMError(RuntimeError):
    """Raised when Grad-CAM cannot be generated."""


def generate_gradcam(image: Image.Image, class_index: int | None = None) -> Dict[str, Any]:
    """Generate original, heatmap, and overlay images for the selected class."""
    if torch is None:
        raise ModelLoadError("PyTorch is not installed. Grad-CAM requires torch.")

    model, _, device, _ = get_loaded_model_context()
    target_layer = model.layer4[-1]
    activations = []
    gradients = []

    def forward_hook(_module, _inputs, output):
        activations.append(output.detach())

    def backward_hook(_module, _grad_input, grad_output):
        gradients.append(grad_output[0].detach())

    forward_handle = target_layer.register_forward_hook(forward_hook)
    backward_handle = target_layer.register_full_backward_hook(backward_hook)

    try:
        model.zero_grad(set_to_none=True)
        tensor = preprocess_image(image)
        logits = model(tensor)
        target_idx = int(class_index) if class_index is not None else int(torch.argmax(logits, dim=1).item())
        score = logits[:, target_idx].sum()
        score.backward()

        if not activations or not gradients:
            raise GradCAMError("Model hooks did not capture activations.")

        acts = activations[0]
        grads = gradients[0]
        weights = torch.mean(grads, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * acts, dim=1).squeeze()
        cam = torch.relu(cam)
        cam_np = cam.detach().cpu().numpy()
        cam_np = _normalize(cam_np)

        heatmap_uint8 = np.uint8(255 * cv2.resize(cam_np, (IMG_SIZE, IMG_SIZE)))
        heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        original_rgb = np.array(image.convert("RGB").resize((IMG_SIZE, IMG_SIZE)))
        original_bgr = cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(original_bgr, 0.58, heatmap_color, 0.42, 0)

        return {
            "original": rgb_to_b64(Image.fromarray(original_rgb)),
            "heatmap": bgr_to_b64(heatmap_color),
            "overlay": bgr_to_b64(overlay),
            "arrays": {
                "heatmap": heatmap_color,
                "overlay": overlay,
            },
        }
    except ModelLoadError:
        raise
    except Exception as exc:
        raise GradCAMError(f"Grad-CAM generation failed: {exc}") from exc
    finally:
        forward_handle.remove()
        backward_handle.remove()
        model.zero_grad(set_to_none=True)


def _normalize(cam: np.ndarray) -> np.ndarray:
    cam = cam - np.min(cam)
    max_val = np.max(cam)
    if max_val <= 1e-8:
        return np.zeros_like(cam, dtype=np.float32)
    return (cam / max_val).astype(np.float32)
