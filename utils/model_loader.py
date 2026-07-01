"""PyTorch ResNet50 model loading and inference for brain MRI classification."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from PIL import Image

LOGGER = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torchvision import models, transforms

    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    models = None  # type: ignore[assignment]
    transforms = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = BASE_DIR / "models" / "best_model.pth"
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names.json"
IMG_SIZE = 224
DISPLAY_NAMES = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "no_tumor": "No Tumor",
    "pituitary": "Pituitary",
}

_MODEL = None
_CLASS_NAMES: List[str] | None = None
_DEVICE = None
_TRANSFORM = None


class ModelLoadError(RuntimeError):
    """Raised when the trained model cannot be loaded."""


class InferenceError(RuntimeError):
    """Raised when inference fails."""


def get_device():
    """Return the active torch device, preferring CUDA when available."""
    if not TORCH_AVAILABLE:
        return "cpu"
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_device_name() -> str:
    """Return a human-readable runtime device name."""
    if not TORCH_AVAILABLE:
        return "torch-unavailable"
    device = get_device()
    if device.type == "cuda":
        return f"cuda:{torch.cuda.get_device_name(0)}"
    return "cpu"


def normalize_class_name(name: str) -> str:
    """Map dataset class ids to UI-friendly clinical labels."""
    key = name.strip().lower().replace(" ", "_").replace("-", "_")
    return DISPLAY_NAMES.get(key, name.strip().title())


def load_class_names() -> List[str]:
    """Load class names from models/class_names.json."""
    if not CLASS_NAMES_PATH.exists():
        raise ModelLoadError(f"Missing class_names.json at {CLASS_NAMES_PATH}")
    with CLASS_NAMES_PATH.open("r", encoding="utf-8") as fp:
        raw = json.load(fp)
    if isinstance(raw, dict):
        raw = raw.get("classes") or raw.get("class_names")
    if not isinstance(raw, list) or not raw:
        raise ModelLoadError("class_names.json must contain a non-empty class list.")
    return [normalize_class_name(str(name)) for name in raw]


def get_preprocess_transform():
    """Build ImageNet-compatible preprocessing for ResNet50."""
    if not TORCH_AVAILABLE:
        raise ModelLoadError("PyTorch and torchvision are not installed.")
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_inference_transforms(image_size: Tuple[int, int] | None = None):
    """Build deterministic test-time augmentation views for robust inference."""
    if not TORCH_AVAILABLE:
        raise ModelLoadError("PyTorch and torchvision are not installed.")

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    inference_transforms = [
        transforms.Compose(
            [
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        transforms.Compose(
            [
                transforms.Resize((IMG_SIZE, IMG_SIZE)),
                transforms.RandomHorizontalFlip(p=1.0),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    ]

    if image_size is None:
        return inference_transforms

    width, height = image_size
    aspect_ratio = max(width, height) / max(1, min(width, height))
    if aspect_ratio > 1.15:
        return inference_transforms

    inference_transforms.extend(
        [
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(IMG_SIZE),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(IMG_SIZE),
                    transforms.RandomHorizontalFlip(p=1.0),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        ]
    )
    return inference_transforms


def load_model(force_reload: bool = False):
    """Load the trained ResNet50 model and cache it for future requests."""
    global _MODEL, _CLASS_NAMES, _DEVICE, _TRANSFORM

    if _MODEL is not None and not force_reload:
        return _MODEL, _CLASS_NAMES, _DEVICE, _TRANSFORM

    if not TORCH_AVAILABLE:
        raise ModelLoadError("PyTorch is not installed. Install requirements.txt before running inference.")
    if not MODEL_PATH.exists():
        raise ModelLoadError(f"Trained model not found at {MODEL_PATH}")

    class_names = load_class_names()
    device = get_device()

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        state_dict = _extract_state_dict(checkpoint)
        state_dict = _strip_module_prefix(state_dict)
        model = models.resnet50(weights=None)
        model.fc = _build_resnet_fc_head(model.fc.in_features, len(class_names), state_dict)
        model.load_state_dict(state_dict, strict=True)
    except Exception as exc:
        raise ModelLoadError(f"Unable to load model weights: {exc}") from exc

    model.to(device)
    model.eval()

    _MODEL = model
    _CLASS_NAMES = class_names
    _DEVICE = device
    _TRANSFORM = get_preprocess_transform()
    LOGGER.info("Loaded ResNet50 model from %s on %s", MODEL_PATH, device)
    return _MODEL, _CLASS_NAMES, _DEVICE, _TRANSFORM


def _extract_state_dict(checkpoint: Any) -> Dict[str, Any]:
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            value = checkpoint.get(key)
            if isinstance(value, dict):
                return value
        if checkpoint and all(hasattr(v, "shape") for v in checkpoint.values()):
            return checkpoint
    raise ModelLoadError("Checkpoint does not contain a valid PyTorch state_dict.")


def _strip_module_prefix(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    return {key.replace("module.", "", 1): value for key, value in state_dict.items()}


def _build_resnet_fc_head(in_features: int, num_classes: int, state_dict: Dict[str, Any]):
    """Build the ResNet FC head shape used by the saved checkpoint.
    
    The training notebook uses: nn.Sequential(nn.Dropout(0.5), nn.Linear(num_features, 4))
    This function reconstructs that exact structure.
    """
    # Check for the training notebook's Sequential structure: fc.0 (Dropout), fc.1 (Linear)
    if "fc.1.weight" in state_dict and "fc.1.bias" in state_dict:
        # This is the Sequential(Dropout, Linear) structure from training
        out_features = int(state_dict["fc.1.weight"].shape[0])
        if out_features != num_classes:
            raise ModelLoadError(
                f"Checkpoint output classes ({out_features}) do not match class_names.json ({num_classes})."
            )
        return nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )
    
    # Fallback: check for simple Linear (fc.weight directly)
    if "fc.weight" in state_dict:
        out_features = int(state_dict["fc.weight"].shape[0])
        if out_features != num_classes:
            raise ModelLoadError(
                f"Checkpoint output classes ({out_features}) do not match class_names.json ({num_classes})."
            )
        return nn.Linear(in_features, num_classes)

    # Fallback: check for other Sequential structures
    linear_keys = sorted(
        key for key in state_dict
        if key.startswith("fc.") and key.endswith(".weight") and len(key.split(".")) == 3
    )
    if not linear_keys:
        raise ModelLoadError("Checkpoint does not contain a supported ResNet fc head.")

    layers = []
    current_features = in_features
    for key in linear_keys:
        layer_index = int(key.split(".")[1])
        weight = state_dict[key]
        out_features = int(weight.shape[0])
        expected_in = int(weight.shape[1])
        if expected_in != current_features:
            raise ModelLoadError(
                f"Checkpoint layer {key} expects {expected_in} input features, "
                f"but the inferred previous layer has {current_features}."
            )
        while len(layers) < layer_index:
            layers.append(nn.Identity())
        layers.append(nn.Linear(current_features, out_features))
        current_features = out_features

    final_out = int(state_dict[linear_keys[-1]].shape[0])
    if final_out != num_classes:
        raise ModelLoadError(
            f"Checkpoint output classes ({final_out}) do not match class_names.json ({num_classes})."
        )
    return nn.Sequential(*layers)


def preprocess_image(image: Image.Image):
    """Convert a PIL image to a batched tensor."""
    _, _, device, transform = load_model()
    tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    return tensor


def preprocess_inference_views(image: Image.Image):
    """Convert an image to batched deterministic inference views."""
    _, _, device, _ = load_model()
    rgb_image = image.convert("RGB")
    tensors = [transform(rgb_image) for transform in get_inference_transforms(rgb_image.size)]
    return torch.stack(tensors).to(device)


def predict_image(image: Image.Image) -> Dict[str, Any]:
    """Run model inference and return prediction, confidence, and probabilities."""
    try:
        model, class_names, _, _ = load_model()
        tensor = preprocess_inference_views(image)
        
        # Debug logging
        LOGGER.info(f"Input image size: {image.size}")
        LOGGER.info(f"Inference tensor shape: {tensor.shape}")
        
        with torch.no_grad():
            logits = model(tensor)
            probs_tensor = F.softmax(logits.mean(dim=0), dim=0).detach().cpu()

        probabilities = {
            class_name: round(float(prob) * 100.0, 4)
            for class_name, prob in zip(class_names, probs_tensor)
        }
        class_index = int(torch.argmax(probs_tensor).item())
        prediction = class_names[class_index]
        confidence = round(float(probs_tensor[class_index]) * 100.0, 4)

        # Debug logging
        LOGGER.info(f"Predicted class: {prediction} (index: {class_index})")
        LOGGER.info(f"Confidence: {confidence:.4f}%")
        LOGGER.info(f"Probabilities: {probabilities}")

        return {
            "prediction": prediction,
            "confidence": confidence,
            "probabilities": probabilities,
            "all_predictions": probabilities,
            "class_index": class_index,
            "model_predictions": {"ResNet-50": confidence},
        }
    except ModelLoadError:
        raise
    except Exception as exc:
        raise InferenceError(f"Model inference failed: {exc}") from exc


def get_loaded_model_context() -> Tuple[Any, List[str], Any, Any]:
    """Expose the cached model context for Grad-CAM."""
    return load_model()
