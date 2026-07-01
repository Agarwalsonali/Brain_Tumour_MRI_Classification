"""
Production-quality MRI validation pipeline with 5-layer validation.
Rejects non-medical images using computer vision and deep learning.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image

# Configure logging
logger = logging.getLogger(__name__)

# Natural image classes to reject (ImageNet classes)
NATURAL_IMAGE_CLASSES = {
    'cat', 'dog', 'bird', 'horse', 'fish', 'person', 'face', 'human',
    'car', 'bus', 'bicycle', 'motorcycle', 'flower', 'tree', 'fruit',
    'food', 'keyboard', 'laptop', 'phone', 'building', 'document'
}

# Global model cache for Layer 4
_natural_image_model = None
_natural_image_transform = None
_device = None


def _get_device() -> torch.device:
    """Get the appropriate device for model inference."""
    global _device
    if _device is None:
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return _device


def _load_natural_image_classifier():
    """
    Load a pretrained MobileNetV3 for natural image detection.
    Cached globally for performance.
    """
    global _natural_image_model, _natural_image_transform
    
    if _natural_image_model is not None:
        return _natural_image_model, _natural_image_transform
    
    try:
        device = _get_device()
        logger.info("Loading MobileNetV3 for natural image detection...")
        
        # Load pretrained MobileNetV3 Small (faster, good for CPU)
        model = torchvision.models.mobilenet_v3_small(pretrained=True)
        model.eval()
        model.to(device)
        
        # Standard ImageNet normalization
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        _natural_image_model = model
        _natural_image_transform = transform
        
        logger.info(f"MobileNetV3 loaded on {device}")
        return model, transform
        
    except Exception as e:
        logger.error(f"Failed to load natural image classifier: {e}")
        return None, None


def layer1_basic_validation(image: Image.Image) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Layer 1: Basic image validation.
    
    Checks:
    - File type/corrupt image detection
    - Image resolution check
    - Basic image integrity
    """
    checks = {}
    
    try:
        # Convert to array for analysis
        img_array = np.array(image)
        
        # Check if image is corrupted (all zeros or uniform)
        if img_array.size == 0:
            return False, "Image appears to be corrupted or empty", checks
        
        # Resolution check
        h, w = img_array.shape[:2]
        checks['resolution'] = f"{w}x{h}"
        
        if h < 128 or w < 128:
            return False, f"Resolution too low: {w}x{h}. Minimum required: 128x128", checks
        
        if h > 4096 or w > 4096:
            return False, f"Resolution too high: {w}x{h}. Maximum allowed: 4096x4096", checks
        
        # Check for uniform/corrupted images
        if len(img_array.shape) == 3:
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        else:
            gray = img_array
        
        if np.std(gray) < 2:
            return False, "Image appears to be corrupted or uniform", checks
        
        checks['corruption_check'] = 'passed'
        return True, "Basic validation passed", checks
        
    except Exception as e:
        logger.error(f"Layer 1 validation error: {e}")
        return False, f"Basic validation failed: {str(e)}", checks


def layer2_image_quality(image: Image.Image) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Layer 2: Image quality analysis.
    
    Checks:
    - Blur detection using Laplacian variance
    - Brightness analysis
    - Contrast analysis
    - Noise detection
    """
    checks = {}
    
    try:
        img_array = np.array(image.convert("RGB"))
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Blur detection using Laplacian variance
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        checks['laplacian_variance'] = round(lap_var, 2)
        
        # MRI images should have reasonable sharpness
        # Lowered threshold from 50 to 15 to accommodate MRI soft tissue characteristics
        if lap_var < 15:
            return False, f"Image too blurry (sharpness: {lap_var:.1f})", checks
        
        # Brightness analysis
        brightness = float(np.mean(gray))
        checks['brightness'] = round(brightness, 2)
        
        # MRI images typically have specific brightness ranges
        if brightness < 20:
            return False, f"Image too dark (brightness: {brightness:.1f})", checks
        if brightness > 240:
            return False, f"Image too bright (brightness: {brightness:.1f})", checks
        
        # Contrast analysis (standard deviation)
        contrast = float(np.std(gray))
        checks['contrast'] = round(contrast, 2)
        
        if contrast < 10:
            return False, f"Image has very low contrast (contrast: {contrast:.1f})", checks
        
        # Noise detection (high-frequency content)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.mean(edges > 0))
        checks['edge_density'] = round(edge_density, 4)
        
        # MRI should have reasonable edge density
        if edge_density < 0.005:
            return False, f"Image has insufficient detail (edge density: {edge_density:.4f})", checks
        
        return True, "Image quality check passed", checks
        
    except Exception as e:
        logger.error(f"Layer 2 validation error: {e}")
        return False, f"Image quality check failed: {str(e)}", checks


def layer3_mri_characteristics(image: Image.Image) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Layer 3: MRI-specific characteristics analysis.
    
    Checks:
    - Grayscale dominance
    - Histogram analysis
    - Intensity distribution
    - Background ratio (dark background typical in MRI)
    """
    checks = {}
    
    try:
        img_array = np.array(image.convert("RGB"))
        h, w = img_array.shape[:2]
        
        # Grayscale dominance check
        if len(img_array.shape) == 3:
            r, g, b = cv2.split(img_array)
            channel_delta = np.mean(np.maximum(np.maximum(r, g), b) - 
                                   np.minimum(np.minimum(r, g), b))
            grayscale_score = max(0.0, 1.0 - float(channel_delta) / 50.0)
            checks['grayscale_score'] = round(grayscale_score, 3)
            
            # MRI images should be predominantly grayscale
            if grayscale_score < 0.6:
                return False, f"Image appears to be color photograph (grayscale score: {grayscale_score:.2f})", checks
        
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Histogram analysis
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        
        # MRI typically has:
        # 1. Significant dark background (low intensities)
        # 2. Peak in mid-range intensities (brain tissue)
        # 3. Some high intensities (bright regions)
        
        # Background ratio (dark pixels)
        dark_ratio = float(np.sum(hist[:30]))
        checks['dark_background_ratio'] = round(dark_ratio, 3)
        
        if dark_ratio < 0.15:
            return False, f"Insufficient dark background typical of MRI (ratio: {dark_ratio:.2f})", checks
        
        # Mid-range tissue ratio
        tissue_ratio = float(np.sum(hist[30:200]))
        checks['tissue_ratio'] = round(tissue_ratio, 3)
        
        if tissue_ratio < 0.3:
            return False, f"Insufficient mid-range intensity tissue (ratio: {tissue_ratio:.2f})", checks
        
        # Bright regions ratio
        bright_ratio = float(np.sum(hist[200:]))
        checks['bright_ratio'] = round(bright_ratio, 3)
        
        # Intensity distribution skewness
        intensities = gray.flatten()
        skewness = float(((intensities - np.mean(intensities))**3).mean() / 
                        (np.std(intensities)**3 + 1e-10))
        checks['intensity_skewness'] = round(skewness, 3)
        
        # MRI typically has negative skewness (more dark pixels), but some valid MRIs
        # can have positive skewness due to augmentation or different acquisition parameters.
        # Relaxed threshold from 1.0 to 2.5 to accept valid MRIs with positive skewness.
        if skewness > 2.5:
            return False, f"Intensity distribution not typical of MRI (skewness: {skewness:.2f})", checks
        
        return True, "MRI characteristics check passed", checks
        
    except Exception as e:
        logger.error(f"Layer 3 validation error: {e}")
        return False, f"MRI characteristics check failed: {str(e)}", checks


def layer4_natural_image_detection(image: Image.Image) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Layer 4: Natural image detection using pretrained ImageNet model.
    
    Uses MobileNetV3 to detect if the image contains common natural objects
    like cats, dogs, people, cars, etc.
    """
    checks = {}
    
    try:
        model, transform = _load_natural_image_classifier()
        
        if model is None:
            # If model fails to load, skip this layer but log warning
            logger.warning("Natural image classifier not available, skipping Layer 4")
            checks['skipped'] = 'model_not_available'
            return True, "Natural image detection skipped (model unavailable)", checks
        
        device = _get_device()
        
        # Prepare image
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        # Inference
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        
        # Get top predictions
        top5_prob, top5_indices = torch.topk(probabilities, 5)
        
        # Load ImageNet class names (simplified subset)
        # In production, you'd load the full 1000-class mapping
        # For now, we'll use a heuristic based on common natural image indices
        
        # Common natural image class indices in ImageNet (simplified)
        natural_indices = {
            # Cats: 281-285
            281, 282, 283, 284, 285,
            # Dogs: 151-268
            *range(151, 269),
            # Birds: 80-100
            *range(80, 101),
            # Horses: 339-341
            339, 340, 341,
            # Fish: 389-397
            *range(389, 398),
            # People/face-related
            836, 837, 838, 839, 840, 841, 842, 843, 844, 845,
            # Vehicles
            817, 864, 865, 666, 667, 668, 669, 670, 671, 672, 673, 674, 675,
            # Flowers, trees, plants
            985, 986, 987, 988, 989, 990, 991, 992, 993, 994, 995, 996, 997,
            # Food
            *range(900, 969),
            # Electronics
            795, 796, 797, 798, 799, 800, 801, 802, 803, 804, 805, 806, 807,
            808, 809, 810, 811, 812, 813, 814, 815, 816,
        }
        
        top5_indices_list = top5_indices.cpu().numpy().tolist()
        top5_prob_list = top5_prob.cpu().numpy().tolist()
        
        checks['top5_predictions'] = [
            {'class_idx': int(idx), 'probability': round(float(prob), 4)}
            for idx, prob in zip(top5_indices_list, top5_prob_list)
        ]
        
        # Check if any top prediction is a natural image
        natural_image_detected = False
        detected_class = None
        max_natural_prob = 0.0
        
        for idx, prob in zip(top5_indices_list, top5_prob_list):
            # Increased threshold from 0.3 to 0.6 (60%) to reduce false rejections of valid MRIs
            if idx in natural_indices and prob > 0.6:
                natural_image_detected = True
                detected_class = int(idx)
                max_natural_prob = max(max_natural_prob, float(prob))
        
        if natural_image_detected:
            class_names = {
                281: 'tabby cat', 282: 'Egyptian cat', 283: 'fox',
                284: 'tiger', 285: 'lion',
                836: 'person', 837: 'face',
                817: 'car', 864: 'bicycle', 865: 'motorcycle',
                985: 'flower', 986: 'tree',
            }
            detected_name = class_names.get(detected_class, f'class_{detected_class}')
            return False, f"This image appears to be a {detected_name}, not a brain MRI", checks
        
        checks['natural_image_detected'] = False
        return True, "No natural image detected", checks
        
    except Exception as e:
        logger.error(f"Layer 4 validation error: {e}")
        # If this layer fails, we skip it rather than reject
        checks['error'] = str(e)
        return True, "Natural image detection skipped due to error", checks


def layer5_final_validation(image: Image.Image, all_checks: Dict[str, Any]) -> Tuple[bool, str, float]:
    """
    Layer 5: Final validation decision.
    
    Combines results from all previous layers to make a final decision.
    """
    try:
        # Calculate overall confidence score
        layer_scores = {
            'layer1': 1.0 if all_checks.get('layer1', {}).get('passed') else 0.0,
            'layer2': 1.0 if all_checks.get('layer2', {}).get('passed') else 0.0,
            'layer3': 1.0 if all_checks.get('layer3', {}).get('passed') else 0.0,
            'layer4': 1.0 if all_checks.get('layer4', {}).get('passed') else 0.0,
        }
        
        # Weight the layers (Layer 4 is most important)
        weights = {
            'layer1': 0.15,
            'layer2': 0.25,
            'layer3': 0.30,
            'layer4': 0.30,
        }
        
        overall_confidence = sum(
            layer_scores[layer] * weights[layer]
            for layer in layer_scores
        )
        
        # Threshold for acceptance
        if overall_confidence >= 0.75:
            return True, "Valid Brain MRI", overall_confidence
        else:
            return False, "This image does not appear to be a valid Brain MRI scan", overall_confidence
            
    except Exception as e:
        logger.error(f"Layer 5 validation error: {e}")
        return False, f"Final validation failed: {str(e)}", 0.0


def validate_brain_mri(image: Image.Image) -> Dict[str, Any]:
    """
    Main validation function that runs all 5 layers.
    
    Args:
        image: PIL Image object to validate
        
    Returns:
        Dictionary with validation result:
        {
            "valid": bool,
            "confidence": float,
            "reason": str,
            "layers": {
                "layer1": {...},
                "layer2": {...},
                "layer3": {...},
                "layer4": {...},
            },
            "validation_time": float
        }
    """
    start_time = time.time()
    
    result = {
        "valid": False,
        "confidence": 0.0,
        "reason": "",
        "layers": {},
        "validation_time": 0.0
    }
    
    try:
        # Layer 1: Basic validation
        layer1_passed, layer1_reason, layer1_checks = layer1_basic_validation(image)
        result['layers']['layer1'] = {
            'passed': layer1_passed,
            'reason': layer1_reason,
            'checks': layer1_checks
        }
        
        if not layer1_passed:
            result['valid'] = False
            result['confidence'] = 0.0
            result['reason'] = layer1_reason
            result['validation_time'] = time.time() - start_time
            return result
        
        # Layer 2: Image quality
        layer2_passed, layer2_reason, layer2_checks = layer2_image_quality(image)
        result['layers']['layer2'] = {
            'passed': layer2_passed,
            'reason': layer2_reason,
            'checks': layer2_checks
        }
        
        if not layer2_passed:
            result['valid'] = False
            result['confidence'] = 0.3
            result['reason'] = layer2_reason
            result['validation_time'] = time.time() - start_time
            return result
        
        # Layer 3: MRI characteristics
        layer3_passed, layer3_reason, layer3_checks = layer3_mri_characteristics(image)
        result['layers']['layer3'] = {
            'passed': layer3_passed,
            'reason': layer3_reason,
            'checks': layer3_checks
        }
        
        if not layer3_passed:
            result['valid'] = False
            result['confidence'] = 0.5
            result['reason'] = layer3_reason
            result['validation_time'] = time.time() - start_time
            return result
        
        # Layer 4: Natural image detection
        layer4_passed, layer4_reason, layer4_checks = layer4_natural_image_detection(image)
        result['layers']['layer4'] = {
            'passed': layer4_passed,
            'reason': layer4_reason,
            'checks': layer4_checks
        }
        
        if not layer4_passed:
            result['valid'] = False
            result['confidence'] = 0.95
            result['reason'] = layer4_reason
            result['validation_time'] = time.time() - start_time
            return result
        
        # Layer 5: Final decision
        layer5_passed, layer5_reason, confidence = layer5_final_validation(
            image, result['layers']
        )
        
        result['valid'] = layer5_passed
        result['confidence'] = round(confidence, 2)
        result['reason'] = layer5_reason
        result['validation_time'] = round(time.time() - start_time, 3)
        
        logger.info(f"MRI validation completed: valid={result['valid']}, "
                   f"confidence={result['confidence']:.2f}, "
                   f"time={result['validation_time']:.3f}s")
        
        return result
        
    except Exception as e:
        logger.error(f"MRI validation error: {e}")
        result['valid'] = False
        result['confidence'] = 0.0
        result['reason'] = f"Validation error: {str(e)}"
        result['validation_time'] = time.time() - start_time
        return result


# Backward compatibility alias for existing code
def validate_mri(image: Image.Image) -> Dict[str, Any]:
    """Alias for validate_brain_mri for backward compatibility."""
    result = validate_brain_mri(image)
    
    # Convert to the old format expected by app.py
    return {
        "accepted": result['valid'],
        "message": result['reason'],
        "score": result['confidence'] * 100,
        "checks": result.get('layers', {}),
    }
