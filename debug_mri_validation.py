"""
Debug script to test MRI validation on a specific image.
Run this to see which validation layer is rejecting your image.
"""

import sys
from pathlib import Path
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from utils.mri_validator import (
    layer1_basic_validation,
    layer2_image_quality,
    layer3_mri_characteristics,
    layer4_natural_image_detection,
    validate_brain_mri
)
from utils.image_processing import check_image_quality


def debug_validation(image_path):
    """Run each validation layer separately to identify the failure."""
    image_path = Path(image_path)
    
    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        return
    
    print(f"Testing image: {image_path}")
    print(f"{'='*70}")
    
    try:
        image = Image.open(image_path).convert('RGB')
        print(f"Image size: {image.size}")
        print(f"Mode: {image.mode}")
    except Exception as e:
        print(f"ERROR loading image: {e}")
        return
    
    print(f"\n{'='*70}")
    print("LAYER 1: Basic Validation (file type, corruption, resolution)")
    print(f"{'='*70}")
    passed, reason, checks = layer1_basic_validation(image)
    print(f"Result: {'✓ PASSED' if passed else '✗ FAILED'}")
    print(f"Reason: {reason}")
    print(f"Checks: {checks}")
    
    if not passed:
        print("\n❌ Image failed at Layer 1")
        return
    
    print(f"\n{'='*70}")
    print("LAYER 2: Image Quality (blur, brightness, contrast, noise)")
    print(f"{'='*70}")
    passed, reason, checks = layer2_image_quality(image)
    print(f"Result: {'✓ PASSED' if passed else '✗ FAILED'}")
    print(f"Reason: {reason}")
    print(f"Checks: {checks}")
    
    if not passed:
        print("\n❌ Image failed at Layer 2")
        return
    
    print(f"\n{'='*70}")
    print("LAYER 3: MRI Characteristics (grayscale, histogram, intensity)")
    print(f"{'='*70}")
    passed, reason, checks = layer3_mri_characteristics(image)
    print(f"Result: {'✓ PASSED' if passed else '✗ FAILED'}")
    print(f"Reason: {reason}")
    print(f"Checks: {checks}")
    
    if not passed:
        print("\n❌ Image failed at Layer 3")
        return
    
    print(f"\n{'='*70}")
    print("LAYER 4: Natural Image Detection (ImageNet classifier)")
    print(f"{'='*70}")
    passed, reason, checks = layer4_natural_image_detection(image)
    print(f"Result: {'✓ PASSED' if passed else '✗ FAILED'}")
    print(f"Reason: {reason}")
    print(f"Checks: {checks}")
    
    if not passed:
        print("\n❌ Image failed at Layer 4")
        return
    
    print(f"\n{'='*70}")
    print("FULL VALIDATION")
    print(f"{'='*70}")
    result = validate_brain_mri(image)
    print(f"Valid: {result['valid']}")
    print(f"Reason: {result['reason']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Time: {result['validation_time']}s")
    
    print(f"\n{'='*70}")
    print("IMAGE QUALITY CHECK (from image_processing.py)")
    print(f"{'='*70}")
    quality = check_image_quality(image)
    print(f"Suitable: {quality['suitable']}")
    print(f"Reason: {quality['reason']}")
    if quality.get('metrics'):
        print(f"Metrics: {quality['metrics']}")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Debug MRI validation on a single image')
    parser.add_argument('image', type=str, help='Path to image to debug')
    
    args = parser.parse_args()
    debug_validation(args.image)
