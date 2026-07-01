"""Diagnostic script for checking Flask/PyTorch inference on dataset images."""

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from utils.image_processing import check_image_quality
from utils.model_loader import load_model, predict_image
from utils.mri_validator import validate_brain_mri


EXPECTED_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
DISPLAY_TO_DATASET_CLASS = {
    "glioma": "glioma",
    "meningioma": "meningioma",
    "no_tumor": "notumor",
    "notumor": "notumor",
    "pituitary": "pituitary",
}


def to_dataset_class(label):
    key = label.lower().replace(" ", "_").replace("-", "_")
    return DISPLAY_TO_DATASET_CLASS.get(key, key)


def test_on_directory(dataset_path, max_images_per_class=20, skip_gates=False):
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"ERROR: Dataset path not found: {dataset_path}")
        return

    try:
        _, class_names, _, transform = load_model()
        print("Model loaded successfully")
        print(f"Classes: {class_names}")
        print(f"Transform: {transform}")
    except Exception as exc:
        print(f"ERROR: Failed to load model: {exc}")
        return

    results = {
        "total_tested": 0,
        "correct": 0,
        "incorrect": 0,
        "validation_failed": 0,
        "quality_failed": 0,
        "by_class": {},
    }

    for class_name in EXPECTED_CLASSES:
        class_dir = dataset_path / class_name
        if not class_dir.exists():
            print(f"WARNING: Class directory not found: {class_dir}")
            continue

        print(f"\n{'=' * 60}\nTesting class: {class_name}\n{'=' * 60}")
        class_results = {"total": 0, "correct": 0, "incorrect": 0, "predictions": []}
        image_files = (
            list(class_dir.glob("*.jpg"))
            + list(class_dir.glob("*.jpeg"))
            + list(class_dir.glob("*.png"))
        )[:max_images_per_class]

        for img_path in image_files:
            try:
                image = Image.open(img_path).convert("RGB")

                if not skip_gates:
                    quality = check_image_quality(image)
                    if not quality["suitable"]:
                        print(f"  GATE {img_path.name} - Quality failed: {quality['reason']}")
                        results["quality_failed"] += 1
                        continue

                    mri_val = validate_brain_mri(image)
                    if not mri_val["valid"]:
                        print(f"  GATE {img_path.name} - MRI validation failed: {mri_val['reason']}")
                        results["validation_failed"] += 1
                        continue

                pred = predict_image(image)
                predicted_class = to_dataset_class(pred["prediction"])
                confidence = pred["confidence"]
                is_correct = predicted_class == class_name

                class_results["total"] += 1
                results["total_tested"] += 1
                if is_correct:
                    class_results["correct"] += 1
                    results["correct"] += 1
                    print(f"  OK   {img_path.name} - {predicted_class} ({confidence:.1f}%)")
                else:
                    class_results["incorrect"] += 1
                    results["incorrect"] += 1
                    print(
                        f"  FAIL {img_path.name} - Predicted: {predicted_class} "
                        f"({confidence:.1f}%), Actual: {class_name}"
                    )

                class_results["predictions"].append(
                    {
                        "image": str(img_path),
                        "predicted": predicted_class,
                        "actual": class_name,
                        "confidence": confidence,
                        "correct": is_correct,
                        "probabilities": pred["probabilities"],
                    }
                )
            except Exception as exc:
                print(f"  FAIL {img_path.name} - Error: {exc}")

        if class_results["total"] > 0:
            class_accuracy = class_results["correct"] / class_results["total"] * 100
            print(
                f"\nClass {class_name} accuracy: {class_accuracy:.1f}% "
                f"({class_results['correct']}/{class_results['total']})"
            )
            results["by_class"][class_name] = class_results

    print(f"\n{'=' * 60}\nOVERALL RESULTS\n{'=' * 60}")
    print(f"Total tested: {results['total_tested']}")
    print(f"Correct: {results['correct']}")
    print(f"Incorrect: {results['incorrect']}")
    print(f"Quality check failed: {results['quality_failed']}")
    print(f"MRI validation failed: {results['validation_failed']}")
    if results["total_tested"] > 0:
        print(f"\nOverall accuracy: {results['correct'] / results['total_tested'] * 100:.1f}%")

    results_path = BASE_DIR / "test_results.json"
    with results_path.open("w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2)
    print(f"\nResults saved to: {results_path}")


def test_single_image(image_path, skip_gates=False):
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}")
        return

    _, class_names, _, transform = load_model()
    print(f"Classes: {class_names}")
    print(f"Transform: {transform}")
    print(f"Testing image: {image_path}")
    print(f"{'=' * 60}")
    image = Image.open(image_path).convert("RGB")
    print(f"Image size: {image.size}")

    if not skip_gates:
        quality = check_image_quality(image)
        print(f"\nQuality check: {quality['reason']}")
        if quality.get("metrics"):
            print(f"  Metrics: {quality['metrics']}")
        if not quality["suitable"]:
            print("Image failed quality check")
            return

        mri_val = validate_brain_mri(image)
        print(f"\nMRI validation: {mri_val['reason']}")
        print(f"  Valid: {mri_val['valid']}")
        print(f"  Confidence: {mri_val['confidence']:.2f}")
        print(f"  Validation time: {mri_val['validation_time']:.3f}s")
        if not mri_val["valid"]:
            print("Image failed MRI validation")
            return

    pred = predict_image(image)
    print("\nPrediction:")
    print(f"  Class: {pred['prediction']}")
    print(f"  Dataset class: {to_dataset_class(pred['prediction'])}")
    print(f"  Confidence: {pred['confidence']:.2f}%")
    print(f"  Class index: {pred['class_index']}")
    print("\nProbabilities:")
    for class_name, prob in sorted(pred["probabilities"].items(), key=lambda x: x[1], reverse=True):
        print(f"  {class_name}: {prob:.2f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test brain tumor classification model")
    parser.add_argument("--image", type=str, help="Path to single image to test")
    parser.add_argument("--dataset", type=str, help="Path to dataset directory for batch testing")
    parser.add_argument("--max-per-class", type=int, default=20, help="Max images per class for batch testing")
    parser.add_argument("--skip-gates", action="store_true", help="Bypass quality and MRI validation gates")
    args = parser.parse_args()

    if args.image:
        test_single_image(args.image, args.skip_gates)
    elif args.dataset:
        test_on_directory(args.dataset, args.max_per_class, args.skip_gates)
    else:
        print("Usage:")
        print("  python test_model_accuracy.py --image path/to/image.jpg --skip-gates")
        print("  python test_model_accuracy.py --dataset path/to/dataset --max-per-class 20 --skip-gates")
