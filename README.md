# NeuroAI Brain Tumor MRI Classification

Production-oriented Flask and PyTorch application for classifying brain MRI images with a trained ResNet50 model, real Grad-CAM explainability, image quality gates, MRI validation, clinical summaries, AI second opinion text, and downloadable PDF reports.

## Architecture

```text
Brain_Tumour_MRI_Classification/
  app.py                         Flask API, static frontend serving, PDF generation
  index_v2.html                  Primary web UI
  models/
    best_model.pth               Trained ResNet50 weights
    class_names.json             Class order used during training
    model_config.json            Model metadata
  utils/
    model_loader.py              Model loading, preprocessing, inference
    image_processing.py          Upload decoding, extension checks, quality checks, base64 encoding
    gradcam.py                   Hook-based Grad-CAM for ResNet50
    mri_validator.py             Non-MRI rejection guardrail
    report_helper.py             Tumor knowledge base, clinical summary, second opinion
```

Request flow:

```text
Upload -> extension check -> decode image -> quality check -> MRI validation
       -> ResNet50 inference -> softmax probabilities -> Grad-CAM
       -> clinical summary + second opinion -> JSON response / PDF report
```

## Dataset

The model is designed for a four-class brain MRI classification task:

- Glioma
- Meningioma
- No Tumor
- Pituitary

The expected class order is stored in `models/class_names.json`. Keep this file aligned with the class index order used during training.

## Training

Training scripts are included for experimentation and reproducibility:

- `01_data_exploration.py`
- `02_train_models.py`
- `03_evaluate_models.py`
- `04_clinical_report.py`

The production API loads `models/best_model.pth` and does not simulate predictions. If you retrain the model, export the new weights to `models/best_model.pth` and update `models/class_names.json` if the class order changes.

## Model

- Architecture: ResNet50
- Input size: 224 x 224
- Normalization: ImageNet mean and standard deviation
- Runtime: CPU or CUDA, detected automatically
- Output: prediction, confidence, per-class softmax probabilities
- Explainability: Grad-CAM from the final ResNet layer block

## Features

- Real PyTorch inference from `best_model.pth`
- Real Grad-CAM heatmap and overlay generation
- Brain MRI validation before inference
- Professional quality checks for corrupted, dark, bright, blurred, or low-resolution images
- Dynamic clinical summary based on prediction and confidence
- Dynamic AI second opinion with severity, reasoning, next steps, follow-up, and disclaimer
- PDF report with MRI image, Grad-CAM, prediction, confidence, probability table, clinical summary, second opinion, disclaimer, and timestamp
- JSON API errors for empty uploads, invalid extensions, corrupted images, non-MRI images, missing model files, inference errors, and Grad-CAM errors
- Frontend upload path uses the selected `File` object directly so first-attempt uploads are reliable

## Screenshots

Add screenshots from your local run here:

```text
docs/screenshots/upload.png
docs/screenshots/prediction.png
docs/screenshots/report.png
```

## Installation

Create and activate a Python environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Start the Flask app:

```powershell
python app.py
```

Open the application:

```text
http://127.0.0.1:5000
```

## API

### Health

```http
GET /api/health
```

Returns runtime status, device, class names, Torch availability, confidence threshold, and timestamp.

### Prediction

```http
POST /api/predict
Content-Type: multipart/form-data
field: image
```

Successful response includes:

```json
{
  "prediction": "Glioma",
  "confidence": 91.23,
  "probabilities": {
    "Glioma": 91.23,
    "Meningioma": 3.21,
    "No Tumor": 2.11,
    "Pituitary": 3.45
  },
  "images": {
    "original": "base64_png",
    "heatmap": "base64_png",
    "overlay": "base64_png"
  },
  "quality_check": {},
  "mri_validation": {},
  "clinical_summary": {},
  "second_opinion": {},
  "timestamp": "2026-06-29T20:00:00"
}
```

### PDF Report

```http
POST /api/report
Content-Type: application/json
```

Accepts patient metadata plus the prediction payload and returns a downloadable PDF.

## Error Handling

The API returns structured JSON errors:

- `EMPTY_UPLOAD`
- `EMPTY_FILENAME`
- `INVALID_IMAGE`
- `IMAGE_QUALITY_FAILED`
- `NOT_BRAIN_MRI`
- `MODEL_LOAD_ERROR`
- `INFERENCE_ERROR`
- `GRADCAM_ERROR`
- `SERVER_ERROR`

Example:

```json
{
  "error": "This is not a Brain MRI",
  "code": "NOT_BRAIN_MRI",
  "mri_validation": {
    "accepted": false,
    "message": "This is not a Brain MRI"
  }
}
```

## Future Work

- Replace heuristic MRI validation with a dedicated medical-image classifier
- Add DICOM support with metadata extraction
- Add calibration metrics and uncertainty estimation
- Add model versioning and audit logging
- Add Docker packaging and CI tests
- Add protected clinical deployment mode with authentication

## Medical Disclaimer

This project is for research and educational support only. It is not a medical device, does not provide a diagnosis, and must not replace review by qualified healthcare professionals.
