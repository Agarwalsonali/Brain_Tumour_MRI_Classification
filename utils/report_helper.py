"""Clinical knowledge base and dynamic report text helpers."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict


TUMOR_INFO: Dict[str, Dict[str, str]] = {
    "Glioma": {
        "description": (
            "Glioma is a primary brain tumor arising from glial cells. MRI patterns may show "
            "infiltrative margins, heterogeneous signal, edema, or mass effect depending on grade."
        ),
        "symptoms": (
            "Persistent headache, seizure, progressive weakness, speech difficulty, visual change, "
            "cognitive change, nausea, vomiting, or personality change."
        ),
        "next_steps": (
            "Urgent neurology or neurosurgery review, contrast-enhanced MRI correlation, biopsy or "
            "histopathology when clinically indicated, and multidisciplinary neuro-oncology planning."
        ),
        "severity": "High",
        "specialist": "Neuro-Oncologist / Neurosurgeon",
    },
    "Meningioma": {
        "description": (
            "Meningioma is usually an extra-axial tumor arising from the meninges. Many are benign, "
            "but location, size, edema, and compression determine clinical impact."
        ),
        "symptoms": (
            "Headache, seizure, focal weakness, sensory change, vision or hearing disturbance, memory "
            "difficulty, or incidental asymptomatic presentation."
        ),
        "next_steps": (
            "Neurosurgical evaluation, contrast MRI review, interval surveillance for small stable "
            "lesions, and treatment planning when there is growth or neurological compromise."
        ),
        "severity": "Low-Moderate",
        "specialist": "Neurosurgeon",
    },
    "No Tumor": {
        "description": (
            "The model did not identify imaging features consistent with the trained tumor classes. "
            "Clinical correlation remains essential, especially if neurological symptoms persist."
        ),
        "symptoms": (
            "No tumor-specific symptom pattern is inferred from this result. Persistent symptoms may "
            "require non-tumor neurological evaluation."
        ),
        "next_steps": (
            "Routine follow-up with the referring clinician. Repeat imaging or additional tests should "
            "be guided by symptoms, examination findings, and radiologist interpretation."
        ),
        "severity": "None",
        "specialist": "General Neurologist if symptoms persist",
    },
    "Pituitary": {
        "description": (
            "Pituitary adenomas arise in the sellar region and may affect endocrine function or compress "
            "nearby optic pathways depending on size and extension."
        ),
        "symptoms": (
            "Headache, visual field disturbance, menstrual irregularity, galactorrhea, fatigue, weight "
            "change, acromegalic features, or other hormone-related symptoms."
        ),
        "next_steps": (
            "Endocrinology review, pituitary hormone panel, visual field testing when indicated, and "
            "dedicated pituitary MRI correlation for treatment planning."
        ),
        "severity": "Low-Moderate",
        "specialist": "Endocrinologist / Neurosurgeon",
    },
}


DISCLAIMER = (
    "This AI-generated report is for research and educational support only. It is not a medical "
    "diagnosis and must be reviewed by qualified healthcare professionals."
)


def get_tumor_info(prediction: str) -> Dict[str, str]:
    """Return tumor metadata for a prediction label."""
    return TUMOR_INFO.get(prediction, TUMOR_INFO["No Tumor"])


def build_clinical_summary(prediction: str, confidence: float) -> Dict[str, Any]:
    """Create a dynamic clinical summary from the prediction and knowledge base."""
    info = get_tumor_info(prediction)
    confidence_band = _confidence_band(confidence)
    return {
        "title": f"{prediction} assessment",
        "prediction": prediction,
        "confidence": round(float(confidence), 2),
        "confidence_band": confidence_band,
        "severity": info["severity"],
        "description": info["description"],
        "symptoms": info["symptoms"],
        "next_steps": info["next_steps"],
        "summary": (
            f"The ResNet50 model predicts {prediction} with {confidence:.1f}% confidence "
            f"({confidence_band.lower()} certainty). Severity is categorized as {info['severity']}. "
            f"{info['next_steps']}"
        ),
        "generated_at": datetime.now().isoformat(),
        "disclaimer": DISCLAIMER,
    }


def build_second_opinion(prediction: str, confidence: float) -> Dict[str, Any]:
    """Generate severity, reasoning, next steps, and follow-up based on confidence."""
    info = get_tumor_info(prediction)
    band = _confidence_band(confidence)

    if prediction == "No Tumor":
        reasoning = (
            "The model did not find a tumor-class pattern with the strongest probability assigned "
            "to the no-tumor class. This should still be correlated with the radiology report and symptoms."
        )
    else:
        reasoning = (
            f"The probability distribution favors {prediction}. The confidence level is {confidence:.1f}%, "
            f"so this should be treated as a {band.lower()} AI finding and reviewed with the MRI sequences."
        )

    if confidence < 60:
        next_steps = (
            "Do not rely on this result alone. Obtain radiologist review, verify image quality and MRI sequence, "
            "and consider repeat analysis with the complete study."
        )
        follow_up = "Prompt clinical correlation is recommended because model confidence is limited."
    elif confidence < 80:
        next_steps = info["next_steps"]
        follow_up = "Schedule specialist review based on symptoms and radiologist confirmation."
    else:
        next_steps = info["next_steps"]
        follow_up = "Prioritize the recommended pathway; high AI confidence still requires clinician confirmation."

    return {
        "prediction": prediction,
        "confidence": round(float(confidence), 2),
        "confidence_band": band,
        "severity": info["severity"],
        "reasoning": reasoning,
        "severity_rationale": _severity_rationale(prediction, info["severity"], confidence),
        "symptoms": info["symptoms"],
        "specialist": info["specialist"],
        "recommended_tests": _recommended_tests(prediction, confidence),
        "next_steps": next_steps,
        "follow_up": follow_up,
        "disclaimer": DISCLAIMER,
    }


def _confidence_band(confidence: float) -> str:
    if confidence >= 85:
        return "High"
    if confidence >= 70:
        return "Moderate"
    return "Low"


def _severity_rationale(prediction: str, severity: str, confidence: float) -> str:
    if prediction == "No Tumor":
        return "No tumor class is currently favored by the model; clinical context remains important."
    return (
        f"{severity} severity is assigned from the tumor knowledge base. The {confidence:.1f}% confidence "
        "score indicates how strongly the model favors this class, not clinical certainty."
    )


def _recommended_tests(prediction: str, confidence: float) -> str:
    common = "Radiologist interpretation of the full MRI study"
    if prediction == "Glioma":
        tests = "Contrast MRI, MR spectroscopy/perfusion, and histopathology or molecular profiling when indicated"
    elif prediction == "Meningioma":
        tests = "Contrast MRI, CT for calcification or bone involvement, and interval imaging if observed"
    elif prediction == "Pituitary":
        tests = "Pituitary hormone panel, visual field testing, and dedicated pituitary MRI protocol"
    else:
        tests = "Clinical neurological workup only if symptoms persist"
    if confidence < 70:
        return f"{common}; repeat or higher-quality imaging should be considered because AI confidence is low. {tests}."
    return f"{common}. {tests}."
