import os
import io
import base64
import json
import time
import logging
import warnings
import traceback
import numpy as np
import cv2
from PIL import Image as PILImage
from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Image as RLImage, Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from utils.gradcam import GradCAMError, generate_gradcam as generate_real_gradcam
from utils.image_processing import (
    ImageValidationError,
    check_image_quality as run_image_quality_check,
    load_image_from_bytes,
    validate_extension,
)
from utils.model_loader import (
    IMG_SIZE,
    InferenceError,
    ModelLoadError,
    TORCH_AVAILABLE,
    get_device_name,
    load_class_names,
    predict_image,
)
from utils.mri_validator import validate_brain_mri
from utils.report_helper import (
    DISCLAIMER,
    build_clinical_summary as build_dynamic_clinical_summary,
    build_second_opinion as build_dynamic_second_opinion,
    get_tumor_info as get_dynamic_tumor_info,
)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def make_json_safe(value):
    """Convert NumPy scalars/arrays into native JSON-serializable Python values."""
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value

# ── Optional PyTorch ──────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    import torchvision.transforms as transforms
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ── Config ────────────────────────────────────────────────────────────────────
CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
IMG_SIZE    = 224
CONFIDENCE_THRESHOLD = 70.0
DEVICE = (torch.device('cuda' if torch.cuda.is_available() else 'cpu')
          if TORCH_AVAILABLE else 'cpu')

# ── Medical Knowledge Base ────────────────────────────────────────────────────
TUMOR_INFO = {
    'Glioma': {
        'description': (
            'Glioma is the most common primary brain tumor in adults, originating from '
            'glial (supportive) cells. Glioblastoma multiforme (GBM) is the most aggressive '
            'subtype and accounts for about 15% of all brain tumors.'
        ),
        'symptoms': (
            'Persistent headaches, nausea/vomiting, seizures, cognitive changes, '
            'progressive weakness on one side of the body, visual disturbances, '
            'speech difficulties, personality changes, memory loss.'
        ),
        'next_steps': (
            'Immediate neurosurgical consultation within 24–48 hours. MRI with contrast '
            'enhancement. Tissue biopsy for histological grading and molecular profiling '
            '(IDH mutation, MGMT methylation). Multidisciplinary neuro-oncology board review.'
        ),
        'severity': 'High',
        'specialist': 'Neuro-Oncologist / Neurosurgeon'
    },
    'Meningioma': {
        'description': (
            'Meningiomas arise from the meninges (the membranes surrounding the brain '
            'and spinal cord). Over 90% are benign (WHO Grade I), but location and size '
            'determine clinical significance. They are more common in women.'
        ),
        'symptoms': (
            'Headaches, seizures, vision changes, hearing loss, weakness in arms or legs, '
            'memory difficulties, personality changes. Many are asymptomatic and found '
            'incidentally on imaging.'
        ),
        'next_steps': (
            'Neurosurgical evaluation for surgical vs. conservative management. '
            'MRI with gadolinium contrast for detailed characterisation. '
            'Regular MRI surveillance (6–12 month intervals) if watchful waiting. '
            'Stereotactic radiosurgery for inaccessible or recurrent lesions.'
        ),
        'severity': 'Low-Moderate',
        'specialist': 'Neurosurgeon'
    },
    'No Tumor': {
        'description': (
            'No abnormal mass, pathological enhancement, or structural brain abnormality '
            'detected. Brain parenchyma, ventricles, and sulci appear within normal limits '
            'for the patient\'s age group.'
        ),
        'symptoms': (
            'No tumor-related symptoms identified. If symptoms persist despite a normal MRI, '
            'further workup (EEG, lumbar puncture, functional MRI) may be warranted.'
        ),
        'next_steps': (
            'Routine clinical follow-up with the referring physician. Maintain healthy '
            'lifestyle. Report any new or worsening neurological symptoms promptly. '
            'Repeat imaging only if clinically indicated.'
        ),
        'severity': 'None',
        'specialist': 'General Neurologist (if symptoms persist)'
    },
    'Pituitary': {
        'description': (
            'Pituitary adenomas originate from the anterior pituitary gland at the base '
            'of the brain. The vast majority are benign. They are classified as micro- '
            '(<10 mm) or macroadenomas (≥10 mm) and may be functioning (hormone-secreting) '
            'or non-functioning.'
        ),
        'symptoms': (
            'Hormonal imbalances (acromegaly, Cushing\'s disease, hyperprolactinemia), '
            'visual field defects (bitemporal hemianopia), headaches, fatigue, '
            'weight changes, menstrual irregularities, galactorrhoea, growth abnormalities.'
        ),
        'next_steps': (
            'Endocrinology consultation for hormone panel assessment. '
            'Formal visual field testing (perimetry). Dynamic pituitary MRI protocol. '
            'Treatment options: dopamine agonists (prolactinoma), somatostatin analogues '
            '(acromegaly), or transsphenoidal surgical resection.'
        ),
        'severity': 'Low-Moderate',
        'specialist': 'Endocrinologist / Neurosurgeon'
    }
}

SECOND_OPINION = {
    'Glioma': {
        'reasoning': (
            'The model identified an irregular, heterogeneously enhancing mass with '
            'surrounding vasogenic oedema and mass effect. These features — infiltrative '
            'margins, ring enhancement pattern, and peritumoral T2/FLAIR signal — are '
            'hallmarks of high-grade glial neoplasia.'
        ),
        'severity_rationale': (
            'High severity. Gliomas, particularly GBM (WHO Grade IV), are aggressive '
            'with median survival of 14–16 months even with standard therapy. Rapid '
            'progression and blood-brain barrier disruption demand urgent intervention.'
        ),
        'additional_tests': (
            'MRI perfusion imaging (DSC/ASL), MR spectroscopy (Cho:Cr ratio), '
            'diffusion tensor imaging (DTI) for eloquent cortex mapping, '
            'stereotactic biopsy for IDH/ATRX/1p19q molecular profiling, '
            'FET-PET for metabolic assessment'
        ),
        'follow_up': (
            'Urgent neurosurgical consultation (within 24–48 h). Consider maximal safe '
            'surgical resection followed by Stupp protocol (temozolomide + RT). '
            'Bevacizumab for recurrent disease. Clinical trial enrolment if eligible.'
        )
    },
    'Meningioma': {
        'reasoning': (
            'A well-circumscribed, extra-axial dural-based mass with homogeneous '
            'contrast enhancement and a "dural tail" sign was identified. These '
            'characteristics are pathognomonic for meningioma and distinguish it '
            'from other extra-axial lesions.'
        ),
        'severity_rationale': (
            'Low to moderate severity. Most meningiomas are WHO Grade I (benign) with '
            'excellent long-term prognosis. Clinical significance depends on location '
            '(skull base vs. convexity), size, and rate of growth.'
        ),
        'additional_tests': (
            'MRI with gadolinium (thin-slice through region of interest), '
            'CT for bony hyperostosis or invasion, '
            'cerebral angiography if highly vascular tumour anticipated pre-operatively'
        ),
        'follow_up': (
            'Neurosurgical consultation for watchful waiting vs. Simpson Grade I resection. '
            'Annual MRI surveillance initially, then every 2 years if stable. '
            'Stereotactic radiosurgery (SRS/FSRT) for small or residual lesions.'
        )
    },
    'No Tumor': {
        'reasoning': (
            'No focal intracranial lesion, abnormal enhancement, mass effect, or '
            'midline shift detected. White matter signal, cortical gyration, '
            'ventricular size, and posterior fossa structures appear unremarkable.'
        ),
        'severity_rationale': (
            'No tumour detected. Findings are reassuring, however clinical correlation '
            'with the patient\'s symptom history, neurological examination, and laboratory '
            'results is always essential to exclude non-structural pathology.'
        ),
        'additional_tests': (
            'No further tumour workup required. If neurological symptoms persist: '
            'EEG, CSF analysis (lumbar puncture), functional MRI, neuropsychological testing'
        ),
        'follow_up': (
            'Routine clinical follow-up with the referring physician. '
            'Repeat imaging only if symptoms worsen or new focal deficits emerge.'
        )
    },
    'Pituitary': {
        'reasoning': (
            'A sellar/suprasellar mass with characteristic T1 iso/hypointense signal '
            'and avid gadolinium enhancement was identified. The relationship to the '
            'optic chiasm and cavernous sinuses is important for surgical planning. '
            'Signal heterogeneity may indicate haemorrhage within the adenoma.'
        ),
        'severity_rationale': (
            'Low to moderate severity. The majority of pituitary adenomas are benign '
            'and treatable. However, hormone hypersecretion and optic chiasm compression '
            'can cause significant systemic and visual morbidity if untreated.'
        ),
        'additional_tests': (
            'Full anterior pituitary hormone panel (GH, IGF-1, prolactin, ACTH, '
            'morning cortisol, TSH, LH, FSH), '
            'formal Goldmann perimetry, dynamic gadolinium pituitary MRI protocol, '
            'inferior petrosal sinus sampling if Cushing\'s suspected'
        ),
        'follow_up': (
            'Endocrinology consultation for hormone replacement if hypopituitary. '
            'Cabergoline/bromocriptine for prolactinoma. '
            'Transsphenoidal surgery for macroadenomas or failed medical therapy. '
            'Post-operative MRI at 3 months, then annually.'
        )
    }
}

# ── Image Quality Check ───────────────────────────────────────────────────────
def check_image_quality(image):
    """Validate MRI image suitability before inference."""
    try:
        img_array = np.array(image)
        h, w = img_array.shape[:2]

        if h < 128 or w < 128:
            return {
                'suitable': False,
                'reason': 'Resolution too low',
                'details': f'Image is {w}×{h} px — minimum required is 128×128 px.'
            }

        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        if lap_var < 100.0:
            return {
                'suitable': False,
                'reason': 'Image appears blurry',
                'details': (
                    f'Sharpness score {lap_var:.1f} is below the minimum threshold of 100. '
                    'Try uploading a higher-quality scan.'
                )
            }

        brightness = float(np.mean(gray))
        if brightness < 30:
            return {
                'suitable': False,
                'reason': 'Image too dark',
                'details': f'Mean brightness {brightness:.1f} is below the minimum of 30.'
            }
        if brightness > 225:
            return {
                'suitable': False,
                'reason': 'Image too bright / over-exposed',
                'details': f'Mean brightness {brightness:.1f} exceeds the maximum of 225.'
            }

        if np.std(gray) < 5:
            return {
                'suitable': False,
                'reason': 'Image may be corrupted',
                'details': 'Very low pixel variance suggests a uniform or corrupted image.'
            }

        return {
            'suitable': True,
            'reason': 'Image quality acceptable',
            'details': f'{w}×{h} px · sharpness {lap_var:.0f} · brightness {brightness:.0f}',
            'metrics': {
                'resolution': f'{w}×{h}',
                'sharpness':  round(lap_var, 1),
                'brightness': round(brightness, 1)
            }
        }
    except Exception as exc:
        return {'suitable': False, 'reason': 'Quality check error', 'details': str(exc)}


# ── Simulation Prediction ─────────────────────────────────────────────────────
 


# ── Grad-CAM ──────────────────────────────────────────────────────────────────
 


# ── Second Opinion Generator ──────────────────────────────────────────────────
 


# ── Helpers ───────────────────────────────────────────────────────────────────
def bgr_to_b64(img_bgr):
    _, buf = cv2.imencode('.png', img_bgr)
    return base64.b64encode(buf).decode('utf-8')

def rgb_to_b64(img_rgb):
    bgr = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)
    return bgr_to_b64(bgr)


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/api/predict', methods=['POST'])
def predict():
    """Analyse uploaded MRI and return full clinical inference payload."""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided', 'code': 'EMPTY_UPLOAD'}), 400

        file = request.files['image']
        if not file.filename:
            return jsonify({'error': 'Empty filename', 'code': 'EMPTY_FILENAME'}), 400

        validate_extension(file.filename)
        raw = file.read()
        image = load_image_from_bytes(raw)

        # Quality gate
        quality = run_image_quality_check(image)
        if not quality['suitable']:
            return jsonify({
                'error': 'Image quality check failed',
                'code': 'IMAGE_QUALITY_FAILED',
                'quality_check': quality
            }), 400

        # MRI validation gate
        mri_validation = validate_brain_mri(image)
        if not mri_validation['accepted']:
            return jsonify({
                'error': 'This is not a Brain MRI',
                'code': 'NOT_BRAIN_MRI',
                'quality_check': quality,
                'mri_validation': mri_validation
            }), 400

        # Inference
        t0     = time.perf_counter()
        result = predict_image(image)
        ms     = round((time.perf_counter() - t0) * 1000, 2)

        # Grad-CAM
        gradcam = generate_real_gradcam(image, result.get('class_index'))

        # Payload assembly
        confidence    = result['confidence']
        tumor_info    = get_dynamic_tumor_info(result['prediction'])
        clinical_summary = build_dynamic_clinical_summary(result['prediction'], confidence)
        second_opinion = build_dynamic_second_opinion(result['prediction'], confidence)

        payload = {
            **result,
            'images': {
                'original': gradcam['original'],
                'heatmap':  gradcam['heatmap'],
                'overlay':  gradcam['overlay'],
            },
            'gradcam': {
                'original': gradcam['original'],
                'heatmap':  gradcam['heatmap'],
                'overlay':  gradcam['overlay'],
            },
            'timestamp':          datetime.now().isoformat(),
            'inference_time':     ms,
            'quality_check':      quality,
            'mri_validation':     mri_validation,
            'is_low_confidence':  confidence < CONFIDENCE_THRESHOLD,
            'confidence_threshold': CONFIDENCE_THRESHOLD,
            'clinical_summary':   clinical_summary,
            'second_opinion':     second_opinion,
            'tumor_info':         tumor_info,
        }
        return jsonify(make_json_safe(payload))

    except ImageValidationError as exc:
        return jsonify({'error': str(exc), 'code': 'INVALID_IMAGE'}), 400
    except ModelLoadError as exc:
        logging.exception("Model loading failed")
        return jsonify({'error': str(exc), 'code': 'MODEL_LOAD_ERROR'}), 503
    except InferenceError as exc:
        logging.exception("Inference failed")
        return jsonify({'error': str(exc), 'code': 'INFERENCE_ERROR'}), 500
    except GradCAMError as exc:
        logging.exception("Grad-CAM failed")
        return jsonify({'error': str(exc), 'code': 'GRADCAM_ERROR'}), 500
    except Exception as exc:
        logging.exception("Unexpected prediction error")
        return jsonify({'error': 'Unexpected server error during analysis.', 'details': str(exc), 'code': 'SERVER_ERROR'}), 500


@app.route('/api/report', methods=['POST'])
def generate_report():
    """Render a hospital-style PDF clinical report."""
    try:
        data   = request.json or {}
        pat    = lambda k, d='N/A': data.get(k) or d
        styles = getSampleStyleSheet()

        # ── PDF document ─────────────────────────────────────────────────────
        buf = io.BytesIO()
        doc = SimpleDocTemplate(
            buf, pagesize=letter,
            topMargin=0.6*inch, bottomMargin=0.6*inch,
            leftMargin=0.75*inch, rightMargin=0.75*inch
        )
        story = []

        TEAL  = colors.HexColor('#0f766e')
        TEAL2 = colors.HexColor('#ccfbf1')
        DARK  = colors.HexColor('#0f172a')
        MUTED = colors.HexColor('#64748b')
        RED   = colors.HexColor('#dc2626')

        def heading(text, sz=14, clr=TEAL, spaceb=16, spacea=6):
            return Paragraph(text, ParagraphStyle(
                'H', parent=styles['Normal'],
                fontSize=sz, textColor=clr,
                fontName='Helvetica-Bold',
                spaceBefore=spaceb, spaceAfter=spacea
            ))

        def body(text, sz=10.5):
            return Paragraph(text, ParagraphStyle(
                'B', parent=styles['Normal'],
                fontSize=sz, textColor=DARK,
                leading=15, spaceAfter=6, alignment=TA_JUSTIFY
            ))

        def label(text):
            return Paragraph(text, ParagraphStyle(
                'L', parent=styles['Normal'],
                fontSize=8.5, textColor=MUTED,
                fontName='Helvetica', spaceAfter=2
            ))

        def hr():
            return HRFlowable(width='100%', thickness=0.5,
                              color=colors.HexColor('#e2e8f0'),
                              spaceAfter=10, spaceBefore=2)

        # ── Cover header ─────────────────────────────────────────────────────
        story.append(Paragraph(
            '<b>NEUROAI</b> — Clinical MRI Intelligence Platform',
            ParagraphStyle('Title', parent=styles['Normal'],
                           fontSize=20, textColor=TEAL,
                           fontName='Helvetica-Bold',
                           alignment=TA_CENTER, spaceAfter=4)
        ))
        story.append(Paragraph(
            'AI-Powered Brain Tumour Diagnosis Report',
            ParagraphStyle('Sub', parent=styles['Normal'],
                           fontSize=11, textColor=MUTED,
                           alignment=TA_CENTER, spaceAfter=20)
        ))
        story.append(hr())

        # ── Patient info table ────────────────────────────────────────────────
        story.append(heading('Patient Information'))
        pi = [
            ['Name',       pat('patient_name')],
            ['Patient ID', pat('patient_id')],
            ['Age',        str(pat('age'))],
            ['Gender',     pat('gender')],
            ['Report Date', datetime.now().strftime('%d %B %Y')],
            ['Report Time', datetime.now().strftime('%H:%M')],
            ['Analysis ID', f"NAI-{datetime.now().strftime('%Y%m%d%H%M%S')}"],
        ]
        pt = Table(pi, colWidths=[1.6*inch, 4.2*inch])
        pt.setStyle(TableStyle([
            ('BACKGROUND',  (0,0), (0,-1), TEAL2),
            ('FONTNAME',    (0,0), (0,-1), 'Helvetica-Bold'),
            ('FONTNAME',    (1,0), (1,-1), 'Helvetica'),
            ('FONTSIZE',    (0,0), (-1,-1), 9.5),
            ('ALIGN',       (0,0), (-1,-1), 'LEFT'),
            ('VALIGN',      (0,0), (-1,-1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0,0), (-1,-1), [colors.white, colors.HexColor('#f8fafc')]),
            ('GRID',        (0,0), (-1,-1), 0.4, colors.HexColor('#e2e8f0')),
            ('BOTTOMPADDING',(0,0),(-1,-1), 7),
            ('TOPPADDING',  (0,0), (-1,-1), 7),
            ('LEFTPADDING', (0,0), (-1,-1), 10),
        ]))
        story.append(pt)
        story.append(Spacer(1, 0.2*inch))
        story.append(hr())

        # ── Diagnosis result ──────────────────────────────────────────────────
        story.append(heading('AI Diagnosis Result'))
        prediction = data.get('prediction', 'Unknown')
        confidence = data.get('confidence', 0)
        tumor_payload = data.get('tumor_info') or get_dynamic_tumor_info(prediction)
        sev        = tumor_payload.get('severity', '—')

        dr = [
            ['Predicted Condition', prediction],
            ['AI Confidence',       f"{confidence:.1f}%"],
            ['Severity Assessment', sev],
            ['Analysis Timestamp',  data.get('timestamp') or datetime.now().isoformat()],
        ]
        dt = Table(dr, colWidths=[1.6*inch, 4.2*inch])
        dt.setStyle(TableStyle([
            ('BACKGROUND',  (0,0), (0,-1), TEAL2),
            ('FONTNAME',    (0,0), (0,-1), 'Helvetica-Bold'),
            ('FONTNAME',    (1,0), (1,-1), 'Helvetica-Bold'),
            ('FONTSIZE',    (0,0), (-1,-1), 10),
            ('ALIGN',       (0,0), (-1,-1), 'LEFT'),
            ('VALIGN',      (0,0), (-1,-1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0,0), (-1,-1), [TEAL2, colors.white]),
            ('GRID',        (0,0), (-1,-1), 0.4, colors.HexColor('#e2e8f0')),
            ('BOTTOMPADDING',(0,0),(-1,-1), 8),
            ('TOPPADDING',  (0,0), (-1,-1), 8),
            ('LEFTPADDING', (0,0), (-1,-1), 10),
        ]))
        story.append(dt)
        story.append(Spacer(1, 0.2*inch))
        probabilities = data.get('probabilities') or data.get('all_predictions') or {}
        if probabilities:
            story.append(heading('Probability Table', sz=12))
            rows = [['Class', 'Probability']]
            for cls, prob in sorted(probabilities.items(), key=lambda item: item[1], reverse=True):
                rows.append([str(cls), f"{float(prob):.2f}%"])
            prob_table = Table(rows, colWidths=[3.0*inch, 2.8*inch])
            prob_table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), TEAL),
                ('TEXTCOLOR', (0,0), (-1,0), colors.white),
                ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
                ('FONTNAME', (0,1), (-1,-1), 'Helvetica'),
                ('FONTSIZE', (0,0), (-1,-1), 9.5),
                ('ALIGN', (1,1), (1,-1), 'RIGHT'),
                ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8fafc')]),
                ('GRID', (0,0), (-1,-1), 0.4, colors.HexColor('#e2e8f0')),
                ('BOTTOMPADDING', (0,0), (-1,-1), 7),
                ('TOPPADDING', (0,0), (-1,-1), 7),
            ]))
            story.append(prob_table)
            story.append(Spacer(1, 0.2*inch))
        story.append(hr())

        # ── Images ───────────────────────────────────────────────────────────
        def add_b64_image(title, b64str, w=3.2*inch):
            if not b64str:
                return
            story.append(heading(title, sz=12))
            try:
                raw = base64.b64decode(b64str)
                img = RLImage(io.BytesIO(raw), width=w, height=w)
                img.hAlign = 'CENTER'
                story.append(img)
                story.append(Spacer(1, 0.15*inch))
            except Exception:
                pass

        add_b64_image('MRI Scan',                      data.get('image'))
        add_b64_image('Grad-CAM Attention Map',        data.get('gradcam'))

        story.append(hr())

        # ── Clinical sections ─────────────────────────────────────────────────
        ti = tumor_payload
        clinical_summary = data.get('clinical_summary') or build_dynamic_clinical_summary(prediction, float(confidence or 0))
        story.append(heading('Clinical Summary'))
        story.append(body(clinical_summary.get('summary', '—')))
        for title, key in [
            ('Clinical Observation', 'description'),
            ('Possible Symptoms',    'symptoms'),
            ('Recommended Next Steps', 'next_steps'),
        ]:
            story.append(heading(title))
            story.append(body(ti.get(key, '—')))

        story.append(hr())

        # ── AI Second Opinion ─────────────────────────────────────────────────
        so = data.get('second_opinion', {})
        if so:
            story.append(heading('AI Second Opinion', sz=15))
            for lbl, key in [
                ('AI Reasoning',          'reasoning'),
                ('Severity Rationale',    'severity_rationale'),
                ('Recommended Specialist','specialist'),
                ('Recommended Tests',     'recommended_tests'),
                ('Follow-up Plan',        'follow_up'),
            ]:
                story.append(label(lbl.upper()))
                story.append(body(so.get(key, '—')))
            story.append(Spacer(1, 0.1*inch))
            story.append(hr())

        # ── Disclaimer ────────────────────────────────────────────────────────
        story.append(heading('Medical Disclaimer', clr=RED))
        story.append(Paragraph(
            '<b>IMPORTANT:</b> This report is generated by an artificial intelligence '
            'system and is intended for research and educational purposes only. '
            'It does <b>not</b> constitute a medical diagnosis and should <b>not</b> '
            'be used as a substitute for professional medical advice, diagnosis, or '
            'treatment. Always consult a qualified healthcare professional for any '
            'medical decisions.',
            ParagraphStyle('Dis', parent=styles['Normal'],
                           fontSize=9, textColor=RED,
                           leading=13, spaceAfter=8)
        ))

        # ── Footer ────────────────────────────────────────────────────────────
        story.append(Spacer(1, 0.4*inch))
        story.append(Paragraph(
            '© 2026 NeuroAI Clinical MRI Intelligence Platform',
            ParagraphStyle('Foot', parent=styles['Normal'],
                           fontSize=8.5, textColor=MUTED,
                           alignment=TA_CENTER)
        ))

        doc.build(story)
        buf.seek(0)
        return send_file(
            buf,
            as_attachment=True,
            download_name=f'neuroai_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf',
            mimetype='application/pdf'
        )

    except Exception as exc:
        print(traceback.format_exc())
        return jsonify({'error': str(exc)}), 500


@app.route('/api/health', methods=['GET'])
def health():
    try:
        classes = load_class_names()
    except Exception:
        classes = CLASS_NAMES
    return jsonify({
        'status':          'healthy',
        'version':         '2.0.0',
        'device':          get_device_name(),
        'torch_available': TORCH_AVAILABLE,
        'classes':         classes,
        'confidence_threshold': CONFIDENCE_THRESHOLD,
        'timestamp':       datetime.now().isoformat()
    })


@app.route('/', methods=['GET'])
def index():
    try:
        return send_file(os.path.join(BASE_DIR, 'index_v2.html'))
    except FileNotFoundError:
        return jsonify({
            'message':   'NeuroAI API is running.',
            'version':   '2.0.0',
            'endpoints': {
                'POST /api/predict': 'Upload MRI for prediction',
                'POST /api/report':  'Generate PDF report',
                'GET  /api/health':  'Health check'
            }
        })


@app.route('/<path:filename>')
def serve_static(filename):
    try:
        return send_from_directory(BASE_DIR, filename)
    except FileNotFoundError:
        return jsonify({'error': 'File not found'}), 404


if __name__ == '__main__':
    print("=" * 60)
    print("  NeuroAI — Clinical MRI Intelligence Platform v2.0")
    print("=" * 60)
    print(f"  Device  : {DEVICE}")
    print(f"  PyTorch : {'✓' if TORCH_AVAILABLE else '✗ (simulation mode)'}")
    print(f"  URL     : http://localhost:5000")
    print("=" * 60)
    app.run(debug=True, host='0.0.0.0', port=5000)
