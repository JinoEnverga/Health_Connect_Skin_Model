# ============================================================
#  api.py — Flask REST API for Skin Disease Prediction
#
#  Serves best_model_final.tflite (a quantized export of the
#  EfficientNetB3 transfer-learning model trained in Colab on
#  the skin-disease-datasaet/ dataset — 8 classes of bacterial,
#  fungal, parasitic & viral skin infections).
#
#  Runs on the lightweight ai-edge-litert interpreter instead of
#  full TensorFlow, which is what makes this fit inside Render's
#  512MB free-tier memory limit — full TF + the raw .keras model
#  alone can exceed that before a single request is served.
#
#  Files expected alongside this script:
#    - best_model_final.tflite
#    - class_labels.txt
#
#  RUN LOCALLY:
#    pip install -r requirements.txt
#    python api.py
#    -> served at http://localhost:5000
# ============================================================

import io
import logging
import os
import threading

import numpy as np
from ai_edge_litert.interpreter import Interpreter
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image

# ── Config ──────────────────────────────────────────────────
MODEL_PATH  = os.environ.get('MODEL_PATH', 'best_model_final.tflite')
LABELS_PATH = os.environ.get('LABELS_PATH', 'class_labels.txt')
IMG_SIZE    = (224, 224)
MAX_FILE_MB = 10

ARCHITECTURE = 'EfficientNetB3 (Transfer Learning)'
DATASET_NAME = 'Skin Disease Dataset — bacterial, fungal, parasitic & viral infections'

# Human-readable risk info shown alongside a prediction.
# Codes must match class_labels.txt exactly.
RISK_MAP = {
    'ba-cellulitis':              {'level': 'HIGH',     'color': '#dc2626', 'advice': 'See a doctor promptly — cellulitis is a bacterial infection that can spread quickly and may need oral or IV antibiotics.'},
    'vi-shingles':                {'level': 'HIGH',     'color': '#dc2626', 'advice': 'See a doctor as soon as possible — early antiviral treatment reduces the risk of complications, especially if the rash is near the eyes.'},
    'ba-impetigo':                {'level': 'MODERATE', 'color': '#d97706', 'advice': 'Consult a doctor. Impetigo is a contagious bacterial infection that usually clears up with topical or oral antibiotics.'},
    'pa-cutaneous-larva-migrans': {'level': 'MODERATE', 'color': '#d97706', 'advice': 'See a doctor. This parasitic skin infection typically requires antiparasitic medication to clear.'},
    'vi-chickenpox':              {'level': 'MODERATE', 'color': '#d97706', 'advice': 'Consult a doctor, especially for adults, infants, or anyone immunocompromised. Chickenpox is contagious and usually resolves on its own in healthy children.'},
    'fu-ringworm':                {'level': 'LOW',      'color': '#16a34a', 'advice': 'Usually treatable with over-the-counter antifungal cream. See a doctor if it spreads or does not improve in a few weeks.'},
    'fu-athlete-foot':            {'level': 'LOW',      'color': '#16a34a', 'advice': 'Usually treatable with over-the-counter antifungal cream and good foot hygiene. See a doctor if it persists or worsens.'},
    'fu-nail-fungus':             {'level': 'LOW',      'color': '#16a34a', 'advice': 'Not dangerous but can be persistent. See a doctor or podiatrist for antifungal treatment options.'},
}

# ── Setup ───────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # allow requests from your frontend (e.g. React on localhost:5173)

interpreter   = None
input_index   = None
output_index  = None
predict_lock  = threading.Lock()  # tflite Interpreter isn't safe for concurrent invoke() calls

label_list: list[str] = []        # e.g. ['ba-cellulitis', 'ba-impetigo', ...] — index order = model output order
class_names: dict[str, str] = {}  # e.g. {'ba-cellulitis': 'Bacterial Infection - Cellulitis', ...}


def load_labels(path: str) -> None:
    """Parse class_labels.txt (format: `code|Readable Name` per line)."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Labels file not found: {path}")

    label_list.clear()
    class_names.clear()
    with open(path, encoding='utf-8') as f:
        for line in f.read().strip().splitlines():
            code, name = line.split('|', 1)
            code = code.strip()
            label_list.append(code)
            class_names[code] = name.strip()

    logger.info("Loaded %d classes: %s", len(label_list), label_list)


def load_model(path: str):
    """Load the .tflite model into a TFLite interpreter."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    interp = Interpreter(model_path=path)
    interp.allocate_tensors()
    logger.info("Model loaded from %s", path)
    return interp


def preprocess_image(file_bytes: bytes) -> np.ndarray:
    """Decode + resize an uploaded image into the model's expected input.

    NOTE: best_model_final.tflite bundles EfficientNetB3's own preprocessing
    (Rescaling + ImageNet Normalization) as part of the graph, so raw 0-255
    pixel values must be passed in here — do NOT divide by 255.
    """
    img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)  # shape: (1, 224, 224, 3)


def run_inference(arr: np.ndarray) -> np.ndarray:
    """Run one forward pass through the TFLite interpreter and return softmax probs."""
    with predict_lock:
        interpreter.set_tensor(input_index, arr)
        interpreter.invoke()
        return interpreter.get_tensor(output_index)[0]


def build_prediction(probs: np.ndarray) -> dict:
    """Turn a raw softmax vector into the API's response payload."""
    top_idx  = int(np.argmax(probs))
    top_code = label_list[top_idx]
    top_conf = float(probs[top_idx])
    risk     = RISK_MAP.get(top_code, {})

    all_scores = sorted(
        (
            {
                'code':       code,
                'label':      class_names.get(code, code),
                'confidence': round(float(probs[i]) * 100, 2),
            }
            for i, code in enumerate(label_list)
        ),
        key=lambda x: x['confidence'],
        reverse=True,
    )

    return {
        'prediction': {
            'code':       top_code,
            'label':      class_names.get(top_code, top_code),
            'confidence': round(top_conf * 100, 2),
            'risk_level': risk.get('level', 'UNKNOWN'),
            'risk_color': risk.get('color', '#6b7280'),
            'advice':     risk.get('advice', 'Please consult a doctor or dermatologist.'),
        },
        'all_scores': all_scores,
        'disclaimer': (
            'This result is AI-generated and is NOT a medical diagnosis. '
            'Always consult a qualified doctor or dermatologist for proper evaluation.'
        ),
        'model_info': {
            'architecture': ARCHITECTURE,
            'dataset':      DATASET_NAME,
            'classes':      len(label_list),
        },
    }


# ── Load model & labels once, at startup ───────────────────
load_labels(LABELS_PATH)
interpreter  = load_model(MODEL_PATH)
input_index  = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']


# ─────────────────────────────────────────────────────────────
# POST /predict — multipart/form-data with an 'image' file field
# ─────────────────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided. Use field name "image".'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename.'}), 400

    file_bytes = file.read()
    if len(file_bytes) > MAX_FILE_MB * 1024 * 1024:
        return jsonify({'error': f'File too large. Max size: {MAX_FILE_MB}MB'}), 413

    try:
        arr   = preprocess_image(file_bytes)
        probs = run_inference(arr)
    except Exception as e:
        logger.error("Prediction failed: %s", e)
        return jsonify({'error': f'Could not process the image: {e}'}), 500

    response = build_prediction(probs)
    logger.info(
        "Predicted: %s (%s) — confidence %.1f%%",
        response['prediction']['code'],
        response['prediction']['label'],
        response['prediction']['confidence'],
    )
    return jsonify(response), 200


# ─────────────────────────────────────────────────────────────
# GET /health — simple health check
# ─────────────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':  'running',
        'model':   os.path.basename(MODEL_PATH),
        'classes': len(label_list),
        'labels':  class_names,
    }), 200


# ─────────────────────────────────────────────────────────────
# GET /classes — list all supported skin conditions
# ─────────────────────────────────────────────────────────────
@app.route('/classes', methods=['GET'])
def get_classes():
    return jsonify({
        'classes': [
            {
                'code':  code,
                'label': class_names[code],
                'risk':  RISK_MAP.get(code, {}).get('level', 'UNKNOWN'),
            }
            for code in label_list
        ]
    }), 200


if __name__ == '__main__':
    print("\n" + "=" * 55)
    print("  Skin Disease API Server Starting...")
    print("=" * 55 + "\n")

    port = int(os.environ.get('PORT', 5000))  # Render provides PORT; default 5000 locally
    app.run(host='0.0.0.0', port=port, debug=False)
