# ============================================================
#  api.py  — Flask REST API for Skin Disease Prediction
#  Place this file in your skin_model/ folder alongside
#  skin_model.h5 and class_labels.txt
#
#  HOW TO RUN:
#    pip install flask flask-cors tensorflow pillow
#    python api.py
#
#  The server starts at: http://localhost:5000
#  Your React app calls: http://localhost:5000/predict
# ============================================================

import os
import io
import json
import logging
import numpy as np

from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

# Lazy-load TensorFlow to speed up startup
import tensorflow as tf
from tensorflow import keras

# ── Setup ───────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Allow requests from your React app on localhost:5173

# ── Config ──────────────────────────────────────────────────
MODEL_PATH  = 'skin_model.h5'
LABELS_PATH = 'class_labels.txt'
IMG_SIZE    = (224, 224)
MAX_FILE_MB = 10

# ── Load model & class labels on startup ────────────────────
model       = None
label_list  = []    # e.g. ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
class_names = {}    # e.g. {'mel': 'Melanoma', ...}

def load_model_and_labels():
    global model, label_list, class_names

    # Load labels
    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(
            f"Labels file not found: {LABELS_PATH}\n"
            "Run train_model.py first to generate it."
        )
    with open(LABELS_PATH) as f:
        for line in f.read().strip().splitlines():
            code, name = line.split('|', 1)
            label_list.append(code.strip())
            class_names[code.strip()] = name.strip()
    logger.info(f"Loaded {len(label_list)} classes: {label_list}")

    # Load Keras model
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}\n"
            "Run train_model.py first to generate it."
        )
    model = keras.models.load_model(MODEL_PATH)
    logger.info("Model loaded successfully!")

# Call on startup
with app.app_context():
    load_model_and_labels()


# ─────────────────────────────────────────────────────────────
# Helper — preprocess an uploaded image
# ─────────────────────────────────────────────────────────────
def preprocess_image(file_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(file_bytes)).convert('RGB')
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)     # shape: (1, 224, 224, 3)


# ─────────────────────────────────────────────────────────────
# Helper — build a human-readable risk level
# ─────────────────────────────────────────────────────────────
RISK_MAP = {
    'mel':   {'level': 'HIGH',     'color': '#dc2626', 'advice': 'See a dermatologist urgently — melanoma can be life-threatening if not treated early.'},
    'bcc':   {'level': 'MODERATE', 'color': '#d97706', 'advice': 'Schedule a dermatology appointment soon. Basal cell carcinoma is treatable when caught early.'},
    'akiec': {'level': 'MODERATE', 'color': '#d97706', 'advice': 'Consult a dermatologist. Actinic keratosis can progress to cancer if untreated.'},
    'nv':    {'level': 'LOW',      'color': '#16a34a', 'advice': 'Melanocytic nevi are usually benign. Monitor for changes (size, color, shape) and visit a dermatologist annually.'},
    'bkl':   {'level': 'LOW',      'color': '#16a34a', 'advice': 'Benign keratosis is generally harmless. See a dermatologist if it changes or becomes irritated.'},
    'df':    {'level': 'LOW',      'color': '#16a34a', 'advice': 'Dermatofibroma is benign. No treatment needed unless it causes discomfort.'},
    'vasc':  {'level': 'LOW',      'color': '#16a34a', 'advice': 'Vascular lesions are usually benign. Consult a dermatologist if the appearance changes.'},
}


# ─────────────────────────────────────────────────────────────
# POST /predict
# Accepts: multipart/form-data with an 'image' file field
# Returns: JSON with predicted class, confidence, and all scores
# ─────────────────────────────────────────────────────────────
@app.route('/predict', methods=['POST'])
def predict():
    # ── Validate request ──────────────────────────────────────
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided. Use field name "image".'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'Empty filename.'}), 400

    file_bytes = file.read()
    if len(file_bytes) > MAX_FILE_MB * 1024 * 1024:
        return jsonify({'error': f'File too large. Max size: {MAX_FILE_MB}MB'}), 413

    # ── Preprocess & predict ──────────────────────────────────
    try:
        arr   = preprocess_image(file_bytes)
        probs = model.predict(arr, verbose=0)[0]
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return jsonify({'error': f'Could not process the image: {str(e)}'}), 500

    # ── Build response ────────────────────────────────────────
    top_idx  = int(np.argmax(probs))
    top_code = label_list[top_idx]
    top_conf = float(probs[top_idx])
    risk     = RISK_MAP.get(top_code, {})

    # All class scores sorted by confidence
    all_scores = sorted(
        [
            {
                'code':       code,
                'label':      class_names.get(code, code),
                'confidence': round(float(probs[i]) * 100, 2),
            }
            for i, code in enumerate(label_list)
        ],
        key=lambda x: x['confidence'],
        reverse=True,
    )

    response = {
        'prediction': {
            'code':         top_code,
            'label':        class_names.get(top_code, top_code),
            'confidence':   round(top_conf * 100, 2),
            'risk_level':   risk.get('level', 'UNKNOWN'),
            'risk_color':   risk.get('color', '#6b7280'),
            'advice':       risk.get('advice', 'Please consult a dermatologist.'),
        },
        'all_scores':   all_scores,
        'disclaimer':   (
            'This result is AI-generated and is NOT a medical diagnosis. '
            'Always consult a qualified dermatologist for proper evaluation.'
        ),
        'model_info': {
            'architecture': 'EfficientNetB3 (Transfer Learning)',
            'dataset':      'HAM10000 — 10,015 dermoscopy images',
            'classes':      len(label_list),
        },
    }

    logger.info(
        f"Predicted: {top_code} ({class_names.get(top_code)}) "
        f"— confidence {top_conf*100:.1f}%"
    )
    return jsonify(response), 200


# ─────────────────────────────────────────────────────────────
# GET /health  — simple health check
# ─────────────────────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status':  'running',
        'model':   'skin_model.h5',
        'classes': len(label_list),
        'labels':  class_names,
    }), 200


# ─────────────────────────────────────────────────────────────
# GET /classes  — list all supported skin conditions
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


# ─────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("\n" + "=" * 55)
    print("  Skin Disease API  —  http://localhost:5000")
    print("  Endpoints:")
    print("    POST /predict  — analyze a skin image")
    print("    GET  /health   — server status")
    print("    GET  /classes  — list all conditions")
    print("=" * 55 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
