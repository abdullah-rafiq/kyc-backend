
import os
import io
import json
import base64
import sys
import re
import hashlib
import urllib.request
import numpy as np
import cv2
from flask import Flask, request, jsonify
from ultralytics import YOLO
from paddleocr import PaddleOCR
from deepface import DeepFace
from PIL import Image

app = Flask(__name__)

# -------------------- Model Initialization --------------------
# We load models at startup so they remain in memory (Free Spaces have 16GB RAM, ample space)
print("Loading KYC Models...", file=sys.stderr)

# 1. OCR
ocr_en = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
ocr_ar = PaddleOCR(use_angle_cls=True, lang='arabic', show_log=False)

# 2. YOLO
def _load_yolo_model():
    model_ref = os.environ.get('SHOP_YOLO_MODEL', 'yolov8n.pt')
    ref = str(model_ref or 'yolov8n.pt').strip()

    if ref.startswith('http://') or ref.startswith('https://'):
        h = hashlib.sha256(ref.encode('utf-8')).hexdigest()[:16]
        local_path = os.path.join('/tmp', f'shop_yolo_{h}.pt')
        if not os.path.exists(local_path):
            urllib.request.urlretrieve(ref, local_path)
        return YOLO(local_path)

    if ref and os.path.exists(ref):
        return YOLO(ref)

    return YOLO('yolov8n.pt')


yolo_model = _load_yolo_model()

# 3. DeepFace: Lazy loaded usually, but we can verify it loads by a dummy call or just wait.
# It downloads weights to ~/.deepface/weights on first run.

print("Models loaded!", file=sys.stderr)

URDU_RE = re.compile(r'[\u0600-\u06FF]')

SERVICE_KEYWORDS = {
    'electrician': [
        'electrician', 'electrical', 'wiring', 'wire', 'switch', 'socket',
        'breaker', 'mcb', 'fuse', 'voltage', 'amp', 'multimeter',
        'bijli', 'electric', 'bijlee',
        'الیکٹریشن', 'الیکٹرک', 'بجلی', 'وائرنگ', 'سوئچ', 'ساکٹ',
    ],
    'plumber': [
        'plumber', 'plumbing', 'pipe', 'pvc', 'tap', 'faucet', 'valve',
        'drain', 'sewer', 'leak', 'geyser',
        'nal', 'paani', 'pipe fitting',
        'پلمبر', 'پلمبنگ', 'پائپ', 'نل', 'پانی', 'لیک', 'گیزر',
    ],
    'cleaner': [
        'cleaner', 'cleaning', 'wash', 'washing', 'soap', 'detergent',
        'mop', 'broom', 'vacuum', 'sanitary',
        'safai', 'saaf', 'safayi',
        'صفائی', 'صاف', 'کلینر', 'جھاڑو', 'پوچا',
    ],
    'barber': [
        'barber', 'salon', 'saloon', 'hair', 'beard', 'shave', 'trimmer',
        'clipper', 'scissors', 'razor',
        'nai', 'hajjam',
        'نائی', 'حجام', 'حجامت', 'سیلون', 'بال', 'داڑھی',
    ],
    'ac_repair': [
        'ac', 'a.c', 'air conditioner', 'airconditioning', 'inverter',
        'cooling', 'gas', 'refrigerant', 'freon', 'compressor',
        'hvac',
        'اے سی', 'ایئر کنڈیشنر', 'کولنگ', 'گیس',
    ],
    'carpenter': [
        'carpenter', 'carpentry', 'wood', 'furniture', 'door', 'cabinet',
        'plywood', 'board', 'saw', 'drill', 'hinge',
        'barhai', 'lakri',
        'بڑھئی', 'کارپنٹر', 'لکڑی', 'فرنیچر', 'دروازہ',
    ],
    'painter': [
        'painter', 'painting', 'paint', 'roller', 'brush', 'primer',
        'putty', 'wall', 'color', 'colour',
        'rang',
        'پینٹر', 'پینٹنگ', 'رنگ', 'برش',
    ],
}

SERVICE_TOOL_HINTS = {
    'electrician': [
        'multimeter', 'tester', 'screwdriver', 'pliers', 'wire', 'cutter',
        'switchboard', 'mcb', 'breaker', 'socket',
    ],
    'plumber': [
        'pipe', 'pvc_pipe', 'pipe_wrench', 'wrench', 'spanner', 'plier', 'pliers',
        'valve', 'tap', 'faucet',
    ],
    'cleaner': [
        'mop', 'broom', 'vacuum', 'bucket', 'detergent',
    ],
    'barber': [
        'scissors', 'clipper', 'hair_clipper', 'trimmer', 'razor', 'comb',
        'barber_chair',
    ],
    'ac_repair': [
        'ac', 'ac_outdoor_unit', 'ac_indoor_unit', 'compressor', 'gauge',
        'gauge_manifold', 'refrigerant', 'vacuum_pump',
    ],
    'carpenter': [
        'saw', 'drill', 'hammer', 'nail', 'nails', 'hinge', 'wood',
        'measuring_tape',
    ],
    'painter': [
        'roller', 'paint_roller', 'brush', 'paint_brush', 'paint', 'paint_bucket',
        'putty_knife',
    ],
}


def score_services(ocr_text, tools):
    tokens = [str(t or '').lower() for t in (ocr_text or [])]
    joined = ' '.join(tokens)
    tool_tokens = [str(t or '').lower() for t in (tools or [])]
    tool_joined = ' '.join(tool_tokens)

    scores = {}
    for service_id, kws in SERVICE_KEYWORDS.items():
        kws_l = [str(k).lower() for k in kws]
        kw_hits = sum(1 for k in kws_l if k and k in joined)
        kw_score = min(1.0, kw_hits / 2.0)

        hint_tools = [str(t).lower() for t in SERVICE_TOOL_HINTS.get(service_id, [])]
        tool_hits = sum(1 for k in hint_tools if k and (k in tool_joined or k in joined))
        tool_score = min(1.0, tool_hits / 2.0)

        scores[service_id] = round((0.75 * kw_score) + (0.25 * tool_score), 3)

    top = max(scores.items(), key=lambda kv: kv[1]) if scores else (None, 0.0)
    top_service = top[0] if top[1] >= 0.6 else None
    ranked = [k for k, v in sorted(scores.items(), key=lambda kv: kv[1], reverse=True) if v > 0]
    return scores, top_service, ranked

# -------------------- Helper Functions --------------------

def decode_image(image_input):
    """Decodes image from base64 string or file bytes."""
    try:
        # If it's a base64 string
        if isinstance(image_input, str) and "," in image_input:
            image_input = image_input.split(",")[1]
        
        if isinstance(image_input, str):
            img_bytes = base64.b64decode(image_input)
        else:
            img_bytes = image_input # Assume bytes

        img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        raise ValueError(f"Invalid image input: {e}")

def crop_card(image):
    results = yolo_model(image, verbose=False)
    best_box = None
    max_area = 0
    
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                best_box = (int(x1), int(y1), int(x2), int(y2))
    
    if best_box:
        x1, y1, x2, y2 = best_box
        h, w, _ = image.shape
        x1 = max(0, x1 - 10)
        y1 = max(0, y1 - 10)
        x2 = min(w, x2 + 10)
        y2 = min(h, y2 + 10)
        return image[y1:y2, x1:x2]
    
    return image


def preprocess_for_ocr(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _paddle_lines(ocr, image_bgr):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    out = ocr.ocr(rgb, cls=True) or []

    pages = out
    if out and len(out) >= 1 and isinstance(out[0], (list, tuple)):
        first = out[0]
        if len(first) == 2 and isinstance(first[1], (list, tuple)):
            pages = [out]

    lines = []
    for page in pages:
        for item in page or []:
            if not item or len(item) < 2:
                continue
            text_conf = item[1]
            if isinstance(text_conf, (list, tuple)) and len(text_conf) >= 1:
                text = str(text_conf[0] or '').strip()
                if text:
                    lines.append(text)
    return lines


def extract_cnic_info(text_lines):
    info = {
        "fullName": None,
        "fatherName": None,
        "cnicNumber": None,
        "dateOfBirth": None,
        "dateOfIssue": None,
        "dateOfExpiry": None,
        "addressUrdu": {
            "line1": None,
            "line2": None,
            "district": None,
            "tehsil": None,
        },
    }

    if not text_lines:
        return info

    lines = [str(t or "").strip() for t in text_lines if str(t or "").strip()]
    joined = " ".join(lines)

    m = re.search(r"\b\d{5}-\d{7}-\d\b", joined)
    if m:
        info["cnicNumber"] = m.group(0)
    else:
        m2 = re.search(r"\b\d{13}\b", joined)
        if m2:
            d = m2.group(0)
            info["cnicNumber"] = f"{d[:5]}-{d[5:12]}-{d[12:]}"

    dates = re.findall(r"\b(\d{2}[-/\.]\d{2}[-/\.]\d{4})\b", joined)
    if len(dates) >= 1:
        info["dateOfBirth"] = dates[0]
    if len(dates) >= 2:
        info["dateOfIssue"] = dates[1]
    if len(dates) >= 3:
        info["dateOfExpiry"] = dates[2]

    header_phrases = [
        "PAKISTAN",
        "ISLAMIC REPUBLIC OF PAKISTAN",
        "NATIONAL IDENTITY CARD",
        "GOVERNMENT OF PAKISTAN",
    ]

    def is_header_line(s):
        up = str(s or "").upper()
        return any(p in up for p in header_phrases)

    candidates = []
    for ln in lines:
        if not ln or is_header_line(ln):
            continue
        if not any(ch.isalpha() for ch in ln):
            continue
        if sum(ch.isdigit() for ch in ln) > 2:
            continue
        if len(ln.split()) < 2:
            continue
        candidates.append(ln)

    latin = [ln for ln in candidates if not URDU_RE.search(ln)]
    if latin:
        info["fullName"] = latin[0]
        if len(latin) > 1:
            info["fatherName"] = latin[1]
    elif candidates:
        info["fullName"] = candidates[0]
        if len(candidates) > 1:
            info["fatherName"] = candidates[1]

    urdu_lines = [ln for ln in lines if URDU_RE.search(ln)]
    if urdu_lines:
        info["addressUrdu"]["line1"] = urdu_lines[0]
        if len(urdu_lines) > 1:
            info["addressUrdu"]["line2"] = urdu_lines[1]

    return info

# -------------------- Endpoints --------------------

@app.route('/', methods=['GET'])
def health():
    return "KYC Engine is Running"

@app.route('/verify-cnic', methods=['POST'])
def verify_cnic():
    try:
        data = request.json or {}
        image_data = data.get('image')
        if not image_data:
            return jsonify({"error": "No image provided"}), 400

        img = decode_image(image_data)
        cropped = crop_card(img)

        ocr_img = preprocess_for_ocr(cropped)
        ocr_bgr = cv2.cvtColor(ocr_img, cv2.COLOR_GRAY2BGR)
        lines_en = _paddle_lines(ocr_en, ocr_bgr)
        lines_ar = _paddle_lines(ocr_ar, ocr_bgr)
        text_results = [*lines_en, *[t for t in lines_ar if t not in lines_en]]

        extracted = extract_cnic_info(text_results)

        return jsonify({
            "rawText": text_results,
            "raw_text": text_results,
            **extracted,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/face-verify', methods=['POST'])
def face_verify():
    try:
        data = request.json or {}
        img1_data = data.get('image1') or data.get('image')
        img2_data = data.get('image2')
        
        if not img1_data or not img2_data:
            return jsonify({"error": "image1 and image2 required"}), 400

        # DeepFace expects paths or numpy arrays (BGR)
        img1 = decode_image(img1_data)
        img2 = decode_image(img2_data)

        try:
            result = DeepFace.verify(
                img1_path=img1,
                img2_path=img2,
                model_name='ArcFace',
                detector_backend='retinaface',
                enforce_detection=False,
            )
        except Exception:
            result = DeepFace.verify(
                img1_path=img1,
                img2_path=img2,
                model_name='VGG-Face',
                detector_backend='opencv',
                enforce_detection=False,
            )

        distance = float(result.get('distance', 0.0))
        threshold = float(result.get('threshold', 1.0) or 1.0)
        verified = bool(result.get('verified', False))

        confidence = 0.0
        if threshold > 0:
            confidence = 1.0 - (distance / threshold)
            confidence = max(0.0, min(1.0, confidence))

        return jsonify({
            "verified": verified,
            "distance": distance,
            "threshold": threshold,
            "confidence": confidence,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/face-liveness', methods=['POST'])
def face_liveness():
    try:
        data = request.json or {}
        image_data = data.get('image')
        if not image_data:
            return jsonify({"error": "No image provided"}), 400

        img = decode_image(image_data)

        try:
            faces = DeepFace.extract_faces(
                img_path=img,
                detector_backend='retinaface',
                enforce_detection=False,
                anti_spoofing=True,
            )

            face0 = faces[0] if isinstance(faces, list) and len(faces) > 0 else None
            is_real_raw = None
            score_raw = None
            if isinstance(face0, dict):
                is_real_raw = face0.get('is_real', face0.get('isReal'))
                score_raw = face0.get(
                    'antispoof_score',
                    face0.get('anti_spoof_score', face0.get('spoof_score', face0.get('spoofScore'))),
                )

            is_real = bool(is_real_raw) if isinstance(is_real_raw, bool) else None

            liveness_score = None
            if isinstance(score_raw, (int, float)):
                s = float(score_raw)
                if s > 1.0:
                    s = min(1.0, s / 100.0)

                if is_real is False:
                    liveness_score = max(0.0, min(1.0, 1.0 - s))
                else:
                    liveness_score = max(0.0, min(1.0, s))
            elif is_real is True:
                liveness_score = 1.0
            elif is_real is False:
                liveness_score = 0.0

            return jsonify({
                'livenessScore': liveness_score,
                'isReal': is_real,
                'spoofModel': 'deepface_antispoof',
                'method': 'deepface_extract_faces',
            })
        except Exception:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            lap = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness = float(lap.var())
            sharpness_score = max(0.0, min(1.0, sharpness / 900.0))

            mean = float(gray.mean())
            brightness_score = max(0.0, min(1.0, (mean - 30.0) / 170.0))

            liveness_score = max(0.0, min(1.0, 0.65 * sharpness_score + 0.35 * brightness_score))
            return jsonify({
                'livenessScore': round(liveness_score, 3),
                'isReal': None,
                'spoofModel': 'heuristic',
                'method': 'sharpness_brightness',
            })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/shop-verify', methods=['POST'])
def shop_verify():
    try:
        data = request.json or {}
        image_data = data.get('image')
        if not image_data:
            return jsonify({"error": "No image provided"}), 400
        
        img = decode_image(image_data)

        h, w = img.shape[:2]
        shape = 'portrait' if h >= w else 'landscape'
        
        conf_threshold = float(os.environ.get('SHOP_YOLO_CONF', '0.4') or 0.4)
        max_objects = int(os.environ.get('SHOP_MAX_OBJECTS', '50') or 50)

        # Detect objects
        results = yolo_model(img, verbose=False)
        detected_objects = []
        for r in results:
            for box in r.boxes:
                 cls_id = int(box.cls[0])
                 cls_name = yolo_model.names[cls_id]
                 conf = float(box.conf[0])
                 if conf >= conf_threshold:
                     detected_objects.append(cls_name)
                     if len(detected_objects) >= max_objects:
                         break
            if len(detected_objects) >= max_objects:
                break
        
        # OCR on shop doc/signboard
        lines_en = _paddle_lines(ocr_en, img)
        lines_ar = _paddle_lines(ocr_ar, img)
        ocr_text = [*lines_en, *[t for t in lines_ar if t not in lines_en]]

        tool_keywords = [
            'wrench', 'spanner', 'screwdriver', 'hammer', 'plier', 'pliers',
            'drill', 'cutter', 'saw', 'welder', 'welding', 'tool',
            'electrician', 'plumber', 'mechanic',
        ]
        ocr_joined = ' '.join([t.lower() for t in ocr_text])
        tools_from_text = [k for k in tool_keywords if k in ocr_joined]

        unique_objects = sorted(set(detected_objects))
        tools = sorted(set([*unique_objects, *tools_from_text]))

        exclude_env = os.environ.get(
            'SHOP_TOOL_EXCLUDE',
            'person,car,bus,truck,motorcycle,bicycle,bench,chair,table,cell phone,laptop',
        )
        excluded = {t.strip() for t in str(exclude_env).split(',') if t.strip()}
        tools = [t for t in tools if t not in excluded]

        allow_env = os.environ.get('SHOP_TOOL_ALLOWLIST')
        if allow_env and str(allow_env).strip():
            allowed = {t.strip() for t in str(allow_env).split(',') if t.strip()}
            tools = [t for t in tools if t in allowed]

        service_hints, top_service, service_candidates = score_services(ocr_text, tools)
        object_factor = min(1.0, len(unique_objects) / 5.0)
        text_factor = 0.3 if len(ocr_text) > 3 else 0.0
        keyword_boost = 0.2 if tools_from_text else 0.0
        score = max(0.0, min(1.0, object_factor + text_factor + keyword_boost))
        status = 'auto_verified' if score >= 0.6 else 'needs_manual_review'

        notes = []
        if not unique_objects:
            notes.append('no_objects_detected')
        if not ocr_text:
            notes.append('no_text_detected')
        
        return jsonify({
            "detected_objects": unique_objects,
            "detectedObjects": unique_objects,
            "tools": tools,
            "text_content": ocr_text,
            "textContent": ocr_text,
            "ocrText": ocr_text,
            "shape": shape,
            "score": score,
            "status": status,
            "notes": notes,
            "serviceHints": service_hints,
            "topService": top_service,
            "serviceCandidates": service_candidates,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Run on port 7860 (Hugging Face default)
    app.run(host='0.0.0.0', port=7860)
