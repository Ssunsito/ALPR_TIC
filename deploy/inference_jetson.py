# -*- coding: utf-8 -*-
"""
INFERENCE EN JETSON NANO - Deteccion de Placas en Tiempo Real
Optimizado para 2GB RAM con camara nativa
"""

import cv2
import numpy as np
import os
try:
    import tensorflow as tf
except Exception:
    tf = None
import time
import argparse
import re
import shutil
from pathlib import Path
from collections import deque
import logging
import threading
from runtime_config import CONFIG, apply_runtime_profile
from ocr_inference import OCRInference

# Importar post-procesador OCR
try:
    from ocr_postprocessor import postprocess_ocr_prediction, validate_plate_format
except ImportError:
    postprocess_ocr_prediction = None
    validate_plate_format = None
except Exception:
    postprocess_ocr_prediction = None
    validate_plate_format = None

try:
    import pytesseract
except Exception:
    pytesseract = None

try:
    from paddleocr import PaddleOCR
except Exception:
    PaddleOCR = None

try:
    from rapidocr_onnxruntime import RapidOCR
except Exception:
    RapidOCR = None

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _require_tensorflow(feature_name):
    if tf is None:
        raise RuntimeError(
            f"TensorFlow no estÃ¡ instalado, pero se intentÃ³ usar {feature_name}. "
            "Usa modelos YOLO (.pt) y OCR RapidOCR/Tesseract, o instala TensorFlow en este entorno."
        )

def _resolve_tesseract_cmd():
    if pytesseract is None:
        return None

    configured = str(CONFIG.get("ocr_tesseract_cmd", "")).strip()
    if configured and Path(configured).exists():
        return configured

    resolved_from_path = shutil.which("tesseract")
    if resolved_from_path:
        return resolved_from_path

    for candidate in (
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
        Path("/usr/bin/tesseract"),
        Path("/usr/local/bin/tesseract"),
    ):
        if candidate.exists():
            return str(candidate)

    return None

# ============================================================================
# CLASSES
# ============================================================================

PLATE_CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
DIGIT_TO_LETTER_VISUAL = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "3": "B",
    "4": "A",
    "5": "S",
    "6": "G",
    "7": "T",
    "8": "B",
    "9": "P",
}
LETTER_TO_DIGIT_VISUAL = {
    "O": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "S": "5",
    "G": "6",
    "T": "7",
    "B": "8",
    "P": "9",
}


if tf is not None:
    class CTCLayer(tf.keras.layers.Layer):
        def call(self, y_true, y_pred):
            return y_pred
else:
    class CTCLayer:
        pass


def resolve_detector_model_path(preferred_path):
    preferred = Path(preferred_path)
    candidates = [
        preferred,
        Path("models/plate_detector_best.pt"),
        Path("models/yolo_plate_best.pt"),
        Path("models/lp_yolov8n.pt"),
        Path("models/optimized/detector_int8.tflite"),
        Path("models/detector_savedmodel"),
        Path("models/detector_best.h5"),
        Path("models/detector_final.h5"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return str(preferred)

class DetectorInference:
    """Detector optimizado con TFLite, Keras, SavedModel o YOLO."""
    
    def __init__(self, model_path, input_size=320):
        model_path = resolve_detector_model_path(model_path)
        logger.info(f"Cargando detector: {model_path}")
        self.model_path = model_path
        self.backend = "tflite"
        self.interpreter = None
        self.model = None
        self.yolo_model = None
        self.saved_model = None
        self.saved_model_fn = None
        self.input_details = None
        self.output_details = None
        self.input_size = input_size

        model_path_str = str(model_path)

        if model_path_str.lower().endswith(('.pt', '.onnx')):
            if YOLO is None:
                raise RuntimeError(
                    "ultralytics no estÃ¡ instalado, pero se intentÃ³ cargar un modelo YOLO"
                )
            self.backend = "yolo"
            self.input_size = 640
            self.yolo_model = YOLO(model_path_str)
            logger.info("âœ“ Detector listo (backend=yolo)")
            return

        # Si llega un modelo Keras, Ãºsalo directo para evitar dependencia Flex ops
        if model_path_str.lower().endswith((".h5", ".keras")):
            _require_tensorflow("detector backend=keras")
            self.backend = "keras"
            self.model = tf.keras.models.load_model(model_path, compile=False)
            logger.info("âœ“ Detector listo (backend=keras)")
            return

        # SavedModel directory fallback (compatible con TF 2.21 sin Keras deserialization)
        if Path(model_path).is_dir():
            _require_tensorflow("detector backend=saved_model")
            self.backend = "saved_model"
            self.saved_model = tf.saved_model.load(model_path)
            self.saved_model_fn = self.saved_model.signatures["serving_default"]
            logger.info("âœ“ Detector listo (backend=saved_model)")
            return

        # Camino estÃ¡ndar TFLite
        _require_tensorflow("detector backend=tflite")
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        logger.info("âœ“ Detector listo (backend=tflite)")
    
    def predict(self, image):
        """
        Detecta placas en imagen
        
        Args:
            image: BGR image
        
        Returns:
            predictions: [x, y, w, h, confidence]
        """
        # Preprocesar
        orig_h, orig_w = image.shape[:2]

        if self.backend == "yolo":
            results = self.yolo_model.predict(
                source=image,
                imgsz=self.input_size,
                conf=0.15,
                iou=0.5,
                verbose=False,
            )
            if not results or len(results[0].boxes) == 0:
                return {
                    'x': orig_w / 2.0,
                    'y': orig_h / 2.0,
                    'w': 1.0,
                    'h': 1.0,
                    'confidence': 0.0,
                }

            boxes = results[0].boxes
            best_idx = int(np.argmax(boxes.conf.cpu().numpy()))
            x1, y1, x2, y2 = boxes.xyxy[best_idx].cpu().numpy().tolist()
            conf = float(boxes.conf[best_idx].cpu().item())
            x = (x1 + x2) / 2.0
            y = (y1 + y2) / 2.0
            bw = max(1.0, x2 - x1)
            bh = max(1.0, y2 - y1)

            return {
                'x': max(0.0, min(x, orig_w - 1.0)),
                'y': max(0.0, min(y, orig_h - 1.0)),
                'w': bw,
                'h': bh,
                'confidence': min(1.0, max(0.0, conf)),
            }

        img_resized = cv2.resize(image, (self.input_size, self.input_size))
        img_normalized = img_resized.astype(np.float32) / 255.0

        if self.backend == "keras":
            img_input = np.expand_dims(img_normalized.astype(np.float32), 0)
            output = self.model.predict(img_input, verbose=0)[0]
        elif self.backend == "saved_model":
            img_input = np.expand_dims(img_normalized.astype(np.float32), 0)
            result = self.saved_model_fn(tf.constant(img_input))
            output = next(iter(result.values())).numpy()[0]
        else:
            input_dtype = self.input_details[0]['dtype']
            img_input = np.expand_dims(img_normalized.astype(input_dtype), 0)
            self.interpreter.set_tensor(
                self.input_details[0]['index'],
                img_input
            )
            self.interpreter.invoke()
            output = self.interpreter.get_tensor(
                self.output_details[0]['index']
            )[0]  # [x, y, w, h, conf]
        
        # Desnormalizar a imagen original
        x, y, bw, bh, conf = output
        x = float(x) * orig_w
        y = float(y) * orig_h
        bw = float(bw) * orig_w
        bh = float(bh) * orig_h
        
        return {
            'x': max(0.0, min(x, orig_w - 1.0)),
            'y': max(0.0, min(y, orig_h - 1.0)),
            'w': max(1.0, bw),
            'h': max(1.0, bh),
            'confidence': min(1.0, max(0.0, conf))
        }

class PipelineJetson:
    """Pipeline completo optimizado para Jetson"""
    
    def __init__(self, config):
        self.config = config
        try:
            self.detector = DetectorInference(
                config["detector_model"],
                config["input_size"]
            )
        except RuntimeError as e:
            if "Select TensorFlow op" in str(e):
                fallback_candidates = [
                    Path("models/detector_savedmodel"),
                    Path("models/detector_final.h5"),
                    Path("models/detector_best.h5"),
                ]
                fallback = next((p for p in fallback_candidates if p.exists()), None)
                if fallback is None:
                    raise
                logger.warning(
                    "âš ï¸ TFLite requiere Flex ops no disponibles. "
                    "Usando fallback Keras: %s",
                    fallback
                )
                self.detector = DetectorInference(str(fallback), config["input_size"])
            else:
                raise

        if getattr(self.detector, "backend", None) == "yolo":
            self.detector.input_size = int(config.get("yolo_imgsz", 640))

        self.detector_fallback = None
        if config.get("enable_detector_fallback", False):
            fallback_model = config.get("detector_fallback_model", "")
            if fallback_model:
                main_path = str(resolve_detector_model_path(config["detector_model"]))
                fallback_path = str(resolve_detector_model_path(fallback_model))
                if fallback_path != main_path and Path(fallback_path).exists():
                    try:
                        self.detector_fallback = DetectorInference(
                            fallback_path,
                            config["input_size"]
                        )
                        if getattr(self.detector_fallback, "backend", None) == "yolo":
                            self.detector_fallback.input_size = int(config.get("yolo_imgsz", 640))
                        logger.info("âœ“ Detector fallback listo: %s", fallback_path)
                    except Exception as e:
                        logger.warning("âš ï¸ No se pudo cargar detector fallback (%s): %s", fallback_path, e)
        ocr_path = Path(config["ocr_model"])
        if ocr_path.exists():
            self.ocr = OCRInference(config["ocr_model"])
            self.ocr_enabled = True
        else:
            self.ocr = None
            self.ocr_enabled = False
            logger.warning(
                "âš ï¸ OCR no disponible: %s. Continuando solo con detector.",
                config["ocr_model"]
            )
        self.fps_buffer = deque(maxlen=config["fps_buffer_size"])
        self._last_selected_ocr = None
        
        logger.info("âœ“ Pipeline inicializado")

    def _extract_boxes(self, detection, frame_shape):
        h, w = frame_shape[:2]

        draw_scale = self.config.get("plate_box_scale", 1.0)
        ocr_scale = self._adaptive_ocr_scale(detection, frame_shape)

        draw_w_scaled = detection['w'] * draw_scale
        draw_h_scaled = detection['h'] * draw_scale
        ocr_w_scaled = detection['w'] * ocr_scale
        ocr_h_scaled = detection['h'] * ocr_scale

        x1 = int(max(0, detection['x'] - draw_w_scaled / 2))
        y1 = int(max(0, detection['y'] - draw_h_scaled / 2))
        x2 = int(min(w, detection['x'] + draw_w_scaled / 2))
        y2 = int(min(h, detection['y'] + draw_h_scaled / 2))

        x1_ocr = int(max(0, detection['x'] - ocr_w_scaled / 2))
        y1_ocr = int(max(0, detection['y'] - ocr_h_scaled / 2))
        x2_ocr = int(min(w, detection['x'] + ocr_w_scaled / 2))
        y2_ocr = int(min(h, detection['y'] + ocr_h_scaled / 2))

        return (x1, y1, x2, y2, x1_ocr, y1_ocr, x2_ocr, y2_ocr)

    def _adaptive_ocr_scale(self, detection, frame_shape):
        """Ajusta el recorte OCR segÃºn el tamaÃ±o aparente de la placa."""
        plate_h = float(detection.get("h", 0.0))
        frame_h = float(frame_shape[0]) if frame_shape else 1.0
        relative_h = plate_h / max(1.0, frame_h)

        if plate_h >= 110 or relative_h >= 0.12:
            return float(self.config.get("ocr_box_scale_close", 0.78))
        if plate_h >= 75 or relative_h >= 0.08:
            return float(self.config.get("ocr_box_scale_medium", 0.88))
        return float(self.config.get("ocr_box_scale_small", 0.95))

    def _ocr_roi_candidates(self, detection, frame_shape):
        """Genera recortes OCR con escalas y sesgos verticales para robustez."""
        base_scale = self._adaptive_ocr_scale(detection, frame_shape)
        plate_h = float(detection.get("h", 0.0))

        if plate_h >= 110:
            y_bias = float(self.config.get("ocr_roi_y_bias_close", 0.10))
            candidates = [
                (base_scale - 0.20, y_bias),
                (base_scale - 0.12, y_bias * 0.7),
                (base_scale - 0.05, y_bias * 0.4),
                (base_scale, 0.0),
                (base_scale + 0.08, 0.0),
                (base_scale + 0.16, 0.0),
                (base_scale + 0.24, 0.0),
                (base_scale + 0.24, y_bias * 0.5),
            ]
        elif plate_h >= 75:
            y_bias = float(self.config.get("ocr_roi_y_bias_medium", 0.06))
            candidates = [
                (base_scale - 0.12, y_bias),
                (base_scale - 0.06, y_bias * 0.6),
                (base_scale, 0.0),
                (base_scale + 0.07, 0.0),
                (base_scale + 0.12, 0.0),
            ]
        else:
            y_bias = float(self.config.get("ocr_roi_y_bias_small", 0.03))
            candidates = [
                (base_scale - 0.08, y_bias),
                (base_scale, 0.0),
                (base_scale + 0.06, 0.0),
                (base_scale + 0.12, 0.0),
            ]

        unique_candidates = []
        for scale, bias in candidates:
            scale = float(max(0.5, min(scale, 1.15)))
            bias = float(max(-0.18, min(bias, 0.18)))
            key = (round(scale, 4), round(bias, 4))
            if key not in unique_candidates:
                unique_candidates.append(key)
        return unique_candidates

    def _normalize_plate_text(self, text):
        t = str(text).upper().strip()
        t = t.replace('â€”', '-').replace('_', '-')
        t = re.sub(r'[\t ]+', '', t)
        return t

    def _is_valid_plate_text(self, text):
        t = self._normalize_plate_text(text)

        if re.fullmatch(r'[A-Z]{3}-\d{3,4}', t):
            return True, t

        lines = [line.strip().upper() for line in str(text).splitlines() if line.strip()]
        if len(lines) == 2:
            first = re.sub(r'[^A-Z]', '', lines[0])
            second = re.sub(r'[^A-Z0-9]', '', lines[1])
            if re.fullmatch(r'[A-Z]{2}', first) and re.fullmatch(r'\d{3}[A-Z]', second):
                return True, f'{first}\n{second}'

        return False, t

    def _is_plausible_box(self, box, frame_shape, source="primary"):
        x1, y1, x2, y2 = box
        h, w = frame_shape[:2]
        if x2 <= x1 or y2 <= y1:
            return False

        box_w = x2 - x1
        box_h = y2 - y1
        area_ratio = (box_w * box_h) / float(max(1, w * h))
        aspect_ratio = box_w / float(max(1, box_h))

        min_area_ratio = self.config["min_area_ratio"]
        if source == "small_rescue":
            min_area_ratio = self.config.get("min_area_ratio_small", min_area_ratio)

        return (
            min_area_ratio <= area_ratio <= self.config["max_area_ratio"]
            and self.config["min_aspect_ratio"] <= aspect_ratio <= self.config["max_aspect_ratio"]
        )

    def _preprocess_for_detection(self, frame):
        """Mejora contraste local antes de detectar placas en escenas difÃ­ciles."""
        if frame is None or frame.size == 0:
            return frame

        if len(frame.shape) == 2:
            gray = frame
            clahe = cv2.createCLAHE(
                clipLimit=float(self.config.get("detector_preprocess_clahe_clip", 2.4)),
                tileGridSize=(8, 8),
            ).apply(gray)
            blur = cv2.GaussianBlur(clahe, (0, 0), 0.9)
            sharp = cv2.addWeighted(clahe, 1.55, blur, -0.55, 0)
            return cv2.cvtColor(np.clip(sharp, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)

        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(
            clipLimit=float(self.config.get("detector_preprocess_clahe_clip", 2.4)),
            tileGridSize=(8, 8),
        ).apply(l)
        lab_enh = cv2.merge((clahe, a, b))
        bgr = cv2.cvtColor(lab_enh, cv2.COLOR_LAB2BGR)
        blur = cv2.GaussianBlur(bgr, (0, 0), 0.9)
        sharp = cv2.addWeighted(bgr, 1.40, blur, -0.40, 0)
        return np.clip(sharp, 0, 255).astype(np.uint8)

    def _try_detector(self, detector, frame, min_conf, source="primary"):
        detection = detector.predict(frame)
        if detection['confidence'] <= min_conf:
            return None

        boxes = self._extract_boxes(detection, frame.shape)
        if not self._is_plausible_box((boxes[4], boxes[5], boxes[6], boxes[7]), frame.shape, source=source):
            return None

        return detection, boxes

    def _small_plate_rescue(self, frame):
        if not self.config.get("enable_small_plate_rescue", True):
            return None

        rescue_detectors = []
        if getattr(self.detector, "backend", None) == "yolo" and getattr(self.detector, "yolo_model", None) is not None:
            rescue_detectors.append(self.detector)
        if (
            self.detector_fallback is not None
            and getattr(self.detector_fallback, "backend", None) == "yolo"
            and getattr(self.detector_fallback, "yolo_model", None) is not None
            and self.detector_fallback is not self.detector
        ):
            rescue_detectors.append(self.detector_fallback)

        if not rescue_detectors:
            return None

        scale = float(self.config.get("small_plate_rescue_upscale", 1.8))
        if scale <= 1.0:
            return None

        def run_yolo_best(image):
            best = None
            best_conf = -1.0

            for det_obj in rescue_detectors:
                results = det_obj.yolo_model.predict(
                    source=image,
                    imgsz=int(self.config.get("small_plate_rescue_imgsz", 960)),
                    conf=float(self.config.get("small_plate_rescue_conf", 0.08)),
                    iou=float(self.config.get("small_plate_rescue_iou", 0.5)),
                    verbose=False,
                )
                if not results or len(results[0].boxes) == 0:
                    continue
                b = results[0].boxes
                i = int(np.argmax(b.conf.cpu().numpy()))
                x1, y1, x2, y2 = b.xyxy[i].cpu().numpy().tolist()
                conf = float(b.conf[i].cpu().item())
                if conf > best_conf:
                    best_conf = conf
                    best = (x1, y1, x2, y2, conf)
            return best

        def to_det(x1, y1, x2, y2, conf, fw, fh):
            x1 = max(0.0, min(x1, fw - 1.0))
            y1 = max(0.0, min(y1, fh - 1.0))
            x2 = max(1.0, min(x2, fw))
            y2 = max(1.0, min(y2, fh))
            return {
                "x": (x1 + x2) / 2.0,
                "y": (y1 + y2) / 2.0,
                "w": max(1.0, x2 - x1),
                "h": max(1.0, y2 - y1),
                "confidence": min(1.0, max(0.0, conf)),
            }

        h, w = frame.shape[:2]
        best = None
        best_conf = -1.0
        accept = float(self.config.get("small_plate_rescue_accept_conf", 0.20))

        # Pass 1: imagen completa escalada.
        up_w = max(1, int(w * scale))
        up_h = max(1, int(h * scale))
        up = cv2.resize(frame, (up_w, up_h), interpolation=cv2.INTER_CUBIC)
        full = run_yolo_best(up)
        if full is not None:
            x1u, y1u, x2u, y2u, conf = full
            det = to_det(x1u / scale, y1u / scale, x2u / scale, y2u / scale, conf, w, h)
            boxes_px = self._extract_boxes(det, frame.shape)
            if conf >= accept and self._is_plausible_box((boxes_px[4], boxes_px[5], boxes_px[6], boxes_px[7]), frame.shape, source="small_rescue"):
                best = (det, boxes_px, "small_rescue")
                best_conf = conf

        # Pass 2: tiles con solape, solo si no hubo candidato bueno.
        if best is None and bool(self.config.get("small_plate_rescue_tiling", True)):
            grid = max(2, int(self.config.get("small_plate_rescue_tile_grid", 2)))
            overlap = float(self.config.get("small_plate_rescue_tile_overlap", 0.22))
            tile_w = max(64, int(w / grid))
            tile_h = max(64, int(h / grid))
            step_x = max(16, int(tile_w * (1.0 - overlap)))
            step_y = max(16, int(tile_h * (1.0 - overlap)))

            y = 0
            while y < h:
                x = 0
                y2 = min(h, y + tile_h)
                while x < w:
                    x2 = min(w, x + tile_w)
                    tile = frame[y:y2, x:x2]
                    if tile.size == 0:
                        x += step_x
                        continue
                    pred = run_yolo_best(tile)
                    if pred is not None:
                        tx1, ty1, tx2, ty2, conf = pred
                        gx1, gy1 = x + tx1, y + ty1
                        gx2, gy2 = x + tx2, y + ty2
                        det = to_det(gx1, gy1, gx2, gy2, conf, w, h)
                        boxes_px = self._extract_boxes(det, frame.shape)
                        if (
                            conf >= accept
                            and conf > best_conf
                            and self._is_plausible_box((boxes_px[4], boxes_px[5], boxes_px[6], boxes_px[7]), frame.shape, source="small_rescue")
                        ):
                            best = (det, boxes_px, "small_rescue")
                            best_conf = conf
                    if x2 >= w:
                        break
                    x += step_x
                if y2 >= h:
                    break
                y += step_y

        return best

    def _ocr_text_score(self, text, ocr_conf):
        if not text:
            return 0.0

        t = "".join(ch for ch in str(text).upper() if ch.isalnum())
        if not t:
            return 0.0

        score = float(ocr_conf)
        n = len(t)
        if 6 <= n <= 8:
            score += 0.06
        if n == 7:
            score += 0.06
        if re.fullmatch(r"[A-Z]{3}[0-9]{4}", t):
            score += 0.14
        if not any(ch.isdigit() for ch in t):
            score -= 0.07
        if not any(ch.isalpha() for ch in t):
            score -= 0.07
        return score

    def _ocr_second_pass_7char(self, frame, det, base_box, current_text, current_conf):
        """Segundo pase OCR para forzar hipÃ³tesis LLL-NNNN en baja confianza."""
        if not bool(self.config.get("ocr_second_pass_7char_enabled", True)):
            return current_text, current_conf, base_box

        threshold = float(self.config.get("ocr_second_pass_conf_threshold", 0.085))
        if float(current_conf) >= threshold:
            return current_text, current_conf, base_box

        x1, y1, x2, y2 = base_box
        h, w = frame.shape[:2]
        if x2 <= x1 or y2 <= y1:
            return current_text, current_conf, base_box

        bw = float(x2 - x1)
        bh = float(y2 - y1)
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        best_text = current_text
        best_conf = float(current_conf)
        best_box = base_box
        best_score = self._ocr_text_score(current_text, current_conf)

        pattern_bonus = float(self.config.get("ocr_second_pass_pattern_bonus", 0.16))
        scales = (0.94, 1.00, 1.06, 1.12)
        x_biases = (0.00, 0.03, -0.02)
        y_biases = (0.00, 0.04, 0.08)

        for sc in scales:
            for xb in x_biases:
                for yb in y_biases:
                    tw = bw * sc
                    th = bh * sc
                    tcx = cx + (bw * xb)
                    tcy = cy + (bh * yb)

                    tx1 = int(max(0, tcx - tw / 2))
                    ty1 = int(max(0, tcy - th / 2))
                    tx2 = int(min(w, tcx + tw / 2))
                    ty2 = int(min(h, tcy + th / 2))

                    if tx2 <= tx1 or ty2 <= ty1:
                        continue

                    roi = frame[ty1:ty2, tx1:tx2]
                    if roi.size == 0:
                        continue

                    txt, conf = self.ocr.segment_and_recognize_with_conf(roi)
                    norm = "".join(ch for ch in str(txt).upper() if ch.isalnum())
                    if not re.fullmatch(r"[A-Z]{3}[0-9]{4}", norm):
                        continue

                    formatted = f"{norm[:3]}-{norm[3:]}"
                    score = self._ocr_text_score(formatted, conf) + pattern_bonus
                    if score > best_score:
                        best_score = score
                        best_text = formatted
                        best_conf = float(conf)
                        best_box = (tx1, ty1, tx2, ty2)

        return best_text, best_conf, best_box

    def _build_detector_candidates(self, frame, allow_small_rescue=True):
        candidates = []

        primary_min_conf = float(self.config["confidence_threshold"])
        if self.detector_fallback is None:
            # Si no hay fallback real cargado, evita umbral demasiado estricto en el primario.
            primary_min_conf = float(
                self.config.get("fallback_confidence_threshold", primary_min_conf)
            )

        enable_pre = bool(self.config.get("detector_preprocess_enabled", True))
        frame_pre = self._preprocess_for_detection(frame) if enable_pre else None

        primary_options = []
        primary_raw = self._try_detector(
            self.detector,
            frame,
            primary_min_conf,
            source="primary",
        )
        if primary_raw is not None:
            primary_options.append(primary_raw)

        if frame_pre is not None:
            primary_pre = self._try_detector(
                self.detector,
                frame_pre,
                primary_min_conf,
                source="primary",
            )
            if primary_pre is not None:
                primary_options.append(primary_pre)

        if primary_options:
            det, boxes = max(primary_options, key=lambda item: float(item[0]["confidence"]))
            candidates.append((det, boxes, "primary"))

        if self.detector_fallback is not None:
            fb_min_conf = float(self.config.get("fallback_confidence_threshold", self.config["confidence_threshold"]))
            fallback_options = []

            fb_raw = self._try_detector(
                self.detector_fallback,
                frame,
                fb_min_conf,
                source="fallback",
            )
            if fb_raw is not None:
                fallback_options.append(fb_raw)

            if frame_pre is not None:
                fb_pre = self._try_detector(
                    self.detector_fallback,
                    frame_pre,
                    fb_min_conf,
                    source="fallback",
                )
                if fb_pre is not None:
                    fallback_options.append(fb_pre)

            if fallback_options:
                det, boxes = max(fallback_options, key=lambda item: float(item[0]["confidence"]))
                candidates.append((det, boxes, "fallback"))

        # Rescue de placas pequeÃ±as: solo cuando falla lo normal, para no afectar lo ya estable.
        only_on_miss = bool(self.config.get("small_plate_rescue_only_on_miss", True))
        if allow_small_rescue and ((not candidates and only_on_miss) or (not only_on_miss)):
            rescue = self._small_plate_rescue(frame)
            if rescue is not None:
                candidates.append(rescue)

        return candidates

    def select_detection(self, frame):
        self._last_selected_ocr = None
        candidates = self._build_detector_candidates(frame)
        if not candidates:
            return None

        h, w = frame.shape[:2]
        text_w = float(self.config.get("ocr_score_text_weight", 0.75))
        det_w = float(self.config.get("ocr_score_det_weight", 0.25))
        fb_bonus = float(self.config.get("ocr_score_fallback_bonus", 0.02))
        prefer_fb = bool(self.config.get("prefer_fallback_for_ocr", True))

        def evaluate_candidate(det, boxes, src):
            x1_ocr_base, y1_ocr_base, x2_ocr_base, y2_ocr_base = boxes[4], boxes[5], boxes[6], boxes[7]
            if x2_ocr_base <= x1_ocr_base or y2_ocr_base <= y1_ocr_base:
                return None, -1e9

            best_for_det = None
            best_for_det_score = -1e9

            for scale, y_bias in self._ocr_roi_candidates(det, frame.shape):
                draw_w_scaled = det["w"] * scale
                draw_h_scaled = det["h"] * scale
                x1_ocr = int(max(0, det["x"] - draw_w_scaled / 2))
                center_y = det["y"] + (det["h"] * y_bias)
                y1_ocr = int(max(0, center_y - draw_h_scaled / 2))
                x2_ocr = int(min(w, det["x"] + draw_w_scaled / 2))
                y2_ocr = int(min(h, center_y + draw_h_scaled / 2))

                if x2_ocr <= x1_ocr or y2_ocr <= y1_ocr:
                    continue

                roi = frame[y1_ocr:y2_ocr, x1_ocr:x2_ocr]
                if roi.size == 0:
                    continue

                ocr_text, ocr_conf = self.ocr.segment_and_recognize_with_conf(roi)
                ocr_text, ocr_conf, tuned_box = self._ocr_second_pass_7char(
                    frame,
                    det,
                    (x1_ocr, y1_ocr, x2_ocr, y2_ocr),
                    ocr_text,
                    ocr_conf,
                )
                is_valid, normalized = self._is_valid_plate_text(ocr_text)
                text_score = self._ocr_text_score(normalized, ocr_conf)
                if is_valid:
                    text_score += 0.20

                combined = (text_w * text_score) + (det_w * float(det["confidence"]))
                if prefer_fb and src == "fallback":
                    combined += fb_bonus

                if combined > best_for_det_score:
                    best_for_det_score = combined
                    best_for_det = (
                        det,
                        boxes,
                        src,
                        normalized if is_valid else ocr_text,
                        float(ocr_conf),
                        tuned_box,
                    )

            return best_for_det, best_for_det_score

        # Mantiene comportamiento clÃ¡sico si no hay OCR o estÃ¡ desactivado el selector OCR-aware.
        if (
            len(candidates) == 1
            or not self.config.get("ocr_aware_box_selection", True)
            or not self.ocr_enabled
            or self.ocr is None
        ):
            detection, boxes, detection_source = candidates[0]

            if self.ocr_enabled and self.ocr is not None:
                best_for_det, _ = evaluate_candidate(detection, boxes, detection_source)
                if best_for_det is not None:
                    _, _, _, ocr_text, ocr_conf, ocr_box = best_for_det
                    self._last_selected_ocr = {
                        "text": ocr_text,
                        "conf": ocr_conf,
                        "source": detection_source,
                        "ocr_box": ocr_box,
                    }
            return detection, boxes, detection_source

        best = None
        best_score = -1.0
        for det, boxes, src in candidates:
            best_for_det, best_for_det_score = evaluate_candidate(det, boxes, src)

            if best_for_det is not None and best_for_det_score > best_score:
                best_score = best_for_det_score
                best = best_for_det

        if best is None:
            detection, boxes, detection_source = candidates[0]
            return detection, boxes, detection_source

        detection, boxes, detection_source, ocr_text, ocr_conf, ocr_box = best
        self._last_selected_ocr = {
            "text": ocr_text,
            "conf": ocr_conf,
            "source": detection_source,
            "ocr_box": ocr_box,
        }
        return detection, boxes, detection_source
    
    def process_frame(self, frame):
        """
        Procesa un frame
        
        Returns:
            frame: frame con anotaciones
            detections: lista de placas detectadas
        """
        t_start = time.time()
        detections = []
        
        selection = self.select_detection(frame)
        if selection is not None:
            detection, boxes, detection_source = selection
            x1, y1, x2, y2, x1_ocr, y1_ocr, x2_ocr, y2_ocr = boxes
            plate_roi = frame[y1_ocr:y2_ocr, x1_ocr:x2_ocr]

            # OCR opcional: si no existe modelo OCR, sigue con detector-only
            if self.ocr_enabled and self.ocr is not None:
                cached = self._last_selected_ocr
                if cached is not None and cached.get("source") == detection_source:
                    plate_text = str(cached.get("text", ""))
                else:
                    # Obtener predicción OCR inicial
                    plate_text = self.ocr.segment_and_recognize(plate_roi)
                    
                    # OPCIÓN C: Post-procesamiento inteligente con reintentos
                    if postprocess_ocr_prediction is not None:
                        plate_text, is_valid, adjusted_coords = postprocess_ocr_prediction(
                            plate_text,
                            frame=frame,
                            roi_coords=(x1_ocr, y1_ocr, x2_ocr, y2_ocr),
                            ocr_fn=self.ocr.segment_and_recognize
                        )
                        # Si aún está inválido y es "ECUADOR", intentar reintentos manuales
                        if not is_valid and plate_text.upper() == "ECUADOR" and frame.shape[0] > 0:
                            roi_height = y2_ocr - y1_ocr
                            logger.info(f"⚠ Reintentando OCR (height={roi_height})...")
                            
                            # Intentar múltiples desplazamientos hacia abajo
                            for offset_pct in [0.25, 0.35, 0.15, 0.45]:
                                y_offset = int(roi_height * offset_pct)
                                y1_retry = min(y1_ocr + y_offset, frame.shape[0] - 1)
                                y2_retry = min(y2_ocr + y_offset, frame.shape[0])
                                
                                if y2_retry <= y1_retry or y2_retry - y1_retry < 5:
                                    continue
                                
                                try:
                                    retry_roi = frame[y1_retry:y2_retry, x1_ocr:x2_ocr]
                                    if retry_roi.size == 0:
                                        continue
                                    
                                    retry_text = self.ocr.segment_and_recognize(retry_roi)
                                    retry_processed, is_valid_retry, _ = postprocess_ocr_prediction(retry_text)
                                    
                                    if is_valid_retry:
                                        plate_text = retry_processed
                                        logger.info(f"✓ Reintento OCR exitoso: {plate_text} (offset {offset_pct*100:.0f}%)")
                                        break
                                except Exception as e:
                                    logger.debug(f"Reintento OCR falló: {e}")
                                    pass
                    else:
                        logger.info(f"ℹ postprocess_ocr_prediction no disponible (es None)")
            else:
                plate_text = "PLATE"

            detections.append({
                'bbox': (x1, y1, x2, y2),
                'text': plate_text,
                'confidence': detection['confidence'],
                'source': detection_source,
            })

            # Dibujar
            frame = self._draw_detection(
                frame, x1, y1, x2, y2,
                plate_text,
                detection['confidence']
            )

            cached = self._last_selected_ocr
            if isinstance(cached, dict) and cached.get("ocr_box") is not None:
                ox1, oy1, ox2, oy2 = cached["ocr_box"]
                frame = self._draw_ocr_roi(frame, ox1, oy1, ox2, oy2)
        
        # FPS
        t_end = time.time()
        frame_time = (t_end - t_start) * 1000  # ms
        self.fps_buffer.append(1000 / max(frame_time, 1))
        
        # Dibujar FPS
        fps_avg = np.mean(list(self.fps_buffer))
        cv2.putText(
            frame,
            f"FPS: {fps_avg:.1f} ({frame_time:.0f}ms)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        
        # Dibujar memoria (opcional)
        try:
            import psutil
            memory_percent = psutil.virtual_memory().percent
            cv2.putText(
                frame,
                f"RAM: {memory_percent:.1f}%",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                1
            )
        except:
            pass
        
        return frame, detections
    
    def _draw_detection(self, frame, x1, y1, x2, y2, text, confidence):
        """Dibuja detecciÃ³n"""
        
        # Bounding box
        color = (0, 255, 0) if confidence > 0.8 else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Label
        label = f"{text} {confidence:.2f}"
        label_size, _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1
        )
        
        cv2.rectangle(
            frame,
            (x1, y1 - label_size[1] - 4),
            (x1 + label_size[0], y1),
            color,
            -1
        )
        
        cv2.putText(
            frame,
            label,
            (x1, y1 - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1
        )
        
        return frame

    def _draw_ocr_roi(self, frame, x1, y1, x2, y2):
        """Dibuja el ROI usado para OCR."""
        color = (0, 140, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = "OCR ROI"
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y_label = max(0, y1 - label_size[1] - 6)
        cv2.rectangle(
            frame,
            (x1, y_label),
            (x1 + label_size[0] + 8, y_label + label_size[1] + 6),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 4, y_label + label_size[1] + 1),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        return frame

    def _open_camera(self, video_source):
        """Abre cÃ¡mara con backends adecuados por plataforma y fallback genÃ©rico."""
        if isinstance(video_source, str) and video_source.isdigit():
            video_source = int(video_source)

        if isinstance(video_source, int):
            if os.name == "nt":
                backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
            else:
                # En Linux/Jetson priorizamos GStreamer con v4l2src en GRAY8.
                backends = [cv2.CAP_GSTREAMER, cv2.CAP_V4L2, cv2.CAP_ANY]
        else:
            backends = [cv2.CAP_ANY]

        for backend in backends:
            if isinstance(video_source, int) and backend == cv2.CAP_GSTREAMER:
                gst_src = self._build_v4l2_gstreamer_source(video_source)
                cap = cv2.VideoCapture(gst_src, cv2.CAP_GSTREAMER)
            else:
                cap = cv2.VideoCapture(video_source, backend)

            if cap is not None and cap.isOpened() and isinstance(video_source, int) and backend == cv2.CAP_V4L2:
                self._configure_v4l2_capture(cap)

            if cap is not None and cap.isOpened():
                return cap, backend
            if cap is not None:
                cap.release()

        return None, None

    def _configure_v4l2_capture(self, cap):
        """Configura V4L2 para captura monocroma cruda de forma estable."""
        width = int(self.config.get("camera_capture_width", 1280))
        height = int(self.config.get("camera_capture_height", 720))
        fps = int(self.config.get("camera_capture_fps", 30))
        try:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            # Solicita GREY 8-bit para evitar conversiones errÃ³neas de Y16.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"GREY"))
            # Mantener datos crudos y procesar nosotros.
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
            try:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except cv2.error:
                pass
        except cv2.error:
            pass

    def _build_v4l2_gstreamer_source(self, sensor_id=0):
        """Fuente GStreamer para V4L2 monocromo con salida BGR compatible OpenCV."""
        sid = int(sensor_id)
        width = int(self.config.get("camera_capture_width", 1280))
        height = int(self.config.get("camera_capture_height", 720))
        fps = int(self.config.get("camera_capture_fps", 30))
        return (
            f"v4l2src device=/dev/video{sid} io-mode=2 ! "
            f"video/x-raw,format=GRAY8,width={width},height={height},framerate={fps}/1 ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            "appsink drop=true max-buffers=1 sync=false"
        )

    def _prepare_camera_frame(self, frame):
        """Normaliza frames de cÃ¡mara (mono/16-bit) y los entrega en BGR uint8."""
        if frame is None or frame.size == 0:
            return frame

        out = frame
        src_dtype = getattr(out, "dtype", None)
        src_shape = getattr(out, "shape", None)

        if out.dtype != np.uint8:
            # La cÃ¡mara CSI monocroma suele entregar Y16/Y10. En escenas con
            # rango dinÃ¡mico estrecho, un cast directo puede dejar todo negro.
            # Usamos estiramiento por percentiles para asegurar visibilidad.
            if np.issubdtype(out.dtype, np.integer):
                if out.dtype.itemsize > 1:
                    arr = out.astype(np.float32)
                    p1 = float(np.percentile(arr, 1.0))
                    p99 = float(np.percentile(arr, 99.0))
                    if p99 <= p1:
                        min_v = float(np.min(arr))
                        max_v = float(np.max(arr))
                        p1, p99 = min_v, max_v

                    if p99 > p1:
                        arr = (arr - p1) * (255.0 / (p99 - p1))
                    else:
                        # Fallback si el frame viene casi plano.
                        arr = arr * (255.0 / max(1.0, float(np.max(arr))))

                    out = np.clip(arr, 0, 255).astype(np.uint8)
                else:
                    out = out.astype(np.uint8)
            else:
                arr = out.astype(np.float32)
                min_v = float(np.min(arr))
                max_v = float(np.max(arr))
                if max_v > min_v:
                    arr = (arr - min_v) * (255.0 / (max_v - min_v))
                out = np.clip(arr, 0, 255).astype(np.uint8)

            # Empuje local de contraste para sensores monocromos industriales.
            if len(out.shape) == 2:
                out = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(out)

        if len(out.shape) == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        elif len(out.shape) == 3 and out.shape[2] == 1:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)

        if frame is not None and src_dtype is not None and src_dtype != np.uint8:
            logger.debug(
                "Frame CSI normalizado: shape=%s dtype=%s -> uint8 BGR=%s",
                src_shape,
                src_dtype,
                out.shape,
            )

        if bool(self.config.get("camera_input_grayscale", False)):
            out = self._apply_camera_grayscale(out)

        return out

    def _start_camera_reader(self, cap):
        """Arranca un lector en segundo plano para no bloquear la cÃ¡mara con la inferencia."""
        state = {
            "running": True,
            "frame": None,
            "frame_id": 0,
            "last_frame_ts": 0.0,
            "lock": threading.Lock(),
            "thread": None,
        }

        def _reader():
            while state["running"]:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.01)
                    continue
                with state["lock"]:
                    state["frame"] = frame
                    state["frame_id"] += 1
                    state["last_frame_ts"] = time.time()

        thread = threading.Thread(target=_reader, daemon=True)
        state["thread"] = thread
        thread.start()
        return state

    def _start_ocr_worker(self):
        """Worker de OCR en segundo plano para no congelar el stream de cÃ¡mara."""
        state = {
            "running": True,
            "lock": threading.Lock(),
            "pending_queue": deque(),
            "pending_signatures": set(),
            "in_flight_signature": None,
            "max_pending": max(1, int(self.config.get("camera_ocr_max_pending", 2))),
            "result": None,
            "busy": False,
            "thread": None,
        }

        def _worker():
            while state["running"]:
                task = None
                with state["lock"]:
                    if state["pending_queue"]:
                        task = state["pending_queue"].popleft()
                        state["in_flight_signature"] = task[0]
                        state["busy"] = True

                if task is None:
                    time.sleep(0.005)
                    continue

                signature, roi, det, source, det_box, ocr_box = task
                ocr_best = None
                try:
                    ocr_best = self._ocr_from_roi_fast(roi, det, source, ocr_box)
                except Exception as e:
                    logger.warning("âš ï¸ OCR worker error: %s", e)

                with state["lock"]:
                    state["result"] = (signature, det, source, tuple(det_box), ocr_best, time.time())
                    state["pending_signatures"].discard(signature)
                    state["in_flight_signature"] = None
                    state["busy"] = False

        thread = threading.Thread(target=_worker, daemon=True)
        state["thread"] = thread
        thread.start()
        return state

    def _submit_ocr_task(self, worker_state, signature, roi, det, source, det_box, ocr_box):
        """Encola OCR para una firma Ãºnica y garantiza al menos un intento por placa."""
        with worker_state["lock"]:
            if signature == worker_state.get("in_flight_signature"):
                return "skipped"
            if signature in worker_state["pending_signatures"]:
                return "skipped"

            dropped = False
            max_pending = int(worker_state.get("max_pending", 2))
            while len(worker_state["pending_queue"]) >= max_pending:
                old_signature, *_ = worker_state["pending_queue"].popleft()
                worker_state["pending_signatures"].discard(old_signature)
                dropped = True

            worker_state["pending_queue"].append(
                (
                    signature,
                    roi,
                    dict(det),
                    source,
                    tuple(det_box),
                    tuple(ocr_box),
                )
            )
            worker_state["pending_signatures"].add(signature)
            return "replaced" if dropped else "queued"

    def _pop_ocr_result(self, worker_state):
        """Recupera el Ãºltimo resultado OCR del worker si existe."""
        with worker_state["lock"]:
            result = worker_state["result"]
            worker_state["result"] = None
        return result

    def _start_debug_snapshot_worker(self):
        """Worker dedicado para guardar snapshots OCR sin bloquear el loop de cÃ¡mara."""
        state = {
            "running": True,
            "lock": threading.Lock(),
            "pending_queue": deque(),
            "max_pending": max(1, int(self.config.get("camera_debug_max_pending", 1))),
            "result": None,
            "busy": False,
            "thread": None,
        }

        def _worker():
            while state["running"]:
                task = None
                with state["lock"]:
                    if state["pending_queue"]:
                        task = state["pending_queue"].popleft()
                        state["busy"] = True

                if task is None:
                    time.sleep(0.01)
                    continue

                frame, det, boxes, source, output_dir, now_ts = task
                out_path = None
                text = ""
                conf = 0.0
                err = None
                try:
                    out_path, text, conf = self._save_periodic_ocr_snapshot(
                        frame,
                        det,
                        boxes,
                        source,
                        output_dir,
                        now_ts,
                    )
                except Exception as e:
                    err = str(e)

                with state["lock"]:
                    state["result"] = (out_path, text, conf, err, time.time())
                    state["busy"] = False

        thread = threading.Thread(target=_worker, daemon=True)
        state["thread"] = thread
        thread.start()
        return state

    def _submit_debug_snapshot_task(self, worker_state, frame, det, boxes, source, output_dir, now_ts):
        """Encola snapshot OCR de depuraciÃ³n, conservando el mÃ¡s reciente."""
        with worker_state["lock"]:
            dropped = False
            max_pending = int(worker_state.get("max_pending", 1))
            while len(worker_state["pending_queue"]) >= max_pending:
                worker_state["pending_queue"].popleft()
                dropped = True

            worker_state["pending_queue"].append(
                (
                    frame,
                    dict(det),
                    tuple(boxes),
                    source,
                    output_dir,
                    now_ts,
                )
            )
            return "replaced" if dropped else "queued"

    def _pop_debug_snapshot_result(self, worker_state):
        with worker_state["lock"]:
            result = worker_state["result"]
            worker_state["result"] = None
        return result

    def _resize_for_detection(self, frame, target_width):
        """Reduce el frame para detecciÃ³n rÃ¡pida manteniendo aspecto."""
        h, w = frame.shape[:2]
        if w <= 0 or h <= 0:
            return frame
        target_width = int(max(320, target_width))
        if w <= target_width:
            return frame
        scale = target_width / float(w)
        target_height = max(1, int(round(h * scale)))
        return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)

    def _apply_camera_grayscale(self, frame):
        """Convierte la cÃ¡mara a gris manteniendo un frame BGR para el resto del pipeline."""
        if frame is None or frame.size == 0:
            return frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def _scale_detection_candidate(self, candidate, src_shape, dst_shape):
        """Escala una detecciÃ³n desde src_shape a dst_shape."""
        det, boxes, source = candidate
        src_h, src_w = src_shape[:2]
        dst_h, dst_w = dst_shape[:2]
        scale_x = float(dst_w) / float(max(1, src_w))
        scale_y = float(dst_h) / float(max(1, src_h))

        scaled_det = dict(det)
        scaled_det["x"] = float(det["x"]) * scale_x
        scaled_det["y"] = float(det["y"]) * scale_y
        scaled_det["w"] = float(det["w"]) * scale_x
        scaled_det["h"] = float(det["h"]) * scale_y

        scaled_boxes = []
        for idx, value in enumerate(boxes):
            if idx % 2 == 0:
                scaled_boxes.append(int(max(0, min(round(value * scale_x), dst_w))))
            else:
                scaled_boxes.append(int(max(0, min(round(value * scale_y), dst_h))))

        return scaled_det, tuple(scaled_boxes), source

    def _select_fast_detector_candidate(self, frame, allow_small_rescue=True):
        """Elige la mejor caja solo por el detector, sin OCR, para cÃ¡mara en vivo."""
        candidates = self._build_detector_candidates(frame, allow_small_rescue=allow_small_rescue)
        if not candidates:
            return None

        best = None
        best_score = -1.0
        for det, boxes, src in candidates:
            score = float(det.get("confidence", 0.0))
            if src == "primary":
                score += 0.01
            if score > best_score:
                best_score = score
                best = (det, boxes, src)
        return best

    def _ocr_from_detection(self, frame, det, boxes, source):
        """Corre OCR solo sobre una detecciÃ³n ya elegida."""
        h, w = frame.shape[:2]
        text_w = float(self.config.get("ocr_score_text_weight", 0.75))
        det_w = float(self.config.get("ocr_score_det_weight", 0.25))
        fb_bonus = float(self.config.get("ocr_score_fallback_bonus", 0.02))
        prefer_fb = bool(self.config.get("prefer_fallback_for_ocr", True))

        best = None
        best_score = -1e9
        roi_candidates = self._ocr_roi_candidates(det, frame.shape)
        if self.ocr is not None and str(getattr(self.ocr, "backend", "")).lower() == "rapidocr":
            max_candidates = max(1, int(self.config.get("camera_ocr_roi_candidates_max", 2)))
            roi_candidates = roi_candidates[:max_candidates]

        for scale, y_bias in roi_candidates:
            draw_w_scaled = det["w"] * scale
            draw_h_scaled = det["h"] * scale
            x1_ocr = int(max(0, det["x"] - draw_w_scaled / 2))
            center_y = det["y"] + (det["h"] * y_bias)
            y1_ocr = int(max(0, center_y - draw_h_scaled / 2))
            x2_ocr = int(min(w, det["x"] + draw_w_scaled / 2))
            y2_ocr = int(min(h, center_y + draw_h_scaled / 2))

            if x2_ocr <= x1_ocr or y2_ocr <= y1_ocr:
                continue

            roi = frame[y1_ocr:y2_ocr, x1_ocr:x2_ocr]
            if roi.size == 0:
                continue

            ocr_text, ocr_conf = self.ocr.segment_and_recognize_with_conf(roi)
            ocr_text, ocr_conf, tuned_box = self._ocr_second_pass_7char(
                frame,
                det,
                (x1_ocr, y1_ocr, x2_ocr, y2_ocr),
                ocr_text,
                ocr_conf,
            )
            is_valid, normalized = self._is_valid_plate_text(ocr_text)
            text_score = self._ocr_text_score(normalized, ocr_conf)
            if is_valid:
                text_score += 0.20

            combined = (text_w * text_score) + (det_w * float(det["confidence"]))
            if prefer_fb and source == "fallback":
                combined += fb_bonus

            if combined > best_score:
                best_score = combined
                best = {
                    "text": normalized if is_valid else ocr_text,
                    "ocr_conf": float(ocr_conf),
                    "ocr_box": tuned_box,
                    "text_score": float(text_score),
                }

        return best

    def _ocr_from_roi_fast(self, roi, det, source, ocr_box):
        """OCR rÃ¡pido sobre un Ãºnico recorte de placa para modo cÃ¡mara en tiempo real."""
        if self.ocr is None or roi is None or roi.size == 0:
            return None

        selected_roi, selected_tag = self._select_camera_ocr_variant(roi)
        if selected_roi is None or selected_roi.size == 0:
            return None

        def _looks_like_header_text(norm_text):
            if not norm_text:
                return False
            if norm_text in {"ECUADOR", "ECUADOS", "ECUADUR", "ECOADOR", "ECUAQOR"}:
                return True
            target = "ECUADOR"
            if len(norm_text) >= 5:
                upto = min(len(norm_text), len(target))
                matches = sum(1 for i in range(upto) if norm_text[i] == target[i])
                if matches >= 5:
                    return True
            return False

        ocr_text, ocr_conf = self.ocr.segment_and_recognize_with_conf(selected_roi)
        text_value = str(ocr_text).strip()
        if not text_value:
            return None

        is_valid, normalized = self._is_valid_plate_text(text_value)
        text_out = normalized if is_valid else text_value
        norm = "".join(ch for ch in str(text_out).upper() if ch.isalnum())
        has_digit = any(ch.isdigit() for ch in norm)
        is_header = _looks_like_header_text(norm)

        score = float(self._ocr_text_score(text_out, ocr_conf))
        if re.fullmatch(r"[A-Z]{3}[0-9]{3}", norm):
            score += float(self.config.get("camera_ocr_plate3_bonus", 0.20))
        elif re.fullmatch(r"[A-Z]{3}[0-9]{4}", norm):
            score += float(self.config.get("camera_ocr_plate4_bonus", 0.10))
        if selected_tag == "bottom":
            score += 0.08
        elif selected_tag == "bottom_deep":
            score += 0.12

        if is_header:
            score -= float(self.config.get("camera_ocr_header_penalty", 0.55))
        if norm.isalpha() and len(norm) >= 5:
            score -= float(self.config.get("camera_ocr_alpha_only_penalty", 0.30))

        best = {
            "text": text_out,
            "ocr_conf": float(ocr_conf),
            "ocr_box": tuple(ocr_box),
            "text_score": float(score),
            "norm": norm,
            "has_digit": has_digit,
            "is_header": is_header,
        }

        # Si sigue ganando un encabezado (p.ej. ECUADOR), reintenta en una banda mÃ¡s baja.
        if bool(best.get("is_header")):
            h, w = roi.shape[:2]
            y_retry = int(round(h * 0.62))
            y_retry = max(0, min(h - 1, y_retry))
            retry_roi = roi[y_retry:h, 0:w]
            if retry_roi.size > 0 and retry_roi.shape[0] >= 8:
                retry_text, retry_conf = self.ocr.segment_and_recognize_with_conf(retry_roi)
                retry_text = str(retry_text).strip()
                retry_norm = "".join(ch for ch in retry_text.upper() if ch.isalnum())
                if retry_text and any(ch.isdigit() for ch in retry_norm):
                    is_valid_r, normalized_r = self._is_valid_plate_text(retry_text)
                    text_out_r = normalized_r if is_valid_r else retry_text
                    best = {
                        "text": text_out_r,
                        "ocr_conf": float(retry_conf),
                        "ocr_box": tuple(ocr_box),
                        "text_score": float(self._ocr_text_score(text_out_r, retry_conf) + 0.10),
                        "norm": retry_norm,
                        "has_digit": True,
                        "is_header": False,
                    }

        # Ãšltima salvaguarda: no reportar encabezados ni texto solo alfabÃ©tico largo.
        if bool(best.get("is_header")) or (not bool(best.get("has_digit")) and len(str(best.get("norm", ""))) >= 5):
            return None

        best.pop("norm", None)
        best.pop("has_digit", None)
        best.pop("is_header", None)

        return best

    def _select_camera_ocr_variant(self, roi):
        """Selecciona una sola variante de ROI para OCR en cÃ¡mara (rÃ¡pido y determinista)."""
        if roi is None or roi.size == 0:
            return None, "none"

        # Modo solicitado: usar el recorte completo del detector como entrada OCR.
        if bool(self.config.get("camera_ocr_use_detector_box", True)):
            return roi, "full"

        if not bool(self.config.get("camera_ocr_bottom_band_enabled", True)):
            return roi, "full"

        h, w = roi.shape[:2]
        min_h = int(self.config.get("camera_ocr_bottom_band_min_height", 22))
        if h < max(8, min_h):
            return roi, "full"

        deep_top_ratio = float(self.config.get("camera_ocr_bottom_band_deep_top_ratio", 0.52))
        deep_top_ratio = max(0.0, min(0.9, deep_top_ratio))
        y1 = int(round(h * deep_top_ratio))
        y1 = max(0, min(h - 1, y1))
        deep_band = roi[y1:h, 0:w]
        if deep_band.size > 0 and deep_band.shape[0] >= 8:
            return deep_band, "bottom_deep"

        top_ratio = float(self.config.get("camera_ocr_bottom_band_top_ratio", 0.36))
        top_ratio = max(0.0, min(0.75, top_ratio))
        y0 = int(round(h * top_ratio))
        y0 = max(0, min(h - 1, y0))
        band = roi[y0:h, 0:w]
        if band.size > 0 and band.shape[0] >= 8:
            return band, "bottom"

        return roi, "full"

    def _upscale_camera_ocr_input(self, roi):
        """Reescala el ROI para OCR cuando llega pequeÃ±o desde detector en cÃ¡mara."""
        if roi is None or roi.size == 0:
            return roi
        if not bool(self.config.get("camera_ocr_upscale_enabled", True)):
            return roi

        h, w = roi.shape[:2]
        min_h = int(self.config.get("camera_ocr_upscale_min_height", 140))
        min_w = int(self.config.get("camera_ocr_upscale_min_width", 420))
        max_scale = float(self.config.get("camera_ocr_upscale_max_scale", 2.5))

        scale_h = (float(min_h) / float(max(1, h))) if h < min_h else 1.0
        scale_w = (float(min_w) / float(max(1, w))) if w < min_w else 1.0
        scale = max(scale_h, scale_w, 1.0)
        scale = min(scale, max(1.0, max_scale))

        if scale <= 1.01:
            return roi

        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        return cv2.resize(roi, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    def _enhance_camera_ocr_input(self, roi):
        """Mejora contraste y nitidez del ROI de OCR en cÃ¡mara."""
        if roi is None or roi.size == 0:
            return roi
        if not bool(self.config.get("camera_ocr_enhance_enabled", True)):
            return roi

        if len(roi.shape) == 2:
            gray = roi
        else:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        clip = float(self.config.get("camera_ocr_enhance_clahe_clip", 2.8))
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(gray)
        sigma = float(self.config.get("camera_ocr_enhance_sigma", 1.0))
        blur = cv2.GaussianBlur(clahe, (0, 0), sigma)
        amount = float(self.config.get("camera_ocr_enhance_unsharp_amount", 1.7))
        sharp = cv2.addWeighted(clahe, amount, blur, -(amount - 1.0), 0)
        sharp = np.clip(sharp, 0, 255).astype(np.uint8)
        return cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)

    def _to_bgr(self, image):
        if image is None or image.size == 0:
            return None
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image

    def _resize_tile(self, image, target_h=130, min_w=110):
        img = self._to_bgr(image)
        if img is None:
            return np.zeros((target_h, min_w, 3), dtype=np.uint8)
        h, w = img.shape[:2]
        if h <= 0 or w <= 0:
            return np.zeros((target_h, min_w, 3), dtype=np.uint8)
        new_w = max(min_w, int(round(w * (target_h / float(h)))))
        return cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_AREA)

    def _save_periodic_ocr_snapshot(self, frame, det, boxes, source, output_dir, now_ts):
        """Guarda una lÃ¡mina de depuraciÃ³n OCR: recorte, preprocesados y lectura."""
        ocr_best = None
        ocr_error = ""
        try:
            ocr_best = self._ocr_from_detection(frame, det, boxes, source)
        except Exception as e:
            ocr_error = str(e)

        x1_ocr, y1_ocr, x2_ocr, y2_ocr = boxes[4:8]
        if isinstance(ocr_best, dict) and ocr_best.get("ocr_box"):
            x1_ocr, y1_ocr, x2_ocr, y2_ocr = ocr_best["ocr_box"]

        h, w = frame.shape[:2]
        x1_ocr = int(max(0, min(w - 1, x1_ocr)))
        y1_ocr = int(max(0, min(h - 1, y1_ocr)))
        x2_ocr = int(max(x1_ocr + 1, min(w, x2_ocr)))
        y2_ocr = int(max(y1_ocr + 1, min(h, y2_ocr)))

        roi = frame[y1_ocr:y2_ocr, x1_ocr:x2_ocr]
        variants = []
        if self.ocr is not None and roi.size > 0:
            try:
                variants = self.ocr._simple_preprocess_variants(roi)
            except Exception as e:
                if not ocr_error:
                    ocr_error = str(e)

        tiles = [self._resize_tile(roi)]
        for variant in variants[:4]:
            tiles.append(self._resize_tile(variant))
        panel = cv2.hconcat(tiles)

        text = ""
        conf = 0.0
        status = "EMPTY"
        if ocr_error:
            status = "ERROR"
        if isinstance(ocr_best, dict):
            text = str(ocr_best.get("text", "")).strip()
            conf = float(ocr_best.get("ocr_conf", 0.0))
            if text:
                status = "OK"

        header_h = 64
        canvas = np.zeros((panel.shape[0] + header_h, panel.shape[1], 3), dtype=np.uint8)
        canvas[:header_h, :, :] = (24, 24, 24)
        canvas[header_h:, :, :] = panel

        ts_label = time.strftime("%Y%m%d_%H%M%S", time.localtime(now_ts))
        line1 = f"OCR DEBUG {status}  DET:{float(det.get('confidence', 0.0)):.2f}  OCR:{conf:.2f}  SRC:{source}"
        if ocr_error:
            short_err = re.sub(r"\s+", " ", ocr_error)[:90]
            line2 = f"ERR: {short_err}"
        else:
            line2 = f"TEXT: {text if text else '-'}"
        cv2.putText(canvas, line1, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, line2, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (200, 255, 200), 2, cv2.LINE_AA)

        output_dir.mkdir(parents=True, exist_ok=True)
        text_tag = re.sub(r"[^A-Z0-9-]", "", text.upper())[:14] if text else "EMPTY"
        out_path = output_dir / f"ocr_{ts_label}_{status}_{text_tag}.jpg"
        cv2.imwrite(str(out_path), canvas)
        return out_path, text, conf

    def _save_ocr_input_crop(self, roi, det, source, output_dir, now_ts, signature):
        """Guarda el recorte exacto que se envÃ­a al OCR."""
        if roi is None or roi.size == 0:
            return None

        output_dir.mkdir(parents=True, exist_ok=True)
        ts_label = time.strftime("%Y%m%d_%H%M%S", time.localtime(now_ts))
        conf = float(det.get("confidence", 0.0)) if isinstance(det, dict) else 0.0
        sig_tag = "_".join(str(x) for x in signature[:3]) if isinstance(signature, tuple) else "na"
        out_path = output_dir / f"ocr_input_{ts_label}_{source}_{conf:.2f}_{sig_tag}.jpg"
        cv2.imwrite(str(out_path), roi)
        return out_path

    def _save_plate_capture(self, frame, det_box, ocr_box, plate_text, det_conf, ocr_conf, source, output_dir, now_ts):
        """Guarda captura anotada con caja de detecciÃ³n y texto OCR."""
        if frame is None or frame.size == 0:
            return None

        x1, y1, x2, y2 = [int(v) for v in det_box[:4]]
        canvas = frame.copy()
        label = f"{str(plate_text).strip()} OCR:{float(ocr_conf):.2f}"
        canvas = self._draw_detection(canvas, x1, y1, x2, y2, label, float(det_conf))

        include_ocr_box = bool(self.config.get("camera_plate_capture_include_ocr_box", True))
        if include_ocr_box and ocr_box is not None and len(ocr_box) >= 4:
            ox1, oy1, ox2, oy2 = [int(v) for v in ocr_box[:4]]
            if ox2 > ox1 and oy2 > oy1:
                canvas = self._draw_ocr_roi(canvas, ox1, oy1, ox2, oy2)

        ts_label = time.strftime("%Y%m%d_%H%M%S", time.localtime(now_ts))
        ms = int((now_ts - int(now_ts)) * 1000)
        text_tag = re.sub(r"[^A-Z0-9-]", "", str(plate_text).upper())[:14] if plate_text else "EMPTY"
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (
            f"plate_{ts_label}_{ms:03d}_{text_tag}_{source}_d{float(det_conf):.2f}_o{float(ocr_conf):.2f}.jpg"
        )
        cv2.imwrite(str(out_path), canvas)
        return out_path
    
    def run_video_stream(self, video_source=0):
        """Ejecuta stream de video en modo ligero: detector rÃ¡pido + OCR sÃ³lo al detectar placa."""
        
        logger.info(f"Abriendo cÃ¡mara: {video_source}")
        def open_and_configure(source):
            cap_obj, be = self._open_camera(source)
            if cap_obj is None or not cap_obj.isOpened():
                return None, None
            cap_obj.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap_obj.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap_obj.set(cv2.CAP_PROP_FPS, 30)
            try:
                cap_obj.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            except cv2.error:
                pass
            return cap_obj, be

        cap, backend = open_and_configure(video_source)
        if cap is None:
            logger.error("âŒ No se pudo abrir cÃ¡mara")
            return
        
        logger.info("âœ“ CÃ¡mara abierta. Presiona 'q' para salir.")
        logger.info("âœ“ Backend cÃ¡mara: %s", backend)
        
        frame_count = 0
        start_time = time.time()
        show_window = bool(self.config.get("show_window", True))
        stop_on_plate = bool(self.config.get("camera_stop_on_plate", True))
        found_plate = False
        reader = self._start_camera_reader(cap)
        dump_ocr_input = bool(self.config.get("camera_dump_ocr_input_enabled", True))
        dump_ocr_input_only = bool(self.config.get("camera_dump_ocr_input_only", True))
        use_ocr_worker = bool(self.config.get("camera_ocr_use_worker", True)) and (not dump_ocr_input_only)
        ocr_worker = self._start_ocr_worker() if use_ocr_worker else None
        debug_enabled = bool(self.config.get("camera_debug_ocr_enabled", False))
        debug_worker = self._start_debug_snapshot_worker() if debug_enabled else None
        preview_width = int(self.config.get("camera_preview_width", 800))
        detection_stride = max(1, int(self.config.get("camera_detection_stride", 2)))
        adaptive_stride_enabled = bool(self.config.get("camera_adaptive_stride_enabled", False))
        adaptive_target_fps = max(1.0, float(self.config.get("camera_adaptive_stride_target_fps", 8.0)))
        adaptive_update_sec = max(0.8, float(self.config.get("camera_adaptive_stride_update_sec", 2.0)))
        adaptive_stride_min = max(1, int(self.config.get("camera_adaptive_stride_min", 1)))
        adaptive_stride_max = max(adaptive_stride_min, int(self.config.get("camera_adaptive_stride_max", 12)))
        effective_detection_stride = min(adaptive_stride_max, max(adaptive_stride_min, detection_stride))
        motion_skip_enabled = bool(self.config.get("camera_motion_skip_enabled", False))
        motion_skip_threshold = max(0.0, float(self.config.get("camera_motion_skip_threshold", 1.2)))
        motion_skip_max_frames = max(1, int(self.config.get("camera_motion_skip_max_frames", 8)))
        trigger_conf = float(self.config.get("camera_ocr_trigger_confidence", self.config.get("camera_trigger_confidence", 0.20)))
        debug_min_conf = float(self.config.get("camera_debug_min_confidence", 0.01))
        ocr_cooldown = float(self.config.get("camera_ocr_cooldown_sec", 0.9))
        ocr_settle = max(0.0, float(self.config.get("camera_ocr_settle_sec", 0.25)))
        debug_interval = max(1.0, float(self.config.get("camera_debug_ocr_interval_sec", 10.0)))
        debug_dir = Path(str(self.config.get("camera_debug_ocr_dir", "outputs/ocr_debug")))
        settle_drop_timeout = max(0.35, ocr_settle * 2.0)
        freeze_timeout = float(self.config.get("camera_freeze_timeout_sec", 1.8))
        reconnect_backoff = float(self.config.get("camera_reconnect_backoff_sec", 0.35))
        last_live = None
        last_live_signature = None
        last_live_time = 0.0
        last_submitted_signature = None
        last_submitted_time = 0.0
        pending_signature = None
        current_detection = None
        last_ocr_result = None
        ocr_submitted = 0
        ocr_dropped = 0
        ocr_completed = 0
        ocr_empty = 0
        last_ocr_status = "-"
        ocr_vote_enabled = bool(self.config.get("camera_ocr_temporal_vote_enabled", False))
        ocr_vote_window = max(2, int(self.config.get("camera_ocr_vote_window", 4)))
        ocr_vote_min_hits = max(1, int(self.config.get("camera_ocr_vote_min_hits", 2)))
        ocr_vote_max_age = max(1.0, float(self.config.get("camera_ocr_vote_max_age_sec", 4.0)))
        ocr_vote_force_accept_conf = float(self.config.get("camera_ocr_vote_force_accept_conf", 0.74))
        ocr_vote_history = deque(maxlen=ocr_vote_window)
        capture_enabled = bool(self.config.get("camera_plate_capture_enabled", False))
        capture_dir = Path(str(self.config.get("camera_plate_capture_dir", "outputs/plate_captures")))
        capture_cooldown = max(0.0, float(self.config.get("camera_plate_capture_cooldown_sec", 2.0)))
        capture_same_text_cooldown = max(0.0, float(self.config.get("camera_plate_capture_same_text_cooldown_sec", 6.0)))
        capture_min_ocr_conf = max(0.0, float(self.config.get("camera_plate_capture_min_ocr_conf", 0.45)))
        last_capture_time = 0.0
        last_capture_signature = None
        last_capture_text = ""
        last_capture_text_time = 0.0
        first_detect_time = None
        last_debug_snapshot_time = 0.0
        stable_candidate = None
        last_seen_frame_id = -1
        last_seen_frame_time = time.time()
        fps_window_frames = 0
        fps_window_start = time.time()
        last_stride_update = time.time()
        last_motion_gray = None
        static_skip_run = 0
        detect_cycle_count = 0
        small_rescue_every_n = max(1, int(self.config.get("camera_small_rescue_every_n_detections", 4)))
        small_rescue_only_when_no_live = bool(self.config.get("camera_small_rescue_only_when_no_live", True))
        
        try:
            while True:
                with reader["lock"]:
                    frame = None if reader["frame"] is None else reader["frame"].copy()
                    frame_id = int(reader["frame_id"])

                if frame is None:
                    if time.time() - last_seen_frame_time > freeze_timeout:
                        logger.warning("âš ï¸ Stream sin frames nuevos; reconectando cÃ¡mara...")
                        reader["running"] = False
                        cap.release()
                        time.sleep(reconnect_backoff)
                        cap, backend = open_and_configure(video_source)
                        if cap is None:
                            logger.error("âŒ ReconexiÃ³n fallida de cÃ¡mara")
                            break
                        reader = self._start_camera_reader(cap)
                        last_seen_frame_id = -1
                        last_seen_frame_time = time.time()
                    time.sleep(0.01)
                    continue

                if frame_id != last_seen_frame_id:
                    last_seen_frame_id = frame_id
                    last_seen_frame_time = time.time()
                elif time.time() - last_seen_frame_time > freeze_timeout:
                    logger.warning("âš ï¸ CÃ¡mara congelada (frame estÃ¡tico); reconectando...")
                    reader["running"] = False
                    cap.release()
                    time.sleep(reconnect_backoff)
                    cap, backend = open_and_configure(video_source)
                    if cap is None:
                        logger.error("âŒ ReconexiÃ³n fallida de cÃ¡mara")
                        break
                    reader = self._start_camera_reader(cap)
                    last_seen_frame_id = -1
                    last_seen_frame_time = time.time()
                    continue

                frame = self._prepare_camera_frame(frame)

                frame_count += 1
                fps_window_frames += 1
                loop_now = time.time()

                if adaptive_stride_enabled and (loop_now - last_stride_update) >= adaptive_update_sec:
                    window_elapsed = max(0.001, loop_now - fps_window_start)
                    window_fps = fps_window_frames / window_elapsed
                    prev_stride = effective_detection_stride
                    if window_fps < adaptive_target_fps * 0.90 and effective_detection_stride < adaptive_stride_max:
                        effective_detection_stride += 1
                    elif window_fps > adaptive_target_fps * 1.20 and effective_detection_stride > adaptive_stride_min:
                        effective_detection_stride -= 1

                    if effective_detection_stride != prev_stride:
                        logger.info(
                            "Ajuste stride: %d -> %d (fps ventana=%.1f, objetivo=%.1f)",
                            prev_stride,
                            effective_detection_stride,
                            window_fps,
                            adaptive_target_fps,
                        )

                    last_stride_update = loop_now
                    fps_window_start = loop_now
                    fps_window_frames = 0

                detection_candidate = None
                should_detect = (frame_count % effective_detection_stride == 0) or (last_live is None)
                if should_detect and motion_skip_enabled and last_live is not None:
                    gray_small = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    gray_small = cv2.resize(gray_small, (160, 90), interpolation=cv2.INTER_AREA)
                    if last_motion_gray is not None:
                        motion_score = float(cv2.mean(cv2.absdiff(gray_small, last_motion_gray))[0])
                        if motion_score < motion_skip_threshold and static_skip_run < motion_skip_max_frames:
                            should_detect = False
                            static_skip_run += 1
                        else:
                            static_skip_run = 0
                    last_motion_gray = gray_small

                if should_detect:
                    detect_cycle_count += 1
                    allow_small_rescue = (detect_cycle_count % small_rescue_every_n) == 0
                    if small_rescue_only_when_no_live and (last_live is not None or current_detection is not None):
                        allow_small_rescue = False

                    preview = self._resize_for_detection(frame, preview_width)
                    detection_candidate = self._select_fast_detector_candidate(
                        preview,
                        allow_small_rescue=allow_small_rescue,
                    )

                    if detection_candidate is not None:
                        preview_det, preview_boxes, preview_source = detection_candidate
                        det, boxes, source = self._scale_detection_candidate(
                            detection_candidate,
                            preview.shape,
                            frame.shape,
                        )
                        det_conf = float(det.get("confidence", 0.0))
                        now = time.time()
                        signature = (
                            source,
                            int(round(det["x"] / 12.0)),
                            int(round(det["y"] / 12.0)),
                            int(round(det["w"] / 18.0)),
                            int(round(det["h"] / 18.0)),
                        )

                        if det_conf >= debug_min_conf:
                            if first_detect_time is None:
                                first_detect_time = now

                        if det_conf >= trigger_conf:
                            current_detection = {
                                "det": det,
                                "boxes": boxes,
                                "source": source,
                                "signature": signature,
                            }

                            if (
                                stable_candidate is None
                                or stable_candidate.get("signature") != signature
                            ):
                                stable_candidate = {
                                    "signature": signature,
                                    "first_seen": now,
                                    "last_seen": now,
                                    "frame": frame.copy(),
                                    "det": det,
                                    "boxes": boxes,
                                    "source": source,
                                }
                            else:
                                stable_candidate["last_seen"] = now
                                stable_candidate["frame"] = frame.copy()
                                stable_candidate["det"] = det
                                stable_candidate["boxes"] = boxes
                                stable_candidate["source"] = source

                            stable_ready = (now - stable_candidate["first_seen"]) >= ocr_settle
                            needs_ocr = (
                                last_submitted_signature is None
                                or signature != last_submitted_signature
                                or (now - last_submitted_time) >= ocr_cooldown
                            )
                            if needs_ocr and stable_ready:
                                use_detector_box = bool(self.config.get("camera_ocr_use_detector_box", True))
                                if use_detector_box:
                                    x1_ocr, y1_ocr, x2_ocr, y2_ocr = stable_candidate["boxes"][:4]
                                else:
                                    x1_ocr, y1_ocr, x2_ocr, y2_ocr = stable_candidate["boxes"][4:8]

                                if use_detector_box:
                                    expand = float(self.config.get("camera_ocr_detector_box_expand", 1.18))
                                    if expand > 1.0:
                                        cx = (x1_ocr + x2_ocr) / 2.0
                                        cy = (y1_ocr + y2_ocr) / 2.0
                                        bw = max(2.0, (x2_ocr - x1_ocr) * expand)
                                        bh = max(2.0, (y2_ocr - y1_ocr) * expand)
                                        x1_ocr = int(round(cx - (bw / 2.0)))
                                        y1_ocr = int(round(cy - (bh / 2.0)))
                                        x2_ocr = int(round(cx + (bw / 2.0)))
                                        y2_ocr = int(round(cy + (bh / 2.0)))

                                sh, sw = stable_candidate["frame"].shape[:2]
                                x1_ocr = int(max(0, min(sw - 1, x1_ocr)))
                                y1_ocr = int(max(0, min(sh - 1, y1_ocr)))
                                x2_ocr = int(max(x1_ocr + 1, min(sw, x2_ocr)))
                                y2_ocr = int(max(y1_ocr + 1, min(sh, y2_ocr)))
                                roi = stable_candidate["frame"][y1_ocr:y2_ocr, x1_ocr:x2_ocr]

                                if roi.size != 0:
                                    ocr_variant, _ = self._select_camera_ocr_variant(roi)
                                    if ocr_variant is None or ocr_variant.size == 0:
                                        continue
                                    ocr_variant = self._upscale_camera_ocr_input(ocr_variant)
                                    if ocr_variant is None or ocr_variant.size == 0:
                                        continue
                                    ocr_variant = self._enhance_camera_ocr_input(ocr_variant)
                                    if ocr_variant is None or ocr_variant.size == 0:
                                        continue

                                    if dump_ocr_input:
                                        crop_path = self._save_ocr_input_crop(
                                            ocr_variant,
                                            stable_candidate["det"],
                                            stable_candidate["source"],
                                            debug_dir,
                                            now,
                                            stable_candidate["signature"],
                                        )
                                        if crop_path is not None:
                                            logger.info("OCR input crop guardado: %s", crop_path)

                                    if dump_ocr_input_only:
                                        ocr_submitted += 1
                                        ocr_completed += 1
                                        last_ocr_status = "DUMP"
                                        last_submitted_signature = stable_candidate["signature"]
                                        last_submitted_time = now
                                    elif use_ocr_worker and ocr_worker is not None:
                                        queue_state = self._submit_ocr_task(
                                            ocr_worker,
                                            stable_candidate["signature"],
                                            ocr_variant,
                                            stable_candidate["det"],
                                            stable_candidate["source"],
                                            stable_candidate["boxes"][:4],
                                            (x1_ocr, y1_ocr, x2_ocr, y2_ocr),
                                        )
                                        if queue_state in ("queued", "replaced"):
                                            ocr_submitted += 1
                                            if queue_state == "replaced":
                                                ocr_dropped += 1
                                            pending_signature = stable_candidate["signature"]
                                            last_submitted_signature = stable_candidate["signature"]
                                            last_submitted_time = now
                                else:
                                    # En modo cÃ¡mara se recomienda worker asÃ­ncrono para no congelar la UI.
                                    pass
                    elif stable_candidate is not None:
                        if (time.time() - float(stable_candidate.get("last_seen", 0.0))) >= settle_drop_timeout:
                            stable_candidate = None

                if debug_enabled and debug_worker is not None:
                    debug_result = self._pop_debug_snapshot_result(debug_worker)
                    if debug_result is not None:
                        out_path, snap_text, snap_conf, snap_err, _ = debug_result
                        if snap_err:
                            logger.warning("âš ï¸ Snapshot OCR con error: %s | %s", out_path, snap_err)
                        else:
                            logger.info(
                                "OCR debug guardado: %s | text=%s | conf=%.2f",
                                out_path,
                                snap_text if snap_text else "-",
                                snap_conf,
                            )

                if use_ocr_worker and ocr_worker is not None:
                    ocr_result = self._pop_ocr_result(ocr_worker)
                    if ocr_result is not None:
                        signature, det, source, det_box, ocr_best, finished_at = ocr_result
                        ocr_completed += 1
                        pending_signature = None if pending_signature == signature else pending_signature
                        text_value = ""
                        if isinstance(ocr_best, dict):
                            text_value = str(ocr_best.get("text", "")).strip()

                        if ocr_best is not None and text_value:
                            accept_result = True
                            accepted_text = text_value
                            accepted_conf = float(ocr_best["ocr_conf"])

                            if ocr_vote_enabled:
                                ocr_vote_history.append((finished_at, text_value, float(ocr_best["ocr_conf"])))
                                valid_hist = [
                                    (ts, txt, conf)
                                    for ts, txt, conf in ocr_vote_history
                                    if (finished_at - ts) <= ocr_vote_max_age
                                ]
                                if len(valid_hist) != len(ocr_vote_history):
                                    ocr_vote_history = deque(valid_hist, maxlen=ocr_vote_window)

                                hits = 0
                                best_conf_for_text = 0.0
                                for _ts, txt, conf in ocr_vote_history:
                                    if txt == text_value:
                                        hits += 1
                                        best_conf_for_text = max(best_conf_for_text, float(conf))

                                if accepted_conf >= ocr_vote_force_accept_conf:
                                    accept_result = True
                                elif hits >= ocr_vote_min_hits:
                                    accept_result = True
                                    accepted_conf = max(accepted_conf, best_conf_for_text)
                                else:
                                    accept_result = False

                            if not accept_result:
                                last_ocr_status = "HOLD"
                                continue

                            last_live = {
                                "det": det,
                                "boxes": tuple(det_box) + tuple(ocr_best.get("ocr_box", (0, 0, 0, 0))),
                                "source": source,
                                "text": accepted_text,
                                "ocr_conf": accepted_conf,
                                "ocr_box": ocr_best["ocr_box"],
                            }
                            last_ocr_result = {
                                "signature": signature,
                                "text": accepted_text,
                            }
                            last_live_signature = signature
                            last_live_time = finished_at
                            last_ocr_status = "OK"
                            found_plate = True
                            logger.info(
                                "Placa: %s (det: %.2f, ocr: %.2f)",
                                last_live["text"],
                                float(det["confidence"]),
                                float(last_live["ocr_conf"]),
                            )

                            if capture_enabled and float(last_live["ocr_conf"]) >= capture_min_ocr_conf:
                                save_now = float(finished_at)
                                can_save = True
                                if last_capture_signature == signature and (save_now - last_capture_time) < capture_cooldown:
                                    can_save = False
                                if (
                                    can_save
                                    and last_capture_text == str(last_live["text"])
                                    and (save_now - last_capture_text_time) < capture_same_text_cooldown
                                ):
                                    can_save = False

                                if can_save:
                                    saved_path = self._save_plate_capture(
                                        frame.copy(),
                                        tuple(det_box),
                                        tuple(last_live.get("ocr_box", (0, 0, 0, 0))),
                                        str(last_live["text"]),
                                        float(det["confidence"]),
                                        float(last_live["ocr_conf"]),
                                        str(source),
                                        capture_dir,
                                        save_now,
                                    )
                                    if saved_path is not None:
                                        last_capture_time = save_now
                                        last_capture_signature = signature
                                        last_capture_text = str(last_live["text"])
                                        last_capture_text_time = save_now
                                        logger.info("Captura guardada: %s", saved_path)
                        else:
                            ocr_empty += 1
                            last_ocr_status = "EMPTY"

                display_frame = frame.copy()
                active_detection = current_detection if current_detection is not None else last_live
                if debug_enabled and active_detection is not None and self.ocr_enabled and self.ocr is not None and debug_worker is not None:
                    now = time.time()
                    should_snapshot = (
                        last_debug_snapshot_time <= 0.0
                        or (now - last_debug_snapshot_time) >= debug_interval
                    )
                    if should_snapshot:
                        debug_state = self._submit_debug_snapshot_task(
                            debug_worker,
                            frame.copy(),
                            active_detection["det"],
                            active_detection["boxes"],
                            active_detection.get("source", "active"),
                            debug_dir,
                            now,
                        )
                        if debug_state in ("queued", "replaced"):
                            last_debug_snapshot_time = now

                if active_detection is not None:
                    det = active_detection["det"]
                    boxes = active_detection["boxes"]
                    x1, y1, x2, y2 = boxes[:4]

                    label_text = f"DET {float(det['confidence']):.2f}"
                    if last_ocr_result is not None and active_detection.get("signature") == last_ocr_result.get("signature"):
                        label_text = f"{last_ocr_result['text']} {float(det['confidence']):.2f}"
                    elif isinstance(last_live, dict) and last_live.get("text"):
                        # En modo asÃ­ncrono la firma puede cambiar entre frames y ocultar OCR vÃ¡lido.
                        label_text = f"{last_live['text']} {float(det['confidence']):.2f}"

                    display_frame = self._draw_detection(
                        display_frame,
                        x1,
                        y1,
                        x2,
                        y2,
                        label_text,
                        float(det["confidence"]),
                    )
                else:
                    display_frame = frame

                ocr_backend = "off"
                if self.ocr_enabled and self.ocr is not None:
                    ocr_backend = str(getattr(self.ocr, "backend", "model"))

                ocr_state = "IDLE"
                if use_ocr_worker and pending_signature is not None:
                    ocr_state = "RUNNING"
                elif not use_ocr_worker and active_detection is not None:
                    ocr_state = "SYNC"
                elif active_detection is not None:
                    ocr_state = "WAITING_RESULT"

                queue_depth = 0
                in_flight = False
                if use_ocr_worker and ocr_worker is not None:
                    with ocr_worker["lock"]:
                        queue_depth = len(ocr_worker["pending_queue"])
                        in_flight = ocr_worker.get("in_flight_signature") is not None

                last_text = "-"
                last_conf = 0.0
                if isinstance(last_live, dict) and last_live.get("text"):
                    last_text = str(last_live.get("text", "-"))
                    last_conf = float(last_live.get("ocr_conf", 0.0))

                overlay_y = max(22, display_frame.shape[0] - 44)
                overlay_text_1 = f"OCR[{ocr_backend}] {ocr_state} Q:{queue_depth} IF:{'1' if in_flight else '0'}"
                overlay_text_2 = f"LAST:{last_text}({last_conf:.2f}) ST:{last_ocr_status} OK:{ocr_completed-ocr_empty}/{ocr_completed} ENQ:{ocr_submitted} DROP:{ocr_dropped}"
                cv2.putText(
                    display_frame,
                    overlay_text_1,
                    (10, overlay_y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255) if pending_signature is not None else (200, 255, 200),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    display_frame,
                    overlay_text_2,
                    (10, overlay_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255) if last_text != "-" else (180, 180, 180),
                    2,
                    cv2.LINE_AA,
                )

                display_out = display_frame
                display_width = int(self.config.get("camera_display_width", 0))
                if display_width > 0 and display_frame.shape[1] > display_width:
                    scale = display_width / float(display_frame.shape[1])
                    display_h = max(1, int(round(display_frame.shape[0] * scale)))
                    display_out = cv2.resize(
                        display_frame,
                        (display_width, display_h),
                        interpolation=cv2.INTER_AREA,
                    )

                # Mostrar (si HighGUI estÃ¡ disponible)
                if show_window:
                    try:
                        cv2.imshow("License Plate Detection", display_out)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                    except cv2.error as e:
                        show_window = False
                        logger.warning(
                            "âš ï¸ OpenCV HighGUI no disponible (%s). "
                            "Continuando sin ventana.",
                            e
                        )
                elif frame_count % 120 == 0:
                    logger.info("Procesando sin ventana... frame=%d", frame_count)
        
        finally:
            reader["running"] = False
            if ocr_worker is not None:
                ocr_worker["running"] = False
            if debug_worker is not None:
                debug_worker["running"] = False
            cap.release()
            if show_window:
                try:
                    cv2.destroyAllWindows()
                except cv2.error:
                    pass
            
            elapsed = time.time() - start_time
            logger.info(f"\nâœ“ Procesados {frame_count} frames en {elapsed:.1f}s")
            logger.info(f"âœ“ FPS promedio: {frame_count/elapsed:.1f}")

    def run_single_image(self, image_path, output_path=None, show_window=True):
        """Procesa una sola imagen y termina."""
        image_path = Path(image_path)
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError(f"No se pudo leer la imagen: {image_path}")

        annotated, detections = self.process_frame(frame.copy())
        if detections:
            for detection in detections:
                logger.info(
                    "Placa: %s (det: %.2f, fuente: %s)",
                    detection.get("text", ""),
                    float(detection.get("confidence", 0.0)),
                    detection.get("source", ""),
                )
        else:
            logger.info("No se detectaron placas en %s", image_path.name)

        if output_path:
            output_path = Path(output_path)
        else:
            output_path = image_path.with_name(f"{image_path.stem}_annotated{image_path.suffix}")

        cv2.imwrite(str(output_path), annotated)
        logger.info("âœ“ Imagen anotada guardada en %s", output_path)

        if show_window:
            try:
                cv2.imshow("License Plate Detection", annotated)
                cv2.waitKey(0)
            except cv2.error as e:
                logger.warning(
                    "âš ï¸ OpenCV HighGUI no disponible (%s). Se guardÃ³ la imagen anotada.",
                    e,
                )
            finally:
                try:
                    cv2.destroyAllWindows()
                except cv2.error:
                    pass

        return annotated, detections

    def run_camera_healthcheck(self, video_source=0, seconds=15, show_window=True, save_dir=None):
        """Verifica captura y visualizaciÃ³n sin detector ni OCR."""
        logger.info(f"Healthcheck de cÃ¡mara: {video_source}")
        cap, backend = self._open_camera(video_source)
        if cap is None or not cap.isOpened():
            logger.error("âŒ No se pudo abrir cÃ¡mara en healthcheck")
            return

        logger.info("âœ“ Healthcheck abierto. Backend cÃ¡mara: %s", backend)
        start = time.time()
        frame_count = 0
        saved_count = 0
        save_path = Path(save_dir) if save_dir else None
        if save_path is not None:
            save_path.mkdir(parents=True, exist_ok=True)

        try:
            while (time.time() - start) < float(seconds):
                ret, frame = cap.read()
                if not ret or frame is None:
                    logger.warning("âš ï¸ Healthcheck: no se pudo leer frame")
                    time.sleep(0.02)
                    continue

                frame = self._prepare_camera_frame(frame)
                frame_count += 1

                if frame_count % 30 == 0:
                    logger.info(
                        "healthcheck frame=%d shape=%s dtype=%s min=%d max=%d mean=%.1f",
                        frame_count,
                        frame.shape,
                        frame.dtype,
                        int(frame.min()),
                        int(frame.max()),
                        float(frame.mean()),
                    )

                if save_path is not None and frame_count % 15 == 0:
                    out_file = save_path / f"camera_healthcheck_{frame_count:05d}.jpg"
                    cv2.imwrite(str(out_file), frame)
                    saved_count += 1

                if show_window:
                    try:
                        cv2.imshow("Camera Healthcheck", frame)
                        if (cv2.waitKey(1) & 0xFF) == ord('q'):
                            break
                    except cv2.error as e:
                        logger.warning("âš ï¸ HighGUI no disponible en healthcheck (%s)", e)
                        show_window = False

        finally:
            cap.release()
            if show_window:
                try:
                    cv2.destroyAllWindows()
                except cv2.error:
                    pass

            elapsed = max(0.001, time.time() - start)
            logger.info(
                "âœ“ Healthcheck terminado: frames=%d saved=%d elapsed=%.1fs fps=%.1f",
                frame_count,
                saved_count,
                elapsed,
                frame_count / elapsed,
            )


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="License Plate Detection en Jetson Nano"
    )
    parser.add_argument(
        "--video",
        type=str,
        default="0",
        help="Fuente de video (0 para cÃ¡mara, o ruta a archivo)"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Ruta a una imagen para una ejecuciÃ³n Ãºnica"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Ruta opcional para guardar la imagen anotada en modo --image"
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Desactiva ventana OpenCV (Ãºtil en Jetson headless)"
    )
    parser.add_argument(
        "--camera-healthcheck",
        action="store_true",
        help="Abre solo la cÃ¡mara y muestra/guarda frames sin detector ni OCR"
    )
    parser.add_argument(
        "--camera-healthcheck-seconds",
        type=int,
        default=15,
        help="DuraciÃ³n del healthcheck de cÃ¡mara en segundos"
    )
    parser.add_argument(
        "--camera-healthcheck-save-dir",
        type=str,
        default=None,
        help="Directorio opcional para guardar frames del healthcheck"
    )
    parser.add_argument(
        "--runtime-profile",
        type=str,
        default="current",
        choices=["current", "jetson-stable", "jetson-quality", "jetson-production", "jetson-realtime"],
        help="Perfil de ejecuciÃ³n para edge: current, jetson-stable, jetson-quality, jetson-production o jetson-realtime"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models/optimized",
        help="Directorio de modelos optimiz ados"
    )
    parser.add_argument(
        "--detector-model",
        type=str,
        default=None,
        help="Ruta explÃ­cita del detector (.pt/.h5/.tflite). Si no se define, se mantiene el de CONFIG"
    )
    parser.add_argument(
        "--ocr-model",
        type=str,
        default=None,
        help="Ruta explÃ­cita del OCR. Si no se define, usa <model-dir>/ocr_int8.tflite"
    )
    parser.add_argument(
        "--ocr-engine",
        type=str,
        default=None,
        choices=["rapidocr-lpr", "paddle-latin", "ctc", "simple", "tesseract"],
        help="Motor OCR: rapidocr-lpr (recomendado), paddle-latin, ctc, simple o tesseract"
    )
    parser.add_argument(
        "--ocr-profile",
        type=str,
        default=None,
        choices=["balanced", "far"],
        help="Perfil OCR: balanced para cercanas/mixtas, far para cÃ¡maras lejanas"
    )
    parser.add_argument(
        "--camera-grayscale",
        action="store_true",
        help="Convierte la cÃ¡mara a escala de grises antes de detectar y leer OCR"
    )
    parser.add_argument(
        "--dump-ocr-crops",
        action="store_true",
        help="Guarda en outputs/ocr_debug el recorte exacto enviado al OCR"
    )
    parser.add_argument(
        "--dump-ocr-crops-only",
        action="store_true",
        help="Solo guarda recortes OCR y no ejecuta inferencia OCR (modo depuraciÃ³n)"
    )
    parser.add_argument(
        "--camera-ocr-expand",
        type=float,
        default=None,
        help="Factor de expansiÃ³n de la caja del detector antes de enviar al OCR (ej: 1.2)"
    )
    parser.add_argument(
        "--camera-display-width",
        type=int,
        default=None,
        help="Ancho de visualizaciÃ³n de ventana (solo render). No afecta inferencia"
    )
    
    args = parser.parse_args()
    
    # Parse video source
    try:
        video_source = int(args.video)
    except:
        video_source = args.video
    
    # Actualizar config sin perder el detector ganador por defecto.
    if args.detector_model:
        CONFIG["detector_model"] = args.detector_model

    if args.ocr_engine:
        CONFIG["ocr_engine"] = args.ocr_engine

    if args.ocr_model:
        CONFIG["ocr_model"] = args.ocr_model
    elif str(CONFIG.get("ocr_engine", "")).lower() == "ctc":
        CONFIG["ocr_model"] = f"{args.model_dir}/ocr_crnn_ctc_int8.tflite"

    if args.ocr_profile:
        CONFIG["ocr_profile"] = args.ocr_profile

    if args.camera_grayscale:
        CONFIG["camera_input_grayscale"] = True

    if args.dump_ocr_crops:
        CONFIG["camera_dump_ocr_input_enabled"] = True
    if args.dump_ocr_crops_only:
        CONFIG["camera_dump_ocr_input_enabled"] = True
        CONFIG["camera_dump_ocr_input_only"] = True

    if args.camera_ocr_expand is not None:
        CONFIG["camera_ocr_detector_box_expand"] = max(1.0, float(args.camera_ocr_expand))

    if args.camera_display_width is not None:
        CONFIG["camera_display_width"] = max(0, int(args.camera_display_width))

    if args.no_window:
        CONFIG["show_window"] = False

    apply_runtime_profile(CONFIG, args.runtime_profile)
    
    # Pipeline
    pipeline = PipelineJetson(CONFIG)
    if args.camera_healthcheck:
        pipeline.run_camera_healthcheck(
            video_source,
            seconds=int(args.camera_healthcheck_seconds),
            show_window=bool(CONFIG.get("show_window", True)),
            save_dir=args.camera_healthcheck_save_dir,
        )
        return

    if args.image:
        pipeline.run_single_image(args.image, args.output, show_window=bool(CONFIG.get("show_window", True)))
    else:
        pipeline.run_video_stream(video_source)


if __name__ == "__main__":
    print("\n" + "ðŸš€" * 20)
    print(" JETSON NANO - REAL-TIME LICENSE PLATE DETECTION")
    print("ðŸš€" * 20 + "\n")
    
    main()
    
    print("""
    
    âœ¨ INFERENCE COMPLETADO
    
    Para monitorear recursos en tiempo real (en otra terminal):
        watch -n 0.5 'nvidia-smi'
    
    Logs Ãºtiles:
        dmesg | tail -20  (para errores del kernel)
        dpkg -l | grep cuda  (verificar CUDA)
    """)

