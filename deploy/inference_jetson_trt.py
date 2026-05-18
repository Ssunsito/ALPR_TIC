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
            f"TensorFlow no está instalado, pero se intentó usar {feature_name}. "
            "Usa modelos YOLO (.pt) y OCR RapidOCR/Tesseract, o instala TensorFlow en este entorno."
        )

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

CONFIG = {
    "detector_model": "models/plate_detector_best.pt",
    "detector_fallback_model": "models/plate_detector_best.pt",
    "enable_detector_fallback": True,
    "ocr_model": "models/optimized/ocr_crnn_ctc_int8.tflite",
    "ocr_engine": "rapidocr-lpr",
    "ocr_simple_fallback_to_model": True,
    "ocr_paddle_lang": "latin",
    "ocr_paddle_use_angle_cls": False,
    "ocr_paddle_cls": False,
    "ocr_paddle_det": False,
    "ocr_paddle_show_log": False,
    "ocr_rapid_min_conf": 0.20,
    "ocr_plate_postprocess_enabled": True,
    "ocr_plate_postprocess_strict": True,
    "confidence_threshold": 0.8,
    "fallback_confidence_threshold": 0.20,
    "detector_preprocess_enabled": True,
    "detector_preprocess_clahe_clip": 2.4,
    "iou_threshold": 0.5,
    "input_size": 416,
    "yolo_imgsz": 640,
    "min_area_ratio": 0.002,
    "min_area_ratio_small": 0.0006,
    "max_area_ratio": 1.0,
    "min_aspect_ratio": 0.25,
    "max_aspect_ratio": 8.0,
    "plate_box_scale": 1.0,
    "ocr_box_scale": 0.95,
    "ocr_aware_box_selection": True,
    "prefer_fallback_for_ocr": True,
    "ocr_score_text_weight": 0.75,
    "ocr_score_det_weight": 0.25,
    "ocr_score_fallback_bonus": 0.02,
    "ocr_second_pass_7char_enabled": True,
    "ocr_second_pass_conf_threshold": 0.085,
    "ocr_second_pass_pattern_bonus": 0.16,
    "ctc_len_bonus": 0.12,
    "ctc_len7_bonus": 0.08,
    "ctc_pattern_bonus": 0.25,
    "ctc_no_digit_penalty": 0.15,
    "ctc_no_letter_penalty": 0.15,
    "ctc_vs_char_margin": 0.02,
    "ocr_profile": "balanced",
    "ocr_far_plate_threshold": 160,
    "ocr_far_plate_upscale": 3.0,
    "ocr_far_plate_upscale_threshold": 90,
    "ocr_far_plate_clahe_clip": 6.0,
    "ocr_far_num_chars_min": 5,
    "ocr_far_num_chars_max": 9,
    "ocr_far_char_conf_threshold": 0.25,
    "ocr_far_plate_denoise": True,
    "ocr_far_plate_morph": True,
    "enable_small_plate_rescue": True,
    "small_plate_rescue_only_on_miss": True,
    "small_plate_rescue_upscale": 1.8,
    "small_plate_rescue_imgsz": 960,
    "small_plate_rescue_conf": 0.08,
    "small_plate_rescue_iou": 0.5,
    "small_plate_rescue_accept_conf": 0.20,
    "small_plate_rescue_tiling": True,
    "small_plate_rescue_tile_grid": 2,
    "small_plate_rescue_tile_overlap": 0.22,
    # OCR params para placas pequeñas
    "ocr_small_plate_threshold": 100,  # píxeles mínimos de height para activar estrategia agresiva
    "ocr_small_plate_upscale": 2.0,  # upscale agresivo para placas muy pequeñas (<50px)
    "ocr_small_plate_upscale_threshold": 50,  # height threshold para upscale agresivo
    "ocr_small_plate_clahe_clip": 4.0,  # CLAHE más fuerte para placas pequeñas (vs 2.0 normal)
    "ocr_small_plate_denoise": True,  # aplica fastNlMeansDenoising para placas pequeñas
    "ocr_small_plate_morph": True,  # aplica operaciones morfológicas
    "ocr_char_topk": 3,
    "ocr_plate_pattern_bonus": 0.22,
    "ocr_plate_hyphen_bonus": 0.08,
    "ocr_plate_beam_width": 24,
    "ocr_pos_correction_enabled": True,
    "ocr_pos_confusion_mix": 0.45,
    "ocr_pos_low_conf_threshold": 0.30,
    "ocr_pos_max_changes": 3,
    "ocr_pos_last_letter_xc_swap": True,
    "ocr_pos_last_digit_24_swap": True,
    "ocr_box_scale_close": 0.78,
    "ocr_box_scale_medium": 0.88,
    "ocr_box_scale_small": 0.95,
    "ocr_roi_y_bias_close": 0.10,
    "ocr_roi_y_bias_medium": 0.06,
    "ocr_roi_y_bias_small": 0.03,
    "ocr_tesseract_cmd": r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    "fps_buffer_size": 30,  # para calcular FPS promedio
    "camera_index": 0,  # /dev/video0 en Jetson
    "camera_stop_on_plate": False,
    "camera_preview_width": 800,
    "camera_display_width": 800,
    "camera_detection_stride": 2,
    "camera_trigger_confidence": 0.20,
    "camera_ocr_trigger_confidence": 0.20,
    "camera_debug_min_confidence": 0.01,
    "camera_ocr_cooldown_sec": 0.9,
    "camera_ocr_max_pending": 2,
    "camera_ocr_settle_sec": 0.25,
    "camera_ocr_roi_candidates_max": 2,
    "camera_ocr_use_worker": True,
    "camera_ocr_use_detector_box": True,
    "camera_ocr_detector_box_expand": 1.18,
    "camera_ocr_bottom_band_enabled": False,
    "camera_ocr_upscale_enabled": True,
    "camera_ocr_upscale_min_height": 140,
    "camera_ocr_upscale_min_width": 420,
    "camera_ocr_upscale_max_scale": 2.5,
    "camera_ocr_enhance_enabled": True,
    "camera_ocr_enhance_clahe_clip": 2.8,
    "camera_ocr_enhance_sigma": 1.0,
    "camera_ocr_enhance_unsharp_amount": 1.7,
    "camera_ocr_bottom_band_top_ratio": 0.36,
    "camera_ocr_bottom_band_deep_top_ratio": 0.52,
    "camera_ocr_bottom_band_min_height": 22,
    "camera_ocr_header_penalty": 0.55,
    "camera_ocr_alpha_only_penalty": 0.30,
    "camera_ocr_plate3_bonus": 0.20,
    "camera_ocr_plate4_bonus": 0.10,
    "ocr_rapid_max_variants": 2,
    "camera_debug_ocr_enabled": False,
    "camera_dump_ocr_input_enabled": False,
    "camera_dump_ocr_input_only": False,
    "camera_debug_ocr_interval_sec": 10.0,
    "camera_debug_ocr_dir": "outputs/ocr_debug",
    "camera_debug_max_pending": 1,
    "camera_freeze_timeout_sec": 1.8,
    "camera_reconnect_backoff_sec": 0.35,
    "camera_input_grayscale": False,
    "camera_capture_width": 1280,
    "camera_capture_height": 720,
    "camera_capture_fps": 30,
    "use_gpu": True,  # TensorRT si está disponible
    "show_window": True,
}


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
                    "ultralytics no está instalado, pero se intentó cargar un modelo YOLO"
                )
            self.backend = "yolo"
            self.input_size = 640
            self.yolo_model = YOLO(model_path_str)
            logger.info("✓ Detector listo (backend=yolo)")
            return

        # Si llega un modelo Keras, úsalo directo para evitar dependencia Flex ops
        if model_path_str.lower().endswith((".h5", ".keras")):
            _require_tensorflow("detector backend=keras")
            self.backend = "keras"
            self.model = tf.keras.models.load_model(model_path, compile=False)
            logger.info("✓ Detector listo (backend=keras)")
            return

        # SavedModel directory fallback (compatible con TF 2.21 sin Keras deserialization)
        if Path(model_path).is_dir():
            _require_tensorflow("detector backend=saved_model")
            self.backend = "saved_model"
            self.saved_model = tf.saved_model.load(model_path)
            self.saved_model_fn = self.saved_model.signatures["serving_default"]
            logger.info("✓ Detector listo (backend=saved_model)")
            return

        # Camino estándar TFLite
        _require_tensorflow("detector backend=tflite")
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        logger.info("✓ Detector listo (backend=tflite)")
    
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


class OCRInference:
    """OCR optimizado (char-classifier o secuencial CRNN+CTC)."""
    
    def __init__(self, model_path, char_set=PLATE_CHARS, input_size=32):
        logger.info(f"Cargando OCR: {model_path}")
        self.engine = str(CONFIG.get("ocr_engine", "simple")).lower()
        self.backend = "tflite"
        self.interpreter = None
        self.keras_model = None
        self.input_details = None
        self.output_details = None
        self.char_fallback_interpreter = None
        self.char_fallback_input_details = None
        self.char_fallback_output_details = None
        self.char_set = char_set
        self.input_size = input_size
        self.ocr_mode = "char"
        self.paddle_ocr = None
        self.rapid_ocr = None

        model_path = str(model_path)
        model_suffix = Path(model_path).suffix.lower()
        explicit_keras = model_suffix in {".keras", ".h5"}

        if self.engine in {"rapidocr", "rapidocr-lpr", "rapid"}:
            if RapidOCR is None:
                logger.warning("⚠️ RapidOCR no disponible. Se usará fallback OCR existente.")
            else:
                try:
                    self.rapid_ocr = RapidOCR()
                    self.backend = "rapidocr"
                    self.ocr_mode = "rapid"
                    logger.info("✓ OCR listo (backend=rapidocr, modo=lpr)")
                    return
                except Exception as e:
                    logger.warning("⚠️ No se pudo inicializar RapidOCR: %s", e)

        if self.engine in {"paddle", "paddle-latin", "paddle_latin"}:
            if PaddleOCR is None:
                logger.warning("⚠️ PaddleOCR no disponible. Se usará fallback OCR existente.")
            else:
                try:
                    self.paddle_ocr = PaddleOCR(
                        use_angle_cls=bool(CONFIG.get("ocr_paddle_use_angle_cls", False)),
                        lang=str(CONFIG.get("ocr_paddle_lang", "latin")),
                        det=bool(CONFIG.get("ocr_paddle_det", False)),
                        rec=True,
                        cls=bool(CONFIG.get("ocr_paddle_cls", False)),
                        show_log=bool(CONFIG.get("ocr_paddle_show_log", False)),
                    )
                    self.backend = "paddleocr"
                    self.ocr_mode = "paddle"
                    logger.info("✓ OCR listo (backend=paddleocr, modo=latin)")
                    return
                except Exception as e:
                    logger.warning("⚠️ No se pudo inicializar PaddleOCR: %s", e)

        if self.engine in {"simple", "tesseract", "paddle", "paddle-latin", "paddle_latin", "rapidocr", "rapidocr-lpr", "rapid"}:
            if pytesseract is not None:
                tesseract_cmd = _resolve_tesseract_cmd()
                if tesseract_cmd is not None:
                    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
                self.backend = "tesseract"
                self.ocr_mode = "simple"
                logger.info("✓ OCR listo (backend=tesseract, modo=simple)")
                return
            if not bool(CONFIG.get("ocr_simple_fallback_to_model", True)):
                raise RuntimeError("pytesseract no disponible y ocr_simple_fallback_to_model=False")
            logger.warning("⚠️ pytesseract no disponible. Se usará OCR del modelo como fallback.")

        if explicit_keras:
            _require_tensorflow("ocr backend=keras")
            self.backend = "keras"
            self.keras_model = tf.keras.models.load_model(
                model_path,
                compile=False,
                custom_objects={"CTCLayer": CTCLayer},
            )
            if isinstance(self.keras_model, tf.keras.Model) and len(self.keras_model.inputs) > 1:
                self.keras_model = tf.keras.Model(
                    inputs=self.keras_model.inputs[0],
                    outputs=self.keras_model.get_layer("sequence_probs").output,
                    name="crnn_ctc_pred_fallback",
                )
        else:
            try:
                _require_tensorflow("ocr backend=tflite")
                self.interpreter = tf.lite.Interpreter(model_path=model_path)
                self.interpreter.allocate_tensors()
                self.input_details = self.interpreter.get_input_details()
                self.output_details = self.interpreter.get_output_details()
            except RuntimeError as e:
                if "Select TensorFlow op" not in str(e):
                    raise

                self.backend = "keras"
                dynamic_candidates = [
                    Path(model_path).with_suffix(".keras"),
                    Path(model_path).with_suffix(".h5"),
                ]

                stem = Path(model_path).stem
                tag_prefix = "ocr_crnn_ctc_int8_"
                if stem.startswith(tag_prefix):
                    tag = stem[len(tag_prefix):]
                    if tag:
                        dynamic_candidates.extend(
                            [
                                Path(f"models/ocr_crnn_best_{tag}.keras"),
                                Path(f"models/ocr_crnn_pred_{tag}.h5"),
                            ]
                        )

                model_candidates = [
                    *dynamic_candidates,
                    Path("models/ocr_crnn_best.keras"),
                    Path("models/ocr_crnn_pred.h5"),
                    Path(model_path).with_suffix(".h5"),
                ]
                fallback = next((p for p in model_candidates if p.exists()), None)
                if fallback is None:
                    raise
                logger.warning(
                    "⚠️ OCR TFLite requiere Flex ops. Usando fallback Keras: %s",
                    fallback,
                )
                _require_tensorflow("ocr fallback backend=keras")
                self.keras_model = tf.keras.models.load_model(
                    str(fallback),
                    compile=False,
                    custom_objects={"CTCLayer": CTCLayer},
                )
                if isinstance(self.keras_model, tf.keras.Model) and len(self.keras_model.inputs) > 1:
                    self.keras_model = tf.keras.Model(
                        inputs=self.keras_model.inputs[0],
                        outputs=self.keras_model.get_layer("sequence_probs").output,
                        name="crnn_ctc_pred_fallback",
                    )

        # Inferir tipo de modelo OCR a partir de la forma de salida.
        if self.backend == "keras":
            out_shape = self.keras_model.output_shape
        else:
            out_shape = self.output_details[0].get("shape", [])

        if len(out_shape) >= 3:
            self.ocr_mode = "ctc"
            self.blank_idx = len(self.char_set)
            self._load_char_fallback()
            logger.info("✓ OCR listo (backend=%s, modo=ctc)", self.backend)
        else:
            logger.info("✓ OCR listo (backend=%s, modo=char)", self.backend)

    def _load_char_fallback(self):
        fallback_path = Path("models/optimized/ocr_int8.tflite")
        if not fallback_path.exists():
            return
        try:
            _require_tensorflow("ocr char fallback backend=tflite")
            itp = tf.lite.Interpreter(model_path=str(fallback_path))
            itp.allocate_tensors()
            in_details = itp.get_input_details()
            out_details = itp.get_output_details()

            # Solo aceptar fallback tipo clasificador de caracteres.
            out_shape = out_details[0].get("shape", [])
            if len(out_shape) >= 3:
                return

            self.char_fallback_interpreter = itp
            self.char_fallback_input_details = in_details
            self.char_fallback_output_details = out_details
            logger.info("OCR fallback char habilitado: %s", fallback_path)
        except Exception as e:
            logger.warning("No se pudo habilitar OCR fallback char: %s", e)

    def _simple_preprocess_variants(self, plate_image):
        """Genera variantes livianas para OCR clásico."""
        if plate_image is None or plate_image.size == 0:
            return []

        if len(plate_image.shape) == 3:
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = plate_image.copy()

        h, w = gray.shape[:2]
        if w > 0 and w < 420:
            scale = 420 / float(w)
            gray = cv2.resize(gray, (420, max(1, int(h * scale))), interpolation=cv2.INTER_CUBIC)

        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
        blur = cv2.GaussianBlur(clahe, (0, 0), 0.8)
        sharp = cv2.addWeighted(clahe, 1.7, blur, -0.7, 0)
        _, th = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        inv = cv2.bitwise_not(th)
        return [gray, clahe, sharp, th, inv]

    def _simple_normalize(self, text):
        return "".join(ch for ch in str(text).upper() if ch.isalnum())

    def _simple_format(self, text):
        clean = self._simple_normalize(text)
        if re.fullmatch(r"[A-Z]{3}[0-9]{3}", clean):
            return f"{clean[:3]}-{clean[3:]}"
        if re.fullmatch(r"[A-Z]{3}[0-9]{4}", clean):
            return f"{clean[:3]}-{clean[3:]}"
        return clean

    def _postprocess_plate_text(self, text):
        """Normaliza OCR a formato de placa esperado con correcciones posicionales."""
        if not bool(CONFIG.get("ocr_plate_postprocess_enabled", True)):
            return self._simple_format(text)

        clean = self._simple_normalize(text)
        if not clean:
            return ""

        strict = bool(CONFIG.get("ocr_plate_postprocess_strict", True))

        if re.fullmatch(r"[A-Z]{3}[0-9]{3,4}", clean):
            return self._simple_format(clean)

        if len(clean) not in (6, 7):
            return self._simple_format(clean)

        letters_raw = clean[:3]
        digits_raw = clean[3:]

        letters_fix = []
        for ch in letters_raw:
            if ch.isalpha():
                letters_fix.append(ch)
                continue
            mapped = DIGIT_TO_LETTER_VISUAL.get(ch, "")
            if mapped and mapped.isalpha():
                letters_fix.append(mapped)
            else:
                return self._simple_format(clean)

        digits_fix = []
        for ch in digits_raw:
            if ch.isdigit():
                digits_fix.append(ch)
                continue
            mapped = LETTER_TO_DIGIT_VISUAL.get(ch, "")
            if mapped and mapped.isdigit():
                digits_fix.append(mapped)
            else:
                return self._simple_format(clean)

        candidate = "".join(letters_fix) + "".join(digits_fix)
        if re.fullmatch(r"[A-Z]{3}[0-9]{3,4}", candidate):
            return self._simple_format(candidate)

        return self._simple_format(clean)

    def _simple_ocr_on_image(self, image):
        if pytesseract is None:
            return "", 0.0

        config = "--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        try:
            data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config=config)
        except Exception:
            return "", 0.0

        words = []
        confs = []
        for txt, conf in zip(data.get("text", []), data.get("conf", [])):
            txt_norm = self._simple_normalize(txt)
            if not txt_norm:
                continue
            try:
                conf_val = float(conf)
            except Exception:
                conf_val = -1.0
            if conf_val >= 0:
                words.append(txt_norm)
                confs.append(conf_val)

        raw = "".join(words)
        formatted = self._simple_format(raw)
        formatted = self._postprocess_plate_text(formatted)
        if not formatted:
            return "", 0.0
        mean_conf = float(np.mean(confs)) / 100.0 if confs else 0.0
        return formatted, mean_conf

    def _paddle_extract_candidates(self, node, out):
        if isinstance(node, (list, tuple)):
            if len(node) == 2 and isinstance(node[0], str):
                try:
                    out.append((node[0], float(node[1])))
                except Exception:
                    out.append((node[0], 0.0))
                return
            for item in node:
                self._paddle_extract_candidates(item, out)

    def _paddle_ocr_on_image(self, image):
        if self.paddle_ocr is None:
            return "", 0.0

        try:
            result = self.paddle_ocr.ocr(image, det=False, cls=False)
        except Exception:
            return "", 0.0

        candidates = []
        self._paddle_extract_candidates(result, candidates)
        if not candidates:
            return "", 0.0

        best_text = ""
        best_conf = 0.0
        best_score = -1e9
        for raw_text, conf in candidates:
            clean = self._simple_normalize(raw_text)
            if not clean:
                continue
            score = float(conf)
            if re.fullmatch(r"[A-Z]{3}[0-9]{3,4}", clean):
                score += 0.35
            if len(clean) in (6, 7):
                score += 0.10
            if any(ch.isdigit() for ch in clean) and any(ch.isalpha() for ch in clean):
                score += 0.05

            if score > best_score:
                best_score = score
                best_text = self._postprocess_plate_text(self._simple_format(clean))
                best_conf = float(conf)

        return best_text, best_conf

    def _rapidocr_on_image(self, image):
        if self.rapid_ocr is None:
            return "", 0.0

        try:
            result, _ = self.rapid_ocr(image)
        except Exception:
            return "", 0.0

        if not result:
            return "", 0.0

        min_conf = float(CONFIG.get("ocr_rapid_min_conf", 0.20))
        tokens = []
        for item in result:
            if not isinstance(item, (list, tuple)) or len(item) < 3:
                continue

            raw_text = str(item[1])
            try:
                conf = float(item[2])
            except Exception:
                conf = 0.0
            if conf < min_conf:
                continue

            clean = self._simple_normalize(raw_text)
            if not clean:
                continue

            x_center = 0.0
            try:
                box = np.array(item[0], dtype=np.float32)
                if box.size >= 2:
                    x_center = float(np.mean(box[:, 0]))
            except Exception:
                x_center = 0.0

            tokens.append((clean, conf, x_center))

        if not tokens:
            return "", 0.0

        tokens.sort(key=lambda t: t[2])

        # Candidatos: cada token individual y concatenaciones en orden izquierda->derecha.
        candidates = []
        for clean, conf, _ in tokens:
            candidates.append((clean, conf))

        concat_clean = "".join(t[0] for t in tokens)
        concat_conf = float(np.mean([t[1] for t in tokens])) if tokens else 0.0
        if concat_clean:
            candidates.append((concat_clean, concat_conf))

        if len(tokens) >= 2:
            for i in range(len(tokens) - 1):
                pair_clean = tokens[i][0] + tokens[i + 1][0]
                pair_conf = float((tokens[i][1] + tokens[i + 1][1]) / 2.0)
                candidates.append((pair_clean, pair_conf))

        best_text = ""
        best_conf = 0.0
        best_score = -1e9
        for clean, conf in candidates:

            score = conf
            if re.fullmatch(r"[A-Z]{3}[0-9]{3,4}", clean):
                score += 0.70
            if len(clean) in (6, 7):
                score += 0.18
            if any(ch.isdigit() for ch in clean) and any(ch.isalpha() for ch in clean):
                score += 0.14

            if score > best_score:
                best_score = score
                best_text = self._postprocess_plate_text(self._simple_format(clean))
                best_conf = conf

        return best_text, best_conf

    def _simple_segment_and_recognize_with_conf(self, plate_image):
        best_text = ""
        best_conf = 0.0
        best_score = -1.0

        for variant in self._simple_preprocess_variants(plate_image):
            text, conf = self._simple_ocr_on_image(variant)
            clean = self._simple_normalize(text)
            if not clean:
                continue

            score = float(conf)
            if re.fullmatch(r"[A-Z]{3}[0-9]{3,4}", clean):
                score += 0.35
            if len(clean) in (6, 7):
                score += 0.10
            if any(ch.isdigit() for ch in clean) and any(ch.isalpha() for ch in clean):
                score += 0.05

            if score > best_score:
                best_score = score
                best_text = self._postprocess_plate_text(text)
                best_conf = conf

        return best_text, best_conf

    def _ctc_decode_tflite(self, probs):
        best_ids = np.argmax(probs, axis=-1).tolist()
        best_probs = np.max(probs, axis=-1).tolist()

        chars = []
        confs = []
        prev = None
        blank = getattr(self, "blank_idx", len(self.char_set))
        for idx, conf in zip(best_ids, best_probs):
            if idx == blank or idx == prev:
                prev = idx
                continue
            if 0 <= idx < len(self.char_set):
                chars.append(self.char_set[idx])
                confs.append(float(conf))
            prev = idx

        text = "".join(chars)
        text = "".join(ch for ch in text if ch.isalnum())
        mean_conf = float(np.mean(confs)) if confs else 0.0
        return text, mean_conf

    def _score_ctc_candidate(self, text: str, conf: float) -> float:
        if not text:
            return -1.0

        score = float(conf)
        n = len(text)

        # Longitudes típicas de placa local (6-8) y preferencia fuerte por 7.
        if 6 <= n <= 8:
            score += float(CONFIG.get("ctc_len_bonus", 0.12))
        if n == 7:
            score += float(CONFIG.get("ctc_len7_bonus", 0.08))

        # Priorización de patrón tipo LLLNNNN cuando aparezca.
        if re.fullmatch(r"[A-Z]{3}[0-9]{4}", text):
            score += float(CONFIG.get("ctc_pattern_bonus", 0.25))

        # Penaliza secuencias no plausibles.
        if not any(ch.isdigit() for ch in text):
            score -= float(CONFIG.get("ctc_no_digit_penalty", 0.15))
        if not any(ch.isalpha() for ch in text):
            score -= float(CONFIG.get("ctc_no_letter_penalty", 0.15))

        return score

    def _ctc_input_variants(self, plate_image):
        base = plate_image
        vars_out = [base]

        if len(base.shape) == 3:
            gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
        else:
            gray = base

        # Variante CLAHE para contrastes difíciles.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        vars_out.append(cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR))

        # Variante sharpen suave para placas pequeñas o borrosas.
        blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
        sharp = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)
        vars_out.append(cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR))

        return vars_out

    def _is_small_plate(self, plate_image):
        """Detecta si una placa es pequeña basándose en su altura."""
        h, w = plate_image.shape[:2]
        threshold = float(CONFIG.get("ocr_small_plate_threshold", 100))
        return h < threshold

    def _upscale_for_ocr(self, plate_image):
        """Upscale agresivo para placas MUY pequeñas."""
        h, w = plate_image.shape[:2]
        upscale_threshold = float(CONFIG.get("ocr_small_plate_upscale_threshold", 50))
        if h >= upscale_threshold:
            return plate_image
        
        upscale_factor = float(CONFIG.get("ocr_small_plate_upscale", 2.0))
        new_h = max(int(upscale_threshold), int(h * upscale_factor))
        new_w = int(w * new_h / h)
        return cv2.resize(plate_image, (int(new_w), int(new_h)), interpolation=cv2.INTER_CUBIC)

    def _ocr_plate_variants(self, plate_image):
        """Genera variantes de OCR según perfil y tamaño de placa."""
        profile = str(CONFIG.get("ocr_profile", "balanced")).lower()
        h = plate_image.shape[0]

        use_far_profile = profile == "far" and h < int(CONFIG.get("ocr_far_plate_threshold", 140))
        use_small_profile = profile != "far" and h < int(CONFIG.get("ocr_small_plate_threshold", 100))

        if use_far_profile:
            upscale_threshold = float(CONFIG.get("ocr_far_plate_upscale_threshold", 80))
            upscale_factor = float(CONFIG.get("ocr_far_plate_upscale", 2.5))
            clip_limit = float(CONFIG.get("ocr_far_plate_clahe_clip", 5.0))
            enable_denoise = bool(CONFIG.get("ocr_far_plate_denoise", True))
            enable_morph = bool(CONFIG.get("ocr_far_plate_morph", True))
        elif use_small_profile:
            upscale_threshold = float(CONFIG.get("ocr_small_plate_upscale_threshold", 50))
            upscale_factor = float(CONFIG.get("ocr_small_plate_upscale", 2.0))
            clip_limit = float(CONFIG.get("ocr_small_plate_clahe_clip", 4.0))
            enable_denoise = bool(CONFIG.get("ocr_small_plate_denoise", True))
            enable_morph = bool(CONFIG.get("ocr_small_plate_morph", True))
        else:
            return self._ctc_input_variants(plate_image)

        if h < upscale_threshold:
            target_h = max(int(upscale_threshold), int(h * upscale_factor))
            target_w = max(1, int(plate_image.shape[1] * target_h / max(1, h)))
            base = cv2.resize(plate_image, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        else:
            base = plate_image

        variants = [base]
        if len(base.shape) == 3:
            gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
        else:
            gray = base

        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(4, 4)).apply(gray)
        variants.append(cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR))

        if enable_denoise:
            try:
                denoised = cv2.fastNlMeansDenoising(
                    clahe,
                    h=10,
                    templateWindowSize=7,
                    searchWindowSize=21,
                )
                variants.append(cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR))
            except Exception:
                pass

        if enable_morph:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            morph = cv2.morphologyEx(clahe, cv2.MORPH_OPEN, kernel, iterations=1)
            morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel, iterations=1)
            variants.append(cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR))

        blur_strong = cv2.GaussianBlur(clahe, (0, 0), 0.8)
        sharp_strong = cv2.addWeighted(clahe, 2.2, blur_strong, -1.2, 0)
        sharp_strong = np.clip(sharp_strong, 0, 255).astype(np.uint8)
        variants.append(cv2.cvtColor(sharp_strong, cv2.COLOR_GRAY2BGR))

        p_low, p_high = 2, 98
        vmin, vmax = np.percentile(gray, [p_low, p_high])
        if vmax > vmin:
            contrast_stretched = np.clip(
                (gray.astype(np.float32) - vmin) * 255.0 / (vmax - vmin),
                0,
                255,
            ).astype(np.uint8)
            clahe_stretched = cv2.createCLAHE(
                clipLimit=clip_limit,
                tileGridSize=(4, 4),
            ).apply(contrast_stretched)
            variants.append(cv2.cvtColor(clahe_stretched, cv2.COLOR_GRAY2BGR))

        try:
            adaptive = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=11,
                C=2,
            )
            variants.append(cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR))
        except Exception:
            pass

        return variants

    def _ctc_input_variants_aggressive(self, plate_image):
        """Variantes agresivas de preprocesamiento para placas pequeñas."""
        # Upscale agresivo si es muy pequeño
        plate_upscaled = self._upscale_for_ocr(plate_image)
        
        base = plate_upscaled
        vars_out = [base]

        if len(base.shape) == 3:
            gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
        else:
            gray = base

        # ===== Variante 1: CLAHE MÁS AGRESIVO =====
        clip_limit = float(CONFIG.get("ocr_small_plate_clahe_clip", 4.0))
        clahe_aggressive = cv2.createCLAHE(
            clipLimit=clip_limit, 
            tileGridSize=(4, 4)  # grid más fino para mejor adaptación
        ).apply(gray)
        vars_out.append(cv2.cvtColor(clahe_aggressive, cv2.COLOR_GRAY2BGR))

        # ===== Variante 2: CLAHE + DENOISING =====
        enable_denoise = bool(CONFIG.get("ocr_small_plate_denoise", True))
        if enable_denoise:
            try:
                # fastNlMeansDenoising para placas ruidosas
                denoised = cv2.fastNlMeansDenoising(
                    clahe_aggressive,
                    h=10,  # filter strength
                    templateWindowSize=7,
                    searchWindowSize=21
                )
                vars_out.append(cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR))
            except:
                pass

        # ===== Variante 3: CLAHE + MORPHOLOGICAL OPERATIONS =====
        enable_morph = bool(CONFIG.get("ocr_small_plate_morph", True))
        if enable_morph:
            # Opening seguido de closing para limpiar ruido
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
            morph = cv2.morphologyEx(clahe_aggressive, cv2.MORPH_OPEN, kernel, iterations=1)
            morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel, iterations=1)
            vars_out.append(cv2.cvtColor(morph, cv2.COLOR_GRAY2BGR))

        # ===== Variante 4: CLAHE + SHARPEN FUERTE =====
        blur_strong = cv2.GaussianBlur(clahe_aggressive, (0, 0), 0.8)
        sharp_strong = cv2.addWeighted(clahe_aggressive, 2.2, blur_strong, -1.2, 0)
        sharp_strong = np.clip(sharp_strong, 0, 255).astype(np.uint8)
        vars_out.append(cv2.cvtColor(sharp_strong, cv2.COLOR_GRAY2BGR))

        # ===== Variante 5: CONTRAST STRETCHING =====
        p_low, p_high = 2, 98
        vmin, vmax = np.percentile(gray, [p_low, p_high])
        if vmax > vmin:
            contrast_stretched = np.clip(
                (gray.astype(np.float32) - vmin) * 255.0 / (vmax - vmin),
                0, 255
            ).astype(np.uint8)
            # Luego aplicar CLAHE al stretched
            clahe_stretched = cv2.createCLAHE(
                clipLimit=clip_limit,
                tileGridSize=(4, 4)
            ).apply(contrast_stretched)
            vars_out.append(cv2.cvtColor(clahe_stretched, cv2.COLOR_GRAY2BGR))

        # ===== Variante 6: ADAPTIVE THRESHOLDING para high-contrast =====
        try:
            adaptive = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=11,
                C=2
            )
            vars_out.append(cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR))
        except:
            pass

        return vars_out
    
    def recognize_character(self, char_image):
        """Clasifica un carácter"""
        
        # Preprocesar
        char_resized = cv2.resize(char_image, (self.input_size, self.input_size))
        if len(char_resized.shape) == 2:
            char_resized = cv2.cvtColor(char_resized, cv2.COLOR_GRAY2BGR)
        
        char_normalized = char_resized.astype(np.float32) / 255.0
        char_input = np.expand_dims(char_normalized, 0)

        if self.backend != "tflite":
            raise RuntimeError("recognize_character solo está soportado en backend tflite")

        # Inferencia
        self.interpreter.set_tensor(
            self.input_details[0]['index'],
            char_input
        )
        self.interpreter.invoke()
        
        logits = self.interpreter.get_tensor(
            self.output_details[0]['index']
        )[0]
        
        char_idx = np.argmax(logits)
        confidence = float(logits[char_idx])
        char = self.char_set[char_idx] if char_idx < len(self.char_set) else '?'
        
        return char, confidence

    def _recognize_character_topk(self, char_image, top_k=None):
        """Devuelve las mejores opciones del clasificador de caracteres."""
        if top_k is None:
            top_k = int(CONFIG.get("ocr_char_topk", 3))

        char_resized = cv2.resize(char_image, (self.input_size, self.input_size))
        if len(char_resized.shape) == 2:
            char_resized = cv2.cvtColor(char_resized, cv2.COLOR_GRAY2BGR)

        char_normalized = char_resized.astype(np.float32) / 255.0
        char_input = np.expand_dims(char_normalized, 0)

        if self.backend != "tflite":
            raise RuntimeError("recognize_character solo está soportado en backend tflite")

        self.interpreter.set_tensor(self.input_details[0]['index'], char_input)
        self.interpreter.invoke()
        logits = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        logits = np.asarray(logits, dtype=np.float32)
        if logits.size == 0:
            return [("?", 0.0)]

        logits = logits - float(np.max(logits))
        probs = np.exp(logits)
        denom = float(np.sum(probs))
        if denom <= 0:
            return [("?", 0.0)]
        probs = probs / denom

        top_idx = np.argsort(probs)[::-1][:max(1, int(top_k))]
        candidates = []
        for idx in top_idx:
            char = self.char_set[int(idx)] if int(idx) < len(self.char_set) else '?'
            candidates.append((char, float(probs[int(idx)])))
        return candidates

    def recognize_character_fallback(self, char_image):
        if self.char_fallback_interpreter is None:
            return '?', 0.0

        char_resized = cv2.resize(char_image, (self.input_size, self.input_size))
        if len(char_resized.shape) == 2:
            char_resized = cv2.cvtColor(char_resized, cv2.COLOR_GRAY2BGR)

        char_normalized = char_resized.astype(np.float32) / 255.0
        char_input = np.expand_dims(char_normalized, 0)

        self.char_fallback_interpreter.set_tensor(
            self.char_fallback_input_details[0]['index'],
            char_input,
        )
        self.char_fallback_interpreter.invoke()

        logits = self.char_fallback_interpreter.get_tensor(
            self.char_fallback_output_details[0]['index']
        )[0]

        char_idx = int(np.argmax(logits))
        confidence = float(logits[char_idx])
        char = self.char_set[char_idx] if char_idx < len(self.char_set) else '?'
        return char, confidence

    def _segment_char_fallback(self, plate_image):
        if self.char_fallback_interpreter is None:
            return "", 0.0

        h, w = plate_image.shape[:2]
        if h < 8 or w < 24:
            return "", 0.0

        bands = [
            plate_image,
            plate_image[int(h * 0.15):int(h * 0.9), :],
            plate_image[int(h * 0.30):h, :],
        ]

        best_text = ""
        best_conf = 0.0
        best_score = -1.0

        for band in bands:
            if band is None or band.size == 0:
                continue

            for num_chars in (6, 7, 8):
                crops = self._split_equal_chars(band, num_chars)
                if not crops:
                    continue

                chars = []
                confs = []
                for cimg in crops:
                    ch, conf = self.recognize_character_fallback(cimg)
                    conf = float(conf)
                    confs.append(conf)
                    chars.append(ch if conf >= 0.35 else '?')

                raw_text = "".join(chars)
                clean_text = "".join(ch for ch in raw_text if ch.isalnum())
                coverage = sum(1 for ch in chars if ch != '?') / float(max(1, len(chars)))
                mean_conf = float(np.mean(confs)) if confs else 0.0
                score = mean_conf * (0.65 + 0.35 * coverage)

                candidate = clean_text if len(clean_text) >= 4 else raw_text.replace('?', '')
                if score > best_score and candidate:
                    best_score = score
                    best_text = candidate
                    best_conf = mean_conf

        return best_text, best_conf

    def _format_plate_text(self, text):
        clean = "".join(ch for ch in str(text).upper() if ch.isalnum())
        if re.fullmatch(r"[A-Z]{3}[0-9]{3}", clean):
            return f"{clean[:3]}-{clean[3:]}"
        if re.fullmatch(r"[A-Z]{3}[0-9]{4}", clean):
            return f"{clean[:3]}-{clean[3:]}"
        return clean

    def _expected_digit_start(self, total_segments, profile="balanced"):
        total_segments = int(max(1, total_segments))
        if total_segments >= 6:
            return 3
        if profile == "far" and total_segments >= 5:
            return 3
        return max(1, total_segments - 3)

    def _positionally_correct_text(self, text, conf=0.0, total_segments=None, profile="balanced"):
        if not bool(CONFIG.get("ocr_pos_correction_enabled", True)):
            return "".join(ch for ch in str(text).upper() if ch.isalnum())

        clean = "".join(ch for ch in str(text).upper() if ch.isalnum())
        if not clean:
            return ""

        total = int(total_segments) if total_segments is not None else len(clean)
        digit_start = min(len(clean), self._expected_digit_start(total, profile=profile))
        letter_block = clean[:digit_start]
        digit_block = clean[digit_start:]

        # Si no respeta separación letra/dígito, conserva OCR crudo para soportar formatos no estándar.
        if not letter_block or not digit_block or (not letter_block.isalpha()) or (not digit_block.isdigit()):
            return clean

        max_changes = int(CONFIG.get("ocr_pos_max_changes", 3))
        low_conf = float(conf) < float(CONFIG.get("ocr_pos_low_conf_threshold", 0.30))

        corrected = []
        changes = 0
        for idx, ch in enumerate(clean):
            out_ch = ch
            if idx < digit_start:
                if ch in DIGIT_TO_LETTER_VISUAL and changes < max_changes:
                    out_ch = DIGIT_TO_LETTER_VISUAL[ch]
                    changes += 1
                elif low_conf and idx == 1 and ch == "P" and changes < max_changes:
                    # Confusión frecuente en la letra central: P/D por borde derecho tenue.
                    out_ch = "D"
                    changes += 1
                elif (
                    low_conf
                    and bool(CONFIG.get("ocr_pos_last_letter_xc_swap", True))
                    and idx == (digit_start - 1)
                    and ch in ("X", "C")
                    and changes < max_changes
                ):
                    # En la última letra del bloque, X/C se confunden con trazos finos o reflejos.
                    out_ch = "C" if ch == "X" else "X"
                    changes += 1
            else:
                if ch in LETTER_TO_DIGIT_VISUAL and changes < max_changes:
                    out_ch = LETTER_TO_DIGIT_VISUAL[ch]
                    changes += 1
                elif low_conf and idx == (digit_start + 1) and ch == "8" and changes < max_changes:
                    # En baja confianza, 8 y 9 se confunden con frecuencia en la posición central numérica.
                    out_ch = "9"
                    changes += 1
                elif (
                    low_conf
                    and bool(CONFIG.get("ocr_pos_last_digit_24_swap", True))
                    and idx == (len(clean) - 1)
                    and ch in ("2", "4")
                    and changes < max_changes
                ):
                    # En el último dígito, 2/4 puede variar por perspectiva o recorte lateral.
                    out_ch = "4" if ch == "2" else "2"
                    changes += 1
            corrected.append(out_ch)

        return self._format_plate_text("".join(corrected))

    def _apply_positional_confusion_to_segments(self, segment_candidates, total_segments, profile="balanced"):
        if not bool(CONFIG.get("ocr_pos_correction_enabled", True)):
            return segment_candidates

        digit_start = self._expected_digit_start(total_segments, profile=profile)
        mix = float(CONFIG.get("ocr_pos_confusion_mix", 0.45))
        top_keep = int(CONFIG.get("ocr_char_topk", 3)) + 2

        corrected_segments = []
        for idx, candidates in enumerate(segment_candidates):
            if not candidates:
                corrected_segments.append(candidates)
                continue

            base = {}
            for ch, prob in candidates:
                if not ch or ch == "?":
                    continue
                base[ch] = base.get(ch, 0.0) + max(0.0, float(prob))

            adjusted = dict(base)
            if idx < digit_start:
                for src, dst in DIGIT_TO_LETTER_VISUAL.items():
                    p = base.get(src, 0.0)
                    if p <= 0.0:
                        continue
                    moved = p * mix
                    adjusted[src] = max(0.0, adjusted.get(src, 0.0) - moved)
                    adjusted[dst] = adjusted.get(dst, 0.0) + moved
            else:
                for src, dst in LETTER_TO_DIGIT_VISUAL.items():
                    p = base.get(src, 0.0)
                    if p <= 0.0:
                        continue
                    moved = p * mix
                    adjusted[src] = max(0.0, adjusted.get(src, 0.0) - moved)
                    adjusted[dst] = adjusted.get(dst, 0.0) + moved

            norm = sum(v for v in adjusted.values() if v > 0.0)
            if norm <= 0.0:
                corrected_segments.append(candidates)
                continue

            ranked = sorted(
                ((ch, v / norm) for ch, v in adjusted.items() if v > 0.0),
                key=lambda item: item[1],
                reverse=True,
            )
            corrected_segments.append(ranked[:top_keep])

        return corrected_segments

    def _score_plate_candidate(self, candidate, score, total_segments, profile="balanced"):
        clean = "".join(ch for ch in str(candidate).upper() if ch.isalnum())
        if not clean:
            return -1e9, ""

        bonus = float(score)
        plate_bonus = float(CONFIG.get("ocr_plate_pattern_bonus", 0.22))
        hyphen_bonus = float(CONFIG.get("ocr_plate_hyphen_bonus", 0.08))

        if re.fullmatch(r"[A-Z]{3}[0-9]{3}", clean):
            bonus += plate_bonus
            return bonus, f"{clean[:3]}-{clean[3:]}"

        if profile == "far" and total_segments >= 6 and re.fullmatch(r"[A-Z]{3}[0-9]{4}", clean):
            bonus += plate_bonus * 0.6

        if len(clean) == 6 and clean[:3].isalpha() and clean[3:].isdigit():
            formatted = f"{clean[:3]}-{clean[3:]}"
            bonus += hyphen_bonus
            return bonus, formatted

        return bonus, clean

    def _decode_plate_with_topk(self, segment_candidates, total_segments, profile="balanced"):
        """Beam search corto para elegir la cadena de placa más plausible."""
        segment_candidates = self._apply_positional_confusion_to_segments(
            segment_candidates,
            total_segments=total_segments,
            profile=profile,
        )
        beam_width = int(CONFIG.get("ocr_plate_beam_width", 24))
        beam = [("", 0.0)]

        for pos, candidates in enumerate(segment_candidates):
            next_beam = []
            for prefix, log_score in beam:
                for ch, prob in candidates:
                    if not ch or ch == "?":
                        continue
                    p = max(float(prob), 1e-6)
                    next_beam.append((prefix + ch, log_score + float(np.log(p))))

            if not next_beam:
                return "", 0.0

            next_beam.sort(key=lambda item: item[1], reverse=True)
            beam = next_beam[:beam_width]

        best_text = ""
        best_score = -1e9
        best_conf = 0.0

        for cand, log_score in beam:
            avg_log = float(log_score) / float(max(1, len(cand)))
            corrected = self._positionally_correct_text(
                cand,
                conf=float(np.exp(avg_log)) if avg_log > -50 else 0.0,
                total_segments=total_segments,
                profile=profile,
            )
            score, formatted = self._score_plate_candidate(corrected, avg_log, total_segments, profile=profile)
            if score > best_score:
                best_score = score
                best_text = formatted
                best_conf = float(np.exp(avg_log)) if avg_log > -50 else 0.0

        return best_text, best_conf

    def _prepare_ctc_input(self, plate_image):
        if self.backend == "keras":
            in_shape = self.keras_model.input_shape
        else:
            in_shape = self.input_details[0].get('shape', [1, 48, 192, 1])
        in_h = int(in_shape[1]) if len(in_shape) > 2 and int(in_shape[1]) > 0 else 48
        in_w = int(in_shape[2]) if len(in_shape) > 2 and int(in_shape[2]) > 0 else 192
        in_c = int(in_shape[3]) if len(in_shape) > 3 and int(in_shape[3]) > 0 else 1

        img = cv2.resize(plate_image, (in_w, in_h), interpolation=cv2.INTER_AREA)

        if in_c == 1:
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = cv2.equalizeHist(img)
            img = np.expand_dims(img, axis=-1)
        else:
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

        x = img.astype(np.float32) / 255.0
        return np.expand_dims(x, axis=0)

    def recognize_sequence_ctc(self, plate_image):
        """Reconoce texto completo usando CTC + TTA ligera de preprocesado."""
        variants = self._ocr_plate_variants(plate_image)
        
        best_text = ""
        best_conf = 0.0
        best_score = -1.0

        for variant in variants:
            x = self._prepare_ctc_input(variant)

            if self.backend == "keras":
                probs = self.keras_model.predict(x, verbose=0)
            else:
                self.interpreter.set_tensor(self.input_details[0]['index'], x)
                self.interpreter.invoke()
                probs = self.interpreter.get_tensor(self.output_details[0]['index'])

            if probs is None or probs.size == 0:
                continue

            probs = probs[0]  # [time, classes]
            if probs.ndim != 2:
                continue

            text, conf = self._ctc_decode_tflite(probs)
            text = self._positionally_correct_text(
                text,
                conf=conf,
                total_segments=len("".join(ch for ch in str(text).upper() if ch.isalnum())),
                profile=str(CONFIG.get("ocr_profile", "balanced")).lower(),
            )
            score = self._score_ctc_candidate(text, conf)
            if score > best_score:
                best_score = score
                best_text = text
                best_conf = conf

        # Siempre compara contra fallback char cuando esté disponible;
        # evita quedarse con una salida CTC plausible en confianza pero mala en formato.
        fb_text, fb_conf = self._segment_char_fallback(plate_image)
        fb_text = self._positionally_correct_text(
            fb_text,
            conf=fb_conf,
            total_segments=len("".join(ch for ch in str(fb_text).upper() if ch.isalnum())),
            profile=str(CONFIG.get("ocr_profile", "balanced")).lower(),
        )
        fb_score = self._score_ctc_candidate(fb_text, fb_conf)
        margin = float(CONFIG.get("ctc_vs_char_margin", 0.02))
        if fb_score > (best_score + margin):
            return fb_text, fb_conf

        return best_text, best_conf

    def _split_equal_chars(self, plate_image, num_chars):
        h, w = plate_image.shape[:2]
        if h < 8 or w < num_chars * 4:
            return []

        char_width = w / float(num_chars)
        crops = []
        for i in range(num_chars):
            x_start = int(round(i * char_width))
            x_end = int(round((i + 1) * char_width)) if i < num_chars - 1 else w
            x_start = max(0, min(x_start, w - 1))
            x_end = max(x_start + 1, min(x_end, w))
            char_img = plate_image[:, x_start:x_end]
            if char_img.size == 0 or char_img.shape[1] < 5:
                return []
            crops.append(char_img)
        return crops

    def segment_and_recognize_with_conf(self, plate_image):
        """Decodifica OCR probando multiples longitudes y bandas de texto."""
        if self.ocr_mode == "rapid":
            best_text = ""
            best_conf = 0.0
            best_score = -1.0
            variants = self._simple_preprocess_variants(plate_image)
            max_variants = max(1, int(CONFIG.get("ocr_rapid_max_variants", 2)))
            for variant in variants[:max_variants]:
                text, conf = self._rapidocr_on_image(variant)
                clean = self._simple_normalize(text)
                if not clean:
                    continue

                score = float(conf)
                if re.fullmatch(r"[A-Z]{3}[0-9]{3,4}", clean):
                    score += 0.35
                if len(clean) in (6, 7):
                    score += 0.10
                if any(ch.isdigit() for ch in clean) and any(ch.isalpha() for ch in clean):
                    score += 0.05

                if score > best_score:
                    best_score = score
                    best_text = text
                    best_conf = conf
            return best_text, best_conf

        if self.ocr_mode == "paddle":
            best_text = ""
            best_conf = 0.0
            best_score = -1.0
            for variant in self._simple_preprocess_variants(plate_image):
                text, conf = self._paddle_ocr_on_image(variant)
                clean = self._simple_normalize(text)
                if not clean:
                    continue

                score = float(conf)
                if re.fullmatch(r"[A-Z]{3}[0-9]{3,4}", clean):
                    score += 0.35
                if len(clean) in (6, 7):
                    score += 0.10
                if any(ch.isdigit() for ch in clean) and any(ch.isalpha() for ch in clean):
                    score += 0.05

                if score > best_score:
                    best_score = score
                    best_text = text
                    best_conf = conf
            return best_text, best_conf

        if self.ocr_mode == "simple":
            return self._simple_segment_and_recognize_with_conf(plate_image)
        if self.ocr_mode == "ctc":
            return self.recognize_sequence_ctc(plate_image)

        h, w = plate_image.shape[:2]
        if h < 8 or w < 24:
            return "", 0.0

        best_text = ""
        best_conf = 0.0
        best_score = -1.0

        profile = str(CONFIG.get("ocr_profile", "balanced")).lower()
        if profile == "far":
            num_chars_range = range(
                int(CONFIG.get("ocr_far_num_chars_min", 5)),
                int(CONFIG.get("ocr_far_num_chars_max", 9)) + 1,
            )
            char_conf_threshold = float(CONFIG.get("ocr_far_char_conf_threshold", 0.30))
        else:
            num_chars_range = range(6, 9)
            char_conf_threshold = 0.35

        plate_variants = self._ocr_plate_variants(plate_image)
        for variant in plate_variants:
            if variant is None or variant.size == 0:
                continue

            vh, vw = variant.shape[:2]
            bands = [
                variant,
                variant[int(vh * 0.15):int(vh * 0.9), :],
                variant[int(vh * 0.30):vh, :],
            ]

            if profile == "far":
                bands = [
                    variant,
                    variant[int(vh * 0.10):int(vh * 0.95), :],
                    variant[int(vh * 0.25):vh, :],
                    variant[int(vh * 0.05):int(vh * 0.85), :],
                ]

            for band in bands:
                if band is None or band.size == 0:
                    continue

                for num_chars in num_chars_range:
                    crops = self._split_equal_chars(band, num_chars)
                    if not crops:
                        continue

                    segment_candidates = []
                    confs = []
                    for cimg in crops:
                        top_candidates = self._recognize_character_topk(cimg, top_k=int(CONFIG.get("ocr_char_topk", 3)))
                        segment_candidates.append(top_candidates)
                        if top_candidates:
                            confs.append(float(top_candidates[0][1]))

                    if not segment_candidates:
                        continue

                    candidate_text, candidate_conf = self._decode_plate_with_topk(
                        segment_candidates,
                        total_segments=num_chars,
                        profile=profile,
                    )

                    if not candidate_text:
                        continue

                    coverage = sum(1 for top in segment_candidates if top and top[0][1] >= char_conf_threshold)
                    coverage = coverage / float(max(1, len(segment_candidates)))
                    mean_conf = float(np.mean(confs)) if confs else float(candidate_conf)
                    candidate_text = self._positionally_correct_text(
                        candidate_text,
                        conf=mean_conf,
                        total_segments=num_chars,
                        profile=profile,
                    )
                    score = mean_conf * (0.65 + 0.35 * coverage)
                    if re.fullmatch(r"[A-Z]{3}-?[0-9]{3}", candidate_text):
                        score += float(CONFIG.get("ocr_plate_pattern_bonus", 0.22))

                    if score > best_score:
                        best_score = score
                        best_text = candidate_text
                        best_conf = mean_conf

        return best_text, best_conf
    
    def segment_and_recognize(self, plate_image, num_chars=8):
        """
        Segmenta y reconoce placa completa
        
        Args:
            plate_image: ROI de placa
            num_chars: número de caracteres esperados
        
        Returns:
            plate_string: string de placa reconocida
        """
        plate_text, _ = self.segment_and_recognize_with_conf(plate_image)
        return plate_text


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
                    "⚠️ TFLite requiere Flex ops no disponibles. "
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
                        logger.info("✓ Detector fallback listo: %s", fallback_path)
                    except Exception as e:
                        logger.warning("⚠️ No se pudo cargar detector fallback (%s): %s", fallback_path, e)
        ocr_path = Path(config["ocr_model"])
        if ocr_path.exists():
            self.ocr = OCRInference(config["ocr_model"])
            self.ocr_enabled = True
        else:
            self.ocr = None
            self.ocr_enabled = False
            logger.warning(
                "⚠️ OCR no disponible: %s. Continuando solo con detector.",
                config["ocr_model"]
            )
        self.fps_buffer = deque(maxlen=config["fps_buffer_size"])
        self._last_selected_ocr = None
        
        logger.info("✓ Pipeline inicializado")

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
        """Ajusta el recorte OCR según el tamaño aparente de la placa."""
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
        t = t.replace('—', '-').replace('_', '-')
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
        """Mejora contraste local antes de detectar placas en escenas difíciles."""
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
        """Segundo pase OCR para forzar hipótesis LLL-NNNN en baja confianza."""
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

    def _build_detector_candidates(self, frame):
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

        # Rescue de placas pequeñas: solo cuando falla lo normal, para no afectar lo ya estable.
        only_on_miss = bool(self.config.get("small_plate_rescue_only_on_miss", True))
        if (not candidates and only_on_miss) or (not only_on_miss):
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

        # Mantiene comportamiento clásico si no hay OCR o está desactivado el selector OCR-aware.
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
                    plate_text = self.ocr.segment_and_recognize(plate_roi)
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
        """Dibuja detección"""
        
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
        """Abre cámara con backends adecuados por plataforma y fallback genérico."""
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
            # Solicita GREY 8-bit para evitar conversiones erróneas de Y16.
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
        """Normaliza frames de cámara (mono/16-bit) y los entrega en BGR uint8."""
        if frame is None or frame.size == 0:
            return frame

        out = frame
        src_dtype = getattr(out, "dtype", None)
        src_shape = getattr(out, "shape", None)

        if out.dtype != np.uint8:
            # La cámara CSI monocroma suele entregar Y16/Y10. En escenas con
            # rango dinámico estrecho, un cast directo puede dejar todo negro.
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
        """Arranca un lector en segundo plano para no bloquear la cámara con la inferencia."""
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
        """Worker de OCR en segundo plano para no congelar el stream de cámara."""
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
                    logger.warning("⚠️ OCR worker error: %s", e)

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
        """Encola OCR para una firma única y garantiza al menos un intento por placa."""
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
        """Recupera el último resultado OCR del worker si existe."""
        with worker_state["lock"]:
            result = worker_state["result"]
            worker_state["result"] = None
        return result

    def _start_debug_snapshot_worker(self):
        """Worker dedicado para guardar snapshots OCR sin bloquear el loop de cámara."""
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
        """Encola snapshot OCR de depuración, conservando el más reciente."""
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
        """Reduce el frame para detección rápida manteniendo aspecto."""
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
        """Convierte la cámara a gris manteniendo un frame BGR para el resto del pipeline."""
        if frame is None or frame.size == 0:
            return frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def _scale_detection_candidate(self, candidate, src_shape, dst_shape):
        """Escala una detección desde src_shape a dst_shape."""
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

    def _select_fast_detector_candidate(self, frame):
        """Elige la mejor caja solo por el detector, sin OCR, para cámara en vivo."""
        candidates = self._build_detector_candidates(frame)
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
        """Corre OCR solo sobre una detección ya elegida."""
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
        """OCR rápido sobre un único recorte de placa para modo cámara en tiempo real."""
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

        # Si sigue ganando un encabezado (p.ej. ECUADOR), reintenta en una banda más baja.
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

        # Última salvaguarda: no reportar encabezados ni texto solo alfabético largo.
        if bool(best.get("is_header")) or (not bool(best.get("has_digit")) and len(str(best.get("norm", ""))) >= 5):
            return None

        best.pop("norm", None)
        best.pop("has_digit", None)
        best.pop("is_header", None)

        return best

    def _select_camera_ocr_variant(self, roi):
        """Selecciona una sola variante de ROI para OCR en cámara (rápido y determinista)."""
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
        """Reescala el ROI para OCR cuando llega pequeño desde detector en cámara."""
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
        """Mejora contraste y nitidez del ROI de OCR en cámara."""
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
        """Guarda una lámina de depuración OCR: recorte, preprocesados y lectura."""
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
        """Guarda el recorte exacto que se envía al OCR."""
        if roi is None or roi.size == 0:
            return None

        output_dir.mkdir(parents=True, exist_ok=True)
        ts_label = time.strftime("%Y%m%d_%H%M%S", time.localtime(now_ts))
        conf = float(det.get("confidence", 0.0)) if isinstance(det, dict) else 0.0
        sig_tag = "_".join(str(x) for x in signature[:3]) if isinstance(signature, tuple) else "na"
        out_path = output_dir / f"ocr_input_{ts_label}_{source}_{conf:.2f}_{sig_tag}.jpg"
        cv2.imwrite(str(out_path), roi)
        return out_path
    
    def run_video_stream(self, video_source=0):
        """Ejecuta stream de video en modo ligero: detector rápido + OCR sólo al detectar placa."""
        
        logger.info(f"Abriendo cámara: {video_source}")
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
            logger.error("❌ No se pudo abrir cámara")
            return
        
        logger.info("✓ Cámara abierta. Presiona 'q' para salir.")
        logger.info("✓ Backend cámara: %s", backend)
        
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
        first_detect_time = None
        last_debug_snapshot_time = 0.0
        stable_candidate = None
        last_seen_frame_id = -1
        last_seen_frame_time = time.time()
        
        try:
            while True:
                with reader["lock"]:
                    frame = None if reader["frame"] is None else reader["frame"].copy()
                    frame_id = int(reader["frame_id"])

                if frame is None:
                    if time.time() - last_seen_frame_time > freeze_timeout:
                        logger.warning("⚠️ Stream sin frames nuevos; reconectando cámara...")
                        reader["running"] = False
                        cap.release()
                        time.sleep(reconnect_backoff)
                        cap, backend = open_and_configure(video_source)
                        if cap is None:
                            logger.error("❌ Reconexión fallida de cámara")
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
                    logger.warning("⚠️ Cámara congelada (frame estático); reconectando...")
                    reader["running"] = False
                    cap.release()
                    time.sleep(reconnect_backoff)
                    cap, backend = open_and_configure(video_source)
                    if cap is None:
                        logger.error("❌ Reconexión fallida de cámara")
                        break
                    reader = self._start_camera_reader(cap)
                    last_seen_frame_id = -1
                    last_seen_frame_time = time.time()
                    continue

                frame = self._prepare_camera_frame(frame)

                frame_count += 1

                detection_candidate = None
                if frame_count % detection_stride == 0 or last_live is None:
                    preview = self._resize_for_detection(frame, preview_width)
                    detection_candidate = self._select_fast_detector_candidate(preview)

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
                                    # En modo cámara se recomienda worker asíncrono para no congelar la UI.
                                    pass
                    elif stable_candidate is not None:
                        if (time.time() - float(stable_candidate.get("last_seen", 0.0))) >= settle_drop_timeout:
                            stable_candidate = None

                if debug_enabled and debug_worker is not None:
                    debug_result = self._pop_debug_snapshot_result(debug_worker)
                    if debug_result is not None:
                        out_path, snap_text, snap_conf, snap_err, _ = debug_result
                        if snap_err:
                            logger.warning("⚠️ Snapshot OCR con error: %s | %s", out_path, snap_err)
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
                            last_live = {
                                "det": det,
                                "boxes": tuple(det_box) + tuple(ocr_best.get("ocr_box", (0, 0, 0, 0))),
                                "source": source,
                                "text": text_value,
                                "ocr_conf": ocr_best["ocr_conf"],
                                "ocr_box": ocr_best["ocr_box"],
                            }
                            last_ocr_result = {
                                "signature": signature,
                                "text": text_value,
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
                        # En modo asíncrono la firma puede cambiar entre frames y ocultar OCR válido.
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

                # Mostrar (si HighGUI está disponible)
                if show_window:
                    try:
                        cv2.imshow("License Plate Detection", display_out)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                    except cv2.error as e:
                        show_window = False
                        logger.warning(
                            "⚠️ OpenCV HighGUI no disponible (%s). "
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
            logger.info(f"\n✓ Procesados {frame_count} frames en {elapsed:.1f}s")
            logger.info(f"✓ FPS promedio: {frame_count/elapsed:.1f}")

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
        logger.info("✓ Imagen anotada guardada en %s", output_path)

        if show_window:
            try:
                cv2.imshow("License Plate Detection", annotated)
                cv2.waitKey(0)
            except cv2.error as e:
                logger.warning(
                    "⚠️ OpenCV HighGUI no disponible (%s). Se guardó la imagen anotada.",
                    e,
                )
            finally:
                try:
                    cv2.destroyAllWindows()
                except cv2.error:
                    pass

        return annotated, detections

    def run_camera_healthcheck(self, video_source=0, seconds=15, show_window=True, save_dir=None):
        """Verifica captura y visualización sin detector ni OCR."""
        logger.info(f"Healthcheck de cámara: {video_source}")
        cap, backend = self._open_camera(video_source)
        if cap is None or not cap.isOpened():
            logger.error("❌ No se pudo abrir cámara en healthcheck")
            return

        logger.info("✓ Healthcheck abierto. Backend cámara: %s", backend)
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
                    logger.warning("⚠️ Healthcheck: no se pudo leer frame")
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
                        logger.warning("⚠️ HighGUI no disponible en healthcheck (%s)", e)
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
                "✓ Healthcheck terminado: frames=%d saved=%d elapsed=%.1fs fps=%.1f",
                frame_count,
                saved_count,
                elapsed,
                frame_count / elapsed,
            )


# ============================================================================
# MAIN
# ============================================================================

def apply_runtime_profile(config, profile_name):
    profile = str(profile_name or "current").lower()
    if profile in ("current", "default"):
        return

    if profile == "jetson-stable":
        config.update({
            "yolo_imgsz": 512,
            "camera_detection_stride": 3,
            "camera_preview_width": 640,
            "camera_display_width": 720,
            "camera_ocr_cooldown_sec": 1.1,
            "camera_ocr_max_pending": 1,
            "camera_ocr_settle_sec": 0.30,
            "enable_detector_fallback": False,
            "enable_small_plate_rescue": False,
            "ocr_second_pass_7char_enabled": False,
        })
        logger.info("Perfil aplicado: jetson-stable")
        return

    if profile == "jetson-quality":
        config.update({
            "yolo_imgsz": 640,
            "camera_detection_stride": 2,
            "camera_preview_width": 800,
            "camera_display_width": 800,
            "camera_ocr_cooldown_sec": 0.9,
            "camera_ocr_max_pending": 2,
            "camera_ocr_settle_sec": 0.22,
            "enable_detector_fallback": True,
            "enable_small_plate_rescue": True,
            "ocr_second_pass_7char_enabled": True,
        })
        logger.info("Perfil aplicado: jetson-quality")
        return

    if profile == "jetson-production":
        config.update({
            "yolo_imgsz": 512,
            "camera_detection_stride": 2,
            "camera_preview_width": 720,
            "camera_display_width": 720,
            "camera_ocr_cooldown_sec": 0.55,
            "camera_ocr_max_pending": 1,
            "camera_ocr_settle_sec": 0.16,
            "camera_ocr_detector_box_expand": 1.20,
            "camera_ocr_use_worker": True,
            "camera_ocr_use_detector_box": True,
            "camera_ocr_upscale_enabled": True,
            "camera_ocr_enhance_enabled": True,
            "camera_dump_ocr_input_enabled": False,
            "camera_dump_ocr_input_only": False,
            "camera_debug_ocr_enabled": False,
            "enable_detector_fallback": True,
            "enable_small_plate_rescue": True,
            "ocr_second_pass_7char_enabled": True,
        })
        logger.info("Perfil aplicado: jetson-production")
        return

    if profile == "jetson-realtime":
        config.update({
            "yolo_imgsz": 320,
            "camera_detection_stride": 10,
            "camera_preview_width": 416,
            "camera_display_width": 640,
            "camera_ocr_cooldown_sec": 1.4,
            "camera_ocr_max_pending": 1,
            "camera_ocr_settle_sec": 0.20,
            "camera_ocr_detector_box_expand": 1.20,
            "camera_ocr_use_worker": True,
            "camera_ocr_use_detector_box": True,
            "camera_ocr_upscale_enabled": True,
            "camera_ocr_enhance_enabled": True,
            "camera_dump_ocr_input_enabled": False,
            "camera_dump_ocr_input_only": False,
            "camera_debug_ocr_enabled": False,
            "enable_detector_fallback": False,
            "enable_small_plate_rescue": False,
            "ocr_second_pass_7char_enabled": True,
        })
        logger.info("Perfil aplicado: jetson-realtime")
        return

    logger.warning("Perfil desconocido: %s. Se mantiene configuración actual.", profile_name)

def main():
    parser = argparse.ArgumentParser(
        description="License Plate Detection en Jetson Nano"
    )
    parser.add_argument(
        "--video",
        type=str,
        default="0",
        help="Fuente de video (0 para cámara, o ruta a archivo)"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Ruta a una imagen para una ejecución única"
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
        help="Desactiva ventana OpenCV (útil en Jetson headless)"
    )
    parser.add_argument(
        "--camera-healthcheck",
        action="store_true",
        help="Abre solo la cámara y muestra/guarda frames sin detector ni OCR"
    )
    parser.add_argument(
        "--camera-healthcheck-seconds",
        type=int,
        default=15,
        help="Duración del healthcheck de cámara en segundos"
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
        help="Perfil de ejecución para edge: current, jetson-stable, jetson-quality, jetson-production o jetson-realtime"
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
        help="Ruta explícita del detector (.pt/.h5/.tflite). Si no se define, se mantiene el de CONFIG"
    )
    parser.add_argument(
        "--ocr-model",
        type=str,
        default=None,
        help="Ruta explícita del OCR. Si no se define, usa <model-dir>/ocr_int8.tflite"
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
        help="Perfil OCR: balanced para cercanas/mixtas, far para cámaras lejanas"
    )
    parser.add_argument(
        "--camera-grayscale",
        action="store_true",
        help="Convierte la cámara a escala de grises antes de detectar y leer OCR"
    )
    parser.add_argument(
        "--dump-ocr-crops",
        action="store_true",
        help="Guarda en outputs/ocr_debug el recorte exacto enviado al OCR"
    )
    parser.add_argument(
        "--dump-ocr-crops-only",
        action="store_true",
        help="Solo guarda recortes OCR y no ejecuta inferencia OCR (modo depuración)"
    )
    parser.add_argument(
        "--camera-ocr-expand",
        type=float,
        default=None,
        help="Factor de expansión de la caja del detector antes de enviar al OCR (ej: 1.2)"
    )
    parser.add_argument(
        "--camera-display-width",
        type=int,
        default=None,
        help="Ancho de visualización de ventana (solo render). No afecta inferencia"
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
    print("\n" + "🚀" * 20)
    print(" JETSON NANO - REAL-TIME LICENSE PLATE DETECTION")
    print("🚀" * 20 + "\n")
    
    main()
    
    print("""
    
    ✨ INFERENCE COMPLETADO
    
    Para monitorear recursos en tiempo real (en otra terminal):
        watch -n 0.5 'nvidia-smi'
    
    Logs útiles:
        dmesg | tail -20  (para errores del kernel)
        dpkg -l | grep cuda  (verificar CUDA)
    """)
