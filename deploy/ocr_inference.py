# -*- coding: utf-8 -*-
"""OCR inference module extracted from inference_jetson.py."""

import cv2
import numpy as np
import re
import shutil
from pathlib import Path
import logging

from runtime_config import CONFIG

try:
    import tensorflow as tf
except Exception:
    tf = None

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

logger = logging.getLogger(__name__)


def _require_tensorflow(feature_name):
    if tf is None:
        raise RuntimeError(
            f"TensorFlow no está instalado, pero se intentó usar {feature_name}. "
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
                logger.warning("âš ï¸ RapidOCR no disponible. Se usarÃ¡ fallback OCR existente.")
            else:
                try:
                    self.rapid_ocr = RapidOCR()
                    self.backend = "rapidocr"
                    self.ocr_mode = "rapid"
                    logger.info("âœ“ OCR listo (backend=rapidocr, modo=lpr)")
                    return
                except Exception as e:
                    logger.warning("âš ï¸ No se pudo inicializar RapidOCR: %s", e)

        if self.engine in {"paddle", "paddle-latin", "paddle_latin"}:
            if PaddleOCR is None:
                logger.warning("âš ï¸ PaddleOCR no disponible. Se usarÃ¡ fallback OCR existente.")
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
                    logger.info("âœ“ OCR listo (backend=paddleocr, modo=latin)")
                    return
                except Exception as e:
                    logger.warning("âš ï¸ No se pudo inicializar PaddleOCR: %s", e)

        if self.engine in {"simple", "tesseract", "paddle", "paddle-latin", "paddle_latin", "rapidocr", "rapidocr-lpr", "rapid"}:
            if pytesseract is not None:
                tesseract_cmd = _resolve_tesseract_cmd()
                if tesseract_cmd is not None:
                    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
                self.backend = "tesseract"
                self.ocr_mode = "simple"
                logger.info("âœ“ OCR listo (backend=tesseract, modo=simple)")
                return
            if not bool(CONFIG.get("ocr_simple_fallback_to_model", True)):
                raise RuntimeError("pytesseract no disponible y ocr_simple_fallback_to_model=False")
            logger.warning("âš ï¸ pytesseract no disponible. Se usarÃ¡ OCR del modelo como fallback.")

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
                    "âš ï¸ OCR TFLite requiere Flex ops. Usando fallback Keras: %s",
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
            logger.info("âœ“ OCR listo (backend=%s, modo=ctc)", self.backend)
        else:
            logger.info("âœ“ OCR listo (backend=%s, modo=char)", self.backend)

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
        """Genera variantes livianas para OCR clÃ¡sico."""
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

        # Longitudes tÃ­picas de placa local (6-8) y preferencia fuerte por 7.
        if 6 <= n <= 8:
            score += float(CONFIG.get("ctc_len_bonus", 0.12))
        if n == 7:
            score += float(CONFIG.get("ctc_len7_bonus", 0.08))

        # PriorizaciÃ³n de patrÃ³n tipo LLLNNNN cuando aparezca.
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

        # Variante CLAHE para contrastes difÃ­ciles.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        vars_out.append(cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR))

        # Variante sharpen suave para placas pequeÃ±as o borrosas.
        blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
        sharp = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)
        vars_out.append(cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR))

        return vars_out

    def _is_small_plate(self, plate_image):
        """Detecta si una placa es pequeÃ±a basÃ¡ndose en su altura."""
        h, w = plate_image.shape[:2]
        threshold = float(CONFIG.get("ocr_small_plate_threshold", 100))
        return h < threshold

    def _upscale_for_ocr(self, plate_image):
        """Upscale agresivo para placas MUY pequeÃ±as."""
        h, w = plate_image.shape[:2]
        upscale_threshold = float(CONFIG.get("ocr_small_plate_upscale_threshold", 50))
        if h >= upscale_threshold:
            return plate_image
        
        upscale_factor = float(CONFIG.get("ocr_small_plate_upscale", 2.0))
        new_h = max(int(upscale_threshold), int(h * upscale_factor))
        new_w = int(w * new_h / h)
        return cv2.resize(plate_image, (int(new_w), int(new_h)), interpolation=cv2.INTER_CUBIC)

    def _ocr_plate_variants(self, plate_image):
        """Genera variantes de OCR segÃºn perfil y tamaÃ±o de placa."""
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
        """Variantes agresivas de preprocesamiento para placas pequeÃ±as."""
        # Upscale agresivo si es muy pequeÃ±o
        plate_upscaled = self._upscale_for_ocr(plate_image)
        
        base = plate_upscaled
        vars_out = [base]

        if len(base.shape) == 3:
            gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
        else:
            gray = base

        # ===== Variante 1: CLAHE MÃS AGRESIVO =====
        clip_limit = float(CONFIG.get("ocr_small_plate_clahe_clip", 4.0))
        clahe_aggressive = cv2.createCLAHE(
            clipLimit=clip_limit, 
            tileGridSize=(4, 4)  # grid mÃ¡s fino para mejor adaptaciÃ³n
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
        """Clasifica un carÃ¡cter"""
        
        # Preprocesar
        char_resized = cv2.resize(char_image, (self.input_size, self.input_size))
        if len(char_resized.shape) == 2:
            char_resized = cv2.cvtColor(char_resized, cv2.COLOR_GRAY2BGR)
        
        char_normalized = char_resized.astype(np.float32) / 255.0
        char_input = np.expand_dims(char_normalized, 0)

        if self.backend != "tflite":
            raise RuntimeError("recognize_character solo estÃ¡ soportado en backend tflite")

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
            raise RuntimeError("recognize_character solo estÃ¡ soportado en backend tflite")

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

        # Si no respeta separaciÃ³n letra/dÃ­gito, conserva OCR crudo para soportar formatos no estÃ¡ndar.
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
                    # ConfusiÃ³n frecuente en la letra central: P/D por borde derecho tenue.
                    out_ch = "D"
                    changes += 1
                elif (
                    low_conf
                    and bool(CONFIG.get("ocr_pos_last_letter_xc_swap", True))
                    and idx == (digit_start - 1)
                    and ch in ("X", "C")
                    and changes < max_changes
                ):
                    # En la Ãºltima letra del bloque, X/C se confunden con trazos finos o reflejos.
                    out_ch = "C" if ch == "X" else "X"
                    changes += 1
            else:
                if ch in LETTER_TO_DIGIT_VISUAL and changes < max_changes:
                    out_ch = LETTER_TO_DIGIT_VISUAL[ch]
                    changes += 1
                elif low_conf and idx == (digit_start + 1) and ch == "8" and changes < max_changes:
                    # En baja confianza, 8 y 9 se confunden con frecuencia en la posiciÃ³n central numÃ©rica.
                    out_ch = "9"
                    changes += 1
                elif (
                    low_conf
                    and bool(CONFIG.get("ocr_pos_last_digit_24_swap", True))
                    and idx == (len(clean) - 1)
                    and ch in ("2", "4")
                    and changes < max_changes
                ):
                    # En el Ãºltimo dÃ­gito, 2/4 puede variar por perspectiva o recorte lateral.
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
        """Beam search corto para elegir la cadena de placa mÃ¡s plausible."""
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

        # Siempre compara contra fallback char cuando estÃ© disponible;
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
            num_chars: nÃºmero de caracteres esperados
        
        Returns:
            plate_string: string de placa reconocida
        """
        plate_text, _ = self.segment_and_recognize_with_conf(plate_image)
        return plate_text

