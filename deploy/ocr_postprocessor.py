"""
Post-procesador OCR para mejorar predicciones de placas.
- Rechaza predicciones inválidas (como "ECUADOR")
- Intenta ajustar ROI si la predicción es inválida
- Aplica correcciones visuales comunes
"""
import re
import cv2
import numpy as np
from typing import Tuple, Optional


def validate_plate_format(text: str) -> bool:
    """Valida que el texto tenga formato LLL-NNN o LLL-NNNN."""
    if not text:
        return False
    pattern = r"^[A-Z]{3}-\d{3,4}$"
    return bool(re.match(pattern, text.strip().upper()))


def normalize_plate_text(text: str) -> str:
    """Normaliza texto de placa."""
    if not text:
        return ""
    t = text.upper().strip()
    t = t.replace('–', '-').replace('_', '-').replace(' ', '')
    return t


def correct_common_ocr_errors(text: str) -> str:
    """Corrige errores visuales comunes en OCR."""
    corrections = {
        'O': '0',  # O → 0
        'I': '1',  # I → 1
        'Z': '2',  # Z → 2
        'B': '8',  # B → 8 (a veces)
        'S': '5',  # S → 5
        'G': '6',  # G → 6
        'T': '7',  # T → 7
        'L': '1',  # L → 1
    }
    
    result = []
    for i, char in enumerate(text):
        # Si la posición debe ser dígito y es letra que se parece a dígito
        if i > 3 and char in corrections:  # Después del guión LLL-
            result.append(corrections[char])
        else:
            result.append(char)
    
    return ''.join(result)


def extract_plate_from_text(text: str) -> Optional[str]:
    """
    Intenta extraer una placa válida de texto que puede contener caracteres adicionales.
    Ej: "ECUADOR AAB-4475 Crea" → "AAB-4475"
    """
    if not text:
        return None
    
    # Buscar patrón LLL-NNN o LLL-NNNN
    match = re.search(r'[A-Z]{3}-\d{3,4}', text.upper())
    if match:
        return match.group()
    
    # Si no hay guión, intentar sin guión LLL-NNN
    match = re.search(r'[A-Z]{3}\d{3,4}', text.upper())
    if match:
        extracted = match.group()
        # Formatear con guión
        return f"{extracted[:3]}-{extracted[3:]}"
    
    return None


def adjust_roi_for_invalid_ocr(
    frame: np.ndarray,
    roi: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    invalid_text: str,
    ocr_fn
) -> Tuple[Optional[str], Tuple[int, int, int, int]]:
    """
    Si OCR devuelve texto inválido, intenta ajustar el ROI moviendo hacia abajo/arriba.
    
    Returns:
        (mejor_prediccion, nuevas_coordenadas)
    """
    h, w = frame.shape[:2]
    
    # Estrategia 1: Detectar si OCR capturó texto de encabezado
    if "ECUADOR" in invalid_text.upper() or "REPÚBLICA" in invalid_text.upper():
        # Mover ROI hacia abajo (ignorar encabezado)
        roi_height = y2 - y1
        adjustments = [
            (0.3, 0.0),   # Bajar 30% de la altura
            (0.4, 0.0),   # Bajar 40%
            (0.2, 0.0),   # Bajar 20%
        ]
        
        for y_offset_pct, _ in adjustments:
            y_offset = int(roi_height * y_offset_pct)
            y1_adj = min(y1 + y_offset, h - 1)
            y2_adj = min(y2 + y_offset, h)
            
            if y2_adj <= y1_adj:
                continue
            
            roi_adj = frame[y1_adj:y2_adj, x1:x2]
            if roi_adj.size == 0:
                continue
            
            try:
                text_adj, _ = ocr_fn(roi_adj)
                text_adj = normalize_plate_text(text_adj)
                
                # Intenta extracción y corrección
                extracted = extract_plate_from_text(text_adj)
                if extracted:
                    if validate_plate_format(extracted):
                        return extracted, (x1, y1_adj, x2, y2_adj)
                
                # Intenta corrección de errores comunes
                corrected = correct_common_ocr_errors(text_adj)
                if validate_plate_format(corrected):
                    return corrected, (x1, y1_adj, x2, y2_adj)
            except Exception:
                pass
    
    # Estrategia 2: Ajustar horizontalmente
    adjustments_h = [
        (0, 0.1),    # Desplazar hacia arriba 10%
        (0, -0.1),   # Desplazar hacia abajo 10%
        (0.05, 0),   # Mover derecha 5%
        (-0.05, 0),  # Mover izquierda 5%
    ]
    
    for x_offset_pct, y_offset_pct in adjustments_h:
        roi_width = x2 - x1
        roi_height = y2 - y1
        
        x_offset = int(roi_width * x_offset_pct)
        y_offset = int(roi_height * y_offset_pct)
        
        x1_adj = max(0, x1 + x_offset)
        x2_adj = min(w, x2 + x_offset)
        y1_adj = max(0, y1 + y_offset)
        y2_adj = min(h, y2 + y_offset)
        
        if x2_adj <= x1_adj or y2_adj <= y1_adj:
            continue
        
        roi_adj = frame[y1_adj:y2_adj, x1_adj:x2_adj]
        if roi_adj.size == 0:
            continue
        
        try:
            text_adj, _ = ocr_fn(roi_adj)
            text_adj = normalize_plate_text(text_adj)
            
            if validate_plate_format(text_adj):
                return text_adj, (x1_adj, y1_adj, x2_adj, y2_adj)
            
            # Intenta extracción
            extracted = extract_plate_from_text(text_adj)
            if extracted and validate_plate_format(extracted):
                return extracted, (x1_adj, y1_adj, x2_adj, y2_adj)
            
            # Intenta corrección
            corrected = correct_common_ocr_errors(text_adj)
            if validate_plate_format(corrected):
                return corrected, (x1_adj, y1_adj, x2_adj, y2_adj)
        except Exception:
            pass
    
    return None, (x1, y1, x2, y2)


def postprocess_ocr_prediction(
    text: str,
    frame: Optional[np.ndarray] = None,
    roi_coords: Optional[Tuple[int, int, int, int]] = None,
    ocr_fn = None
) -> Tuple[str, bool, Optional[Tuple[int, int, int, int]]]:
    """
    Post-procesa predicción OCR.
    
    Returns:
        (texto_procesado, es_valido, coordenadas_ajustadas)
    """
    if not text:
        return "", False, roi_coords
    
    # Normalizar
    normalized = normalize_plate_text(text)
    
    # Validación 1: Formato exacto
    if validate_plate_format(normalized):
        return normalized, True, roi_coords
    
    # Validación 2: Extracción de placa dentro de texto
    extracted = extract_plate_from_text(normalized)
    if extracted:
        if validate_plate_format(extracted):
            return extracted, True, roi_coords
    
    # Validación 3: Corrección de errores comunes
    corrected = correct_common_ocr_errors(normalized)
    if validate_plate_format(corrected):
        return corrected, True, roi_coords
    
    # Si todo falla y tenemos frame + ROI + función OCR, intentar ajustar
    if frame is not None and roi_coords is not None and ocr_fn is not None:
        x1, y1, x2, y2 = roi_coords
        roi = frame[y1:y2, x1:x2]
        
        adjusted_text, adjusted_coords = adjust_roi_for_invalid_ocr(
            frame, roi, x1, y1, x2, y2, text, ocr_fn
        )
        
        if adjusted_text:
            return adjusted_text, True, adjusted_coords
    
    # Rechazar predicción inválida
    return text, False, roi_coords


def batch_postprocess_predictions(
    predictions: list[dict]
) -> list[dict]:
    """
    Post-procesa un lote de predicciones OCR.
    
    Cada dict debe tener: {'text': str, 'valid': bool}
    """
    processed = []
    
    for pred in predictions:
        text = pred.get('text', '')
        processed_text, is_valid, _ = postprocess_ocr_prediction(text)
        
        pred_copy = pred.copy()
        pred_copy['text'] = processed_text
        pred_copy['valid'] = is_valid
        processed.append(pred_copy)
    
    return processed
