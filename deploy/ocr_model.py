# -*- coding: utf-8 -*-
"""
Modelo OCR CRNN+CTC - Python Puro
Generado desde: ocr_crnn_best.keras

USAR:
    from ocr_model import load_ocr_crnn, predict_plate
    import cv2
    
    model = load_ocr_crnn()
    imagen = cv2.imread("placa.jpg")
    texto, conf = predict_plate(model, imagen)
    print(f"Placa: {texto} ({conf:.1%})")

INSTALAR:
    pip install tensorflow opencv-python numpy
"""

import numpy as np
import base64
import pickle
import tensorflow as tf
from typing import Tuple, Optional

# ============================================================================
# INFORMACIÓN DEL MODELO
# ============================================================================

MODEL_INFO = {
    "name": "ocr_crnn",
    "input_shape": (48, 192, 1),
    "output_shape": (None, 38),  # (timesteps, num_classes)
    "num_parameters": 0,
    "charset": "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    "has_weights": True,
}


# ============================================================================
# ARQUITECTURA DEL MODELO
# ============================================================================

def build_ocr_crnn() -> tf.keras.Model:
    """Construye la arquitectura OCR CRNN+CTC."""
    
    inputs = tf.keras.Input(shape=(48, 192, 1), name="input_image")
    x = inputs
    
    # === EXTRACCIÓN DE CARACTERÍSTICAS (CNN) ===
    
    # Bloque 1: Conv 64
    x = tf.keras.layers.Conv2D(64, (3, 3), padding="same", activation="relu", name="conv1")(x)
    x = tf.keras.layers.BatchNormalization(name="bn1")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool1")(x)
    
    # Bloque 2: Conv 128
    x = tf.keras.layers.Conv2D(128, (3, 3), padding="same", activation="relu", name="conv2")(x)
    x = tf.keras.layers.BatchNormalization(name="bn2")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool2")(x)
    
    # Bloque 3: Conv 256
    x = tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu", name="conv3")(x)
    x = tf.keras.layers.BatchNormalization(name="bn3")(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), name="pool3")(x)
    
    # Bloque 4: Conv 256 (aspecto 2:1)
    x = tf.keras.layers.Conv2D(256, (3, 3), padding="same", activation="relu", name="conv4")(x)
    x = tf.keras.layers.BatchNormalization(name="bn4")(x)
    x = tf.keras.layers.MaxPooling2D((2, 1), name="pool4")(x)
    
    # Bloque 5: Conv 512
    x = tf.keras.layers.Conv2D(512, (3, 3), padding="same", activation="relu", name="conv5")(x)
    x = tf.keras.layers.BatchNormalization(name="bn5")(x)
    x = tf.keras.layers.MaxPooling2D((2, 1), name="pool5")(x)
    
    # === MODELADO DE SECUENCIAS (RNN) ===
    
    # Reshape estático para LSTM: (batch, 1, 24, 512) → (batch, 24, 512)
    x = tf.keras.layers.Reshape((24, 512), name="reshape_rnn")(x)
    
    # Bidirectional LSTM 1
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True, name="lstm1"),
        name="bilstm1"
    )(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout1")(x)
    
    # Bidirectional LSTM 2
    x = tf.keras.layers.Bidirectional(
        tf.keras.layers.LSTM(128, return_sequences=True, name="lstm2"),
        name="bilstm2"
    )(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout2")(x)
    
    # === CLASIFICACIÓN (Dense) ===
    outputs = tf.keras.layers.Dense(38, activation="softmax", name="output")(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"{MODEL_INFO['name']}")
    return model
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"{MODEL_INFO['name']}")
    return model


# ============================================================================
# DESERIALIZACIÓN DE PESOS
# ============================================================================

WEIGHTS_B64 = """"""


def _load_weights_from_b64() -> Optional[list]:
    """Decodifica y carga pesos desde base64."""
    if not WEIGHTS_B64.strip():
        return None
    
    try:
        encoded = WEIGHTS_B64.strip()
        decoded = base64.b64decode(encoded)
        weights_dict = pickle.loads(decoded)
        return weights_dict.get("weights", [])
    except Exception as e:
        print(f"⚠ Error cargando pesos: {e}")
        return None


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def load_ocr_crnn() -> tf.keras.Model:
    """
    Carga el modelo OCR con pesos preentrenados.
    
    Returns:
        Modelo Keras listo para predicción
    """
    print(f"🔄 Construyendo modelo {MODEL_INFO['name']}...")
    model = build_ocr_crnn()
    
    if MODEL_INFO["has_weights"]:
        print(f"📥 Cargando pesos...")
        weights = _load_weights_from_b64()
        
        if weights:
            try:
                model.set_weights(weights)
                print(f"✓ Pesos cargados correctamente")
            except Exception as e:
                print(f"⚠ Error asignando pesos: {e}")
                print(f"  Modelo funcionará con inicialización aleatoria")
        else:
            print(f"⚠ Sin pesos disponibles")
    else:
        print(f"ℹ Modelo sin pesos incluidos")
    
    return model


# ============================================================================
# UTILIDADES DE PREDICCIÓN
# ============================================================================

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocesa imagen para OCR.
    
    Input:
        Imagen BGR/RGB o grayscale (cualquier tamaño)
    Output:
        Imagen normalizada (1, 48, 192, 1) lista para modelo
    """
    import cv2
    
    # Grayscale
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize a (48, 192)
    image = cv2.resize(image, (192, 48))
    
    # Normalizar [0, 1]
    image = image.astype(np.float32) / 255.0
    
    # Add channel: (48, 192) → (48, 192, 1)
    image = np.expand_dims(image, axis=-1)
    
    # Add batch: (48, 192, 1) → (1, 48, 192, 1)
    image = np.expand_dims(image, axis=0)
    
    return image


def decode_ctc(output: np.ndarray, charset: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") -> Tuple[str, float]:
    """
    Decodifica salida CTC a texto (greedy decoding).
    
    Args:
        output: Salida modelo shape (1, timesteps, 38) o (timesteps, 38)
        charset: Caracteres válidos
    
    Returns:
        (texto_placa, confianza_promedio)
    """
    # Remover batch si existe
    if len(output.shape) == 3:
        output = output[0]
    
    # Argmax por timestep
    predicted_classes = np.argmax(output, axis=-1)
    
    # Decodificar: remover blanks (0), duplicados
    text = ""
    prev_class = -1
    confidences = []
    
    for t, class_idx in enumerate(predicted_classes):
        # Saltar blanks y duplicados
        if class_idx > 0 and class_idx != prev_class:
            char_idx = class_idx - 1
            if char_idx < len(charset):
                text += charset[char_idx]
                confidences.append(output[t, class_idx])
        
        prev_class = class_idx
    
    confidence = float(np.mean(confidences)) if confidences else 0.0
    return text, confidence


def predict_plate(
    model: tf.keras.Model,
    image: np.ndarray,
    verbose: bool = False
) -> Tuple[str, float]:
    """
    Predice placa desde imagen.
    
    Args:
        model: Modelo cargado con load_ocr_crnn()
        image: Imagen BGR/RGB o grayscale
        verbose: Imprimir debug info
    
    Returns:
        (texto_placa, confianza) ej: ("ABC1234", 0.92)
    """
    # Preprocesar
    processed = preprocess_image(image)
    
    if verbose:
        print(f"[DEBUG] Input shape: {image.shape}")
        print(f"[DEBUG] Processed shape: {processed.shape}")
    
    # Predicción
    output = model.predict(processed, verbose=0)
    
    if verbose:
        print(f"[DEBUG] Output shape: {output.shape}")
    
    # Decodificar
    text, conf = decode_ctc(output)
    
    if verbose:
        print(f"[DEBUG] Texto: {text}, Confianza: {conf:.2%}")
    
    return text, conf


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*70)
    print(f"Modelo OCR: {MODEL_INFO['name']}")
    print("="*70)
    
    print(f"\n📊 Información:")
    print(f"  Input shape: {MODEL_INFO['input_shape']}")
    print(f"  Output shape: {MODEL_INFO['output_shape']}")
    print(f"  Parámetros: {MODEL_INFO['num_parameters']}")
    print(f"  Charset: {MODEL_INFO['charset']}")
    
    # Cargar
    try:
        model = load_ocr_crnn()
        print(f"\n✓ Modelo listo")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        exit(1)
    
    # Prueba
    print(f"\n🧪 Predicción de prueba...")
    try:
        dummy_image = np.random.randint(50, 200, (48, 192), dtype=np.uint8)
        text, conf = predict_plate(model, dummy_image, verbose=True)
        print(f"✓ Resultado: {text} (conf: {conf:.1%})")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print(f"\n" + "="*70 + "\n")
    print(f"USAR:")
    print(f"  from ocr_model import load_ocr_crnn, predict_plate")
    print(f"  model = load_ocr_crnn()")
    print(f"  texto, conf = predict_plate(model, cv2.imread('placa.jpg'))")
    print(f"\n" + "="*70 + "\n")
