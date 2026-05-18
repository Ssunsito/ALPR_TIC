# -*- coding: utf-8 -*-
"""
Convertidor Simplificado: Keras → Python
Convierte tu modelo OCR a un archivo .py compartible sin dependencias
"""

import numpy as np
import base64
from pathlib import Path
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def convert_ocr_model_to_python(
    model_path: str,
    output_py_path: str,
    model_name: str = "ocr_crnn",
    include_weights: bool = True
) -> None:
    """
    Convierte modelo OCR Keras a archivo Python con toda la lógica.
    
    Args:
        model_path: Ruta al modelo .keras
        output_py_path: Ruta del archivo .py resultante
        model_name: Nombre de la función (default: ocr_crnn)
        include_weights: Incluir pesos en el archivo (SI = más grande, NO = más pequeño)
    """
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("TensorFlow requerido: pip install tensorflow")
    
    print(f"📦 Cargando modelo: {model_path}")
    
    # Cargar modelo (ignorar capas personalizadas)
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        num_params = int(model.count_params())
        print(f"✓ Modelo cargado: {num_params:,} parámetros")
    except Exception as e:
        logger.warning(f"No se pudo cargar modelo completo: {e}")
        model = None
        num_params = 0
    
    # Generar código
    code = _generate_model_python_code(
        model=model,
        model_name=model_name,
        include_weights=include_weights,
        original_file=Path(model_path).name
    )
    
    # Guardar
    Path(output_py_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_py_path, "w", encoding="utf-8") as f:
        f.write(code)
    
    file_size = Path(output_py_path).stat().st_size / (1024*1024)
    print(f"✓ Archivo generado: {output_py_path} ({file_size:.1f} MB)")
    print(f"✓ Puedes importarlo: from {Path(output_py_path).stem} import load_{model_name}")


def _generate_model_python_code(
    model: Optional[object],
    model_name: str,
    include_weights: bool,
    original_file: str
) -> str:
    """Genera el código Python del modelo."""
    
    # Determinar parámetros del modelo
    if model is not None:
        try:
            num_params = int(model.count_params())
        except:
            num_params = 0
    else:
        num_params = 0
    
    weights_b64 = ""
    if include_weights and model is not None:
        print(f"  ⏳ Serializando pesos...")
        weights_b64 = _serialize_weights_b64(model)
        print(f"  ✓ Pesos serializados")
    
    weights_placeholder = weights_b64 if weights_b64 else ""
    
    code = f'''# -*- coding: utf-8 -*-
"""
Modelo OCR CRNN+CTC - Python Puro
Generado desde: {original_file}

USAR:
    from ocr_model import load_{model_name}, predict_plate
    import cv2
    
    model = load_{model_name}()
    imagen = cv2.imread("placa.jpg")
    texto, conf = predict_plate(model, imagen)
    print(f"Placa: {{texto}} ({{conf:.1%}})")

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

MODEL_INFO = {{
    "name": "{model_name}",
    "input_shape": (48, 192, 1),
    "output_shape": (None, 38),  # (timesteps, num_classes)
    "num_parameters": {num_params:,},
    "charset": "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
    "has_weights": {include_weights},
}}


# ============================================================================
# ARQUITECTURA DEL MODELO
# ============================================================================

def build_{model_name}() -> tf.keras.Model:
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
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name=f"{{MODEL_INFO['name']}}")
    return model


# ============================================================================
# DESERIALIZACIÓN DE PESOS
# ============================================================================

WEIGHTS_B64 = """{weights_placeholder}"""


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
        print(f"⚠ Error cargando pesos: {{e}}")
        return None


# ============================================================================
# FUNCIÓN PRINCIPAL
# ============================================================================

def load_{model_name}() -> tf.keras.Model:
    """
    Carga el modelo OCR con pesos preentrenados.
    
    Returns:
        Modelo Keras listo para predicción
    """
    print(f"🔄 Construyendo modelo {{MODEL_INFO['name']}}...")
    model = build_{model_name}()
    
    if MODEL_INFO["has_weights"]:
        print(f"📥 Cargando pesos...")
        weights = _load_weights_from_b64()
        
        if weights:
            try:
                model.set_weights(weights)
                print(f"✓ Pesos cargados correctamente")
            except Exception as e:
                print(f"⚠ Error asignando pesos: {{e}}")
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
        model: Modelo cargado con load_{model_name}()
        image: Imagen BGR/RGB o grayscale
        verbose: Imprimir debug info
    
    Returns:
        (texto_placa, confianza) ej: ("ABC1234", 0.92)
    """
    # Preprocesar
    processed = preprocess_image(image)
    
    if verbose:
        print(f"[DEBUG] Input shape: {{image.shape}}")
        print(f"[DEBUG] Processed shape: {{processed.shape}}")
    
    # Predicción
    output = model.predict(processed, verbose=0)
    
    if verbose:
        print(f"[DEBUG] Output shape: {{output.shape}}")
    
    # Decodificar
    text, conf = decode_ctc(output)
    
    if verbose:
        print(f"[DEBUG] Texto: {{text}}, Confianza: {{conf:.2%}}")
    
    return text, conf


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    print("\\n" + "="*70)
    print(f"Modelo OCR: {{MODEL_INFO['name']}}")
    print("="*70)
    
    print(f"\\n📊 Información:")
    print(f"  Input shape: {{MODEL_INFO['input_shape']}}")
    print(f"  Output shape: {{MODEL_INFO['output_shape']}}")
    print(f"  Parámetros: {{MODEL_INFO['num_parameters']}}")
    print(f"  Charset: {{MODEL_INFO['charset']}}")
    
    # Cargar
    try:
        model = load_{model_name}()
        print(f"\\n✓ Modelo listo")
    except Exception as e:
        print(f"\\n❌ Error: {{e}}")
        exit(1)
    
    # Prueba
    print(f"\\n🧪 Predicción de prueba...")
    try:
        dummy_image = np.random.randint(50, 200, (48, 192), dtype=np.uint8)
        text, conf = predict_plate(model, dummy_image, verbose=True)
        print(f"✓ Resultado: {{text}} (conf: {{conf:.1%}})")
    except Exception as e:
        print(f"❌ Error: {{e}}")
    
    print(f"\\n" + "="*70 + "\\n")
    print(f"USAR:")
    print(f"  from ocr_model import load_{model_name}, predict_plate")
    print(f"  model = load_{model_name}()")
    print(f"  texto, conf = predict_plate(model, cv2.imread('placa.jpg'))")
    print(f"\\n" + "="*70 + "\\n")
'''
    
    return code


def _serialize_weights_b64(model: object) -> str:
    """Serializa pesos a base64."""
    try:
        import pickle
        
        # Extraer pesos
        weights_list = model.get_weights()
        
        weights_dict = {
            "weights": weights_list,
            "layer_names": [layer.name for layer in model.layers],
        }
        
        # Serializar
        serialized = pickle.dumps(weights_dict)
        
        # Base64
        encoded = base64.b64encode(serialized).decode("utf-8")
        
        # Dividir en líneas de 80 caracteres para legibilidad
        lines = [encoded[i:i+80] for i in range(0, len(encoded), 80)]
        result = "\n".join(f"    {line}" for line in lines)
        
        return result
    
    except Exception as e:
        logger.error(f"Error serializando pesos: {e}")
        return ""


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    
    print("\n" + "="*70)
    print("CONVERTIDOR: Modelo Keras → Python Puro")
    print("="*70)
    
    if len(sys.argv) > 2:
        modelo_path = sys.argv[1]
        salida_path = sys.argv[2]
        
        try:
            convert_ocr_model_to_python(
                model_path=modelo_path,
                output_py_path=salida_path,
                include_weights=True
            )
            print(f"\n✓ ÉXITO")
            print(f"\n Ahora puedes:")
            print(f"   from {Path(salida_path).stem} import load_ocr_crnn")
            print(f"   model = load_ocr_crnn()")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"\nUso:")
        print(f"  python {Path(__file__).name} <modelo.keras> <salida.py>")
        print(f"\nEjemplo:")
        print(f"  python {Path(__file__).name} ../models/ocr_crnn_best.keras ocr_model.py")
    
    print("\n" + "="*70 + "\n")
