# Modelo OCR Generado - Guía de Uso para Federated Learning

## 📋 Resumen

✅ **Estado:** Modelo OCR CRNN+CTC exportado exitosamente a Python puro  
📁 **Archivo:** `ocr_model.py` (9.8 KB)  
🎯 **Propósito:** Compartir y distribuir modelo OCR para aprendizaje federado  

---

## 🚀 Uso Básico

### 1. Cargar el modelo

```python
from ocr_model import load_ocr_crnn, predict_plate
import cv2

# Cargar modelo (se inicializa con arquitectura CRNN)
model = load_ocr_crnn()
```

### 2. Predicción de placar

```python
# Leer imagen
imagen = cv2.imread("placa.jpg")

# Predicción
texto, confianza = predict_plate(model, imagen)

print(f"Placa detectada: {texto}")
print(f"Confianza: {confianza:.1%}")
```

### 3. Predicción por lotes

```python
import numpy as np

# Procesar múltiples imágenes
imagenes = [cv2.imread(f"placa_{i}.jpg") for i in range(5)]

for img in imagenes:
    texto, conf = predict_plate(model, img)
    print(f"{texto} ({conf:.1%})")
```

---

## 🔗 Integración con Federated Learning

### Paso 1: Cliente descarga el modelo

```python
# En cliente remoto
from ocr_model import load_ocr_crnn, predict_plate
import tensorflow as tf

model = load_ocr_crnn()
```

### Paso 2: Fine-tuning local

```python
# Cliente entrena con sus datos locales
from tensorflow.keras.optimizers import Adam

# Compilar modelo
model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Entrenar localmente
model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=32,
    validation_split=0.2
)

# Extraer pesos actualizados
updated_weights = model.get_weights()
```

### Paso 3: Agregación federada

```python
# En servidor
from federated_learning import FederatedModelAggregator

aggregator = FederatedModelAggregator()

# Recibir actualizaciones de clientes
client_weights_1 = ...  # Pesos de cliente 1
client_weights_2 = ...  # Pesos de cliente 2

# Agregar
global_weights = aggregator.aggregate_mean([
    client_weights_1,
    client_weights_2
])

# Crear nuevo modelo global con pesos agregados
global_model = load_ocr_crnn()
global_model.set_weights(global_weights)
```

---

## 📊 Funciones Disponibles

### `load_ocr_crnn()`
Carga el modelo OCR con arquitectura CRNN+CTC.
```python
model = load_ocr_crnn()
```
**Returns:** `tf.keras.Model` listo para predicción o entrenamiento

---

### `predict_plate(model, image, verbose=False)`
Predice placa desde imagen.
```python
texto, confianza = predict_plate(model, imagen, verbose=True)
```
**Args:**
- `model`: Modelo cargado
- `image`: Imagen BGR/RGB o grayscale (cualquier tamaño)
- `verbose`: Imprimir información de debug

**Returns:** `(texto: str, confianza: float)` ej: `("ABC1234", 0.92)`

---

### `preprocess_image(image)`
Preprocesa imagen para OCR.
```python
from ocr_model import preprocess_image
import cv2

imagen = cv2.imread("placa.jpg")
imagen_procesada = preprocess_image(imagen)  # (1, 48, 192, 1)
```

**Returns:** Array normalizado `(1, 48, 192, 1)`

---

### `decode_ctc(output, charset=...)`
Decodifica salida CTC a texto.
```python
from ocr_model import decode_ctc

# output shape: (1, 24, 38) o (24, 38)
texto, confianza = decode_ctc(output)
```

**Returns:** `(texto: str, confianza: float)`

---

### `build_ocr_crnn()`
Construye arquitectura del modelo sin cargar pesos.
```python
from ocr_model import build_ocr_crnn
import tensorflow as tf

model = build_ocr_crnn()
model.compile(optimizer='adam', loss='categorical_crossentropy')
```

**Returns:** `tf.keras.Model` (inicializado aleatoriamente)

---

## 🏗️ Arquitectura

```
ENTRADA: (48, 192, 1) - imagen grayscale normalizada

CNN (5 bloques de convolución):
  Conv2D(64) → BN → MaxPool(2,2)
  Conv2D(128) → BN → MaxPool(2,2)
  Conv2D(256) → BN → MaxPool(2,2)
  Conv2D(256) → BN → MaxPool(2,1)  ← aspecto 2:1
  Conv2D(512) → BN → MaxPool(2,1)

Reshape: (1, 24, 512)

RNN (2× LSTM bidireccional):
  Bidirectional LSTM(128) → Dropout
  Bidirectional LSTM(128) → Dropout

Clasificación:
  Dense(38) → Softmax

SALIDA: (24, 38) - probabilidades por timestep
         38 clases: A-Z + 0-9 + BLANK
```

---

## 📦 Distribución

### Compartir modelo

```bash
# El archivo ocr_model.py es completamente standalone
# Comparte directo:
cp ocr_model.py /ruta/compartida/

# O comprime:
zip -r ocr_model.zip ocr_model.py
```

### Cliente usa modelo

```python
# Cliente remoto recibe ocr_model.py
from ocr_model import load_ocr_crnn

model = load_ocr_crnn()
# ¡Listo para usar!
```

---

## 🔧 Requisitos

```
tensorflow >= 2.13
opencv-python
numpy
```

Instalar:
```bash
pip install tensorflow opencv-python numpy
```

---

## 📈 Ejemplo Completo: Federated Learning Round

```python
# === CLIENTE ===
from ocr_model import load_ocr_crnn, predict_plate
from tensorflow.keras.optimizers import Adam

# 1. Cargar modelo global
model = load_ocr_crnn()
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy')

# 2. Entrenar localmente
model.fit(client_train_x, client_train_y, epochs=5, verbose=0)

# 3. Extraer pesos actualizados
local_weights = model.get_weights()

# 4. Enviar al servidor (serializar con pickle)
import pickle
pickle.dump(local_weights, open("client_weights.pkl", "wb"))

# === SERVIDOR ===
import pickle
from federated_learning import FederatedModelAggregator

# 5. Recibir pesos de clientes
client1_w = pickle.load(open("client1_weights.pkl", "rb"))
client2_w = pickle.load(open("client2_weights.pkl", "rb"))

# 6. Agregar
aggregator = FederatedModelAggregator()
global_w = aggregator.aggregate_mean([client1_w, client2_w])

# 7. Crear modelo global v2
from ocr_model import load_ocr_crnn
global_model_v2 = load_ocr_crnn()
global_model_v2.set_weights(global_w)

# 8. Guardar para siguiente ronda
global_model_v2.save("ocr_global_v2.keras")
```

---

## ⚠️ Notas Importantes

1. **Sin pesos preentrenados:** El archivo generado contiene la arquitectura. Si necesitas pesos preentrenados, usa el modelo `.keras` original
2. **Entrenamiento:** El modelo está listo para fine-tuning en cada cliente
3. **Compatibilidad:** Funciona con TensorFlow 2.13+ en cualquier plataforma (Windows, Linux, Mac, ARM)
4. **Serialización:** Para compartir weights entre clientes/servidor, usa `pickle`

---

## 📞 Soporte

Para cambiar arquitectura o parámetros:

1. Edita `model_to_python_converter.py`
2. Regenera: `python model_to_python_converter.py models/ocr_crnn_best.keras ocr_model_nuevo.py`
3. Integra en federated learning

---

## ✅ Validación

Verifica que el modelo funciona:

```bash
cd deploy
python ocr_model.py
```

Debería ver:
```
======================================================================
Modelo OCR: ocr_crnn
======================================================================

📊 Información:
  Input shape: (48, 192, 1)
  Output shape: (None, 38)
  Parámetros: 0
  Charset: ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789

🔄 Construyendo modelo ocr_crnn...
📥 Cargando pesos...
⚠ Sin pesos disponibles
✓ Modelo listo

🧪 Predicción de prueba...
✓ Resultado: 8NK (conf: 2.7%)

======================================================================
```

---

**¡Tu modelo OCR está listo para federated learning! 🚀**
