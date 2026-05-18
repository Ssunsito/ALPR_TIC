# FEDERACIÓN DE MODELOS - GUÍA RÁPIDA

## 📚 Contenido

1. [Introducción](#introducción)
2. [Instalación y Setup](#instalación-y-setup)
3. [Agregación Federada (FedAvg)](#agregación-federada-fedavg)
4. [Agregación Ponderada](#agregación-ponderada)
5. [Ensemble de Detectores](#ensemble-de-detectores)
6. [Ensemble de OCR](#ensemble-de-ocr)
7. [Entrenamiento Federado](#entrenamiento-federado)
8. [Exportar Arquitectura](#exportar-arquitectura)
9. [Casos de Uso Reales](#casos-de-uso-reales)
10. [Troubleshooting](#troubleshooting)

---

## Introducción

La **federación de modelos** es un paradigma donde:

- **Múltiples clientes** (oficinas, servidores) entrenan modelos localmente
- **Un servidor central** agrega los modelos sin acceder a datos privados
- **Cada cliente** descarga el modelo global mejorado

### Aplicaciones en ALPR (Automatic License Plate Recognition)

```
Municipio A  ← Entrena OCR localmente → Modelo A
Municipio B  ← Entrena OCR localmente → Modelo B
Municipio C  ← Entrena OCR localmente → Modelo C
       ↓
   AGREGACIÓN (FedAvg)
       ↓
  Modelo Global Mejorado
       ↓
Municipio A, B, C descargan modelo mejorado
```

---

## Instalación y Setup

### Dependencias Base

```bash
pip install tensorflow numpy opencv-python ultralytics
```

### Estructura de Carpetas

```
deploy/
├── federated_learning.py                # Módulo principal
├── federated_learning_examples.py       # Ejemplos prácticos
├── FEDERATED_LEARNING_QUICK_REFERENCE.md  # Este archivo
├── federated_ocr_global_v1.keras        # Modelo global generado
└── federated_config.json                # Configuración
```

### Verificar Instalación

```python
python -c "
from federated_learning import FederatedModelAggregator, DetectorEnsemble, OCREnsemble
print('✓ Módulo federated_learning cargado exitosamente')
"
```

---

## Agregación Federada (FedAvg)

### Uso Básico - Promediar 3 Modelos OCR

```python
from federated_learning import FederatedModelAggregator

# Crear agregador
aggregator = FederatedModelAggregator(framework="keras", verbose=True)

# Registrar modelos de clientes
aggregator.register_client_model("client_a", "models/ocr_crnn_best_a_lr3e4_w2.keras")
aggregator.register_client_model("client_b", "models/ocr_crnn_best_b_lr2e4_w3.keras")
aggregator.register_client_model("client_c", "models/ocr_crnn_best_c_lr4e4_w2.keras")

# Agregación simple (promedio)
aggregated_weights = aggregator.aggregate_mean()

# Guardar modelo global
aggregator.save_aggregated_model_keras("federated_ocr_global.keras")
```

**Resultado:**
- Promedia los pesos de los 3 modelos
- Genera nuevo modelo con arquitectura Keras
- Todos los modelos tienen igual peso (1.0)

### Cuándo Usar FedAvg

✅ Todos los clientes tienen datasets de tamaño similar  
✅ Confianza en la calidad de datos de todos los clientes  
✅ Aplicación simple, máximo rendimiento  
❌ No es ideal si hay clientes con datos muy sesgados

---

## Agregación Ponderada

### Penalizar/Priorizar Clientes

```python
from federated_learning import FederatedModelAggregator

aggregator = FederatedModelAggregator(framework="keras")

# Registrar modelos
aggregator.register_client_model("client_a", "models/ocr_crnn_best_a.keras")
aggregator.register_client_model("client_b", "models/ocr_crnn_best_b.keras")
aggregator.register_client_model("client_c", "models/ocr_crnn_best_c.keras")

# Definir pesos por dataset size
dataset_sizes = {
    "client_a": 1500,  # 1500 imágenes → peso 0.3
    "client_b": 1200,  # 1200 imágenes → peso 0.24
    "client_c": 1800,  # 1800 imágenes → peso 0.36 (máximo)
}

# Agregación ponderada
aggregated = aggregator.aggregate_weighted(dataset_sizes)

# Guardar
aggregator.save_aggregated_model_keras("federated_ocr_weighted.keras")
```

**Ventajas:**
- Cliente con más datos → más influencia en modelo global
- Robusto a clientes con datos limitados

**Fórmula:**
```
peso_cliente = dataset_size / sum(todas_sizes)
weights_agregados = sum(modelo_cliente * peso)
```

---

## Ensemble de Detectores

### Combinar 2 Detectores YOLO

```python
from federated_learning import DetectorEnsemble
import cv2

# Cargar modelos
models = [
    "models/plate_detector_best.pt",        # Placas normales
    "models/plate_detector_smallfar_best.pt" # Placas pequeñas/lejanas
]

ensemble = DetectorEnsemble(models, framework="pytorch", device="cpu")

# Configurar pesos (70% estándar, 30% small)
ensemble.set_weights([0.7, 0.3])

# Predicción
image = cv2.imread("calle.jpg")
detections = ensemble.predict(image, conf_threshold=0.5)

for det in detections:
    print(f"Placa en ({det['x1']:.0f}, {det['y1']:.0f}) "
          f"→ ({det['x2']:.0f}, {det['y2']:.0f})")
    print(f"  Confianza: {det['conf']:.2f}, Votos: {det['num_votes']}")
```

**Cómo Funciona:**
1. Ambos detectores predicen en paralelo
2. Se buscan boxes superpuestas (NMS ensemble)
3. Se promedian confianzas según pesos
4. Se cuenta número de "votos" (models que detectaron la misma placa)

**Resultado:**
- Más robusto: detecta placas que un solo modelo podría perder
- Confianza más confiable (consenso de modelos)

---

## Ensemble de OCR

### Múltiples Modelos OCR con Voting Temporal

```python
from federated_learning import OCREnsemble
import cv2

# 3 modelos OCR
ocr_models = [
    "models/ocr_crnn_best.keras",
    "models/ocr_crnn_best_a_lr3e4_w2.keras",
    "models/ocr_crnn_best_b_lr2e4_w3.keras"
]

# Crear ensemble con voting temporal (3 frames)
ocr_ensemble = OCREnsemble(ocr_models, voting_window=3)

# En cada frame
image = cv2.imread("placa_recortada.jpg")

# Predicción
texto, confianza = ocr_ensemble.predict(image)

print(f"Texto detectado: {texto} (conf: {confianza:.2f})")
print(f"Historial voting: {ocr_ensemble.voting_history}")
```

**Voting Temporal:**
```
Frame 1: AAB1234
Frame 2: AAB-123 (typo)
Frame 3: AAB1234
         ↓
RESULTADO: AAB1234 (mayoria 2/3)
```

**Ventajas:**
- Reduce falsos positivos OCR
- Consenso entre frames = OCR más estable
- Ideal para video streaming

---

## Entrenamiento Federado

### Flujo Multi-Ronda

```python
from federated_learning import FederatedLearningTrainer

# Inicializar con modelo global
trainer = FederatedLearningTrainer(
    global_model_path="models/ocr_crnn_best.keras",
    framework="keras"
)

# RONDA 1
print("=== RONDA 1 ===")
trainer.register_client_update("client_a", "models/ocr_crnn_best_a.keras")
trainer.register_client_update("client_b", "models/ocr_crnn_best_b.keras")
trainer.register_client_update("client_c", "models/ocr_crnn_best_c.keras")
trainer.aggregate_round()

# RONDA 2
print("=== RONDA 2 ===")
# Clientes descargan modelo mejorado de ronda 1, entrenan...
trainer.register_client_update("client_a", "client_a_retrained_r2.keras")
trainer.register_client_update("client_b", "client_b_retrained_r2.keras")
trainer.register_client_update("client_c", "client_c_retrained_r2.keras")
trainer.aggregate_round()

# Guardar resumen
trainer.save_training_summary("fl_training_summary.json")
```

**Flujo:**
```
Ronda 1:
  Clientes entrenan localmente con su data
  → Envían modelos entrenados
  → Servidor agrega (FedAvg)
  → Modelo global v2

Ronda 2:
  Clientes descargan modelo global v2
  → Continúan entrenamiento local
  → Envían modelos mejorados
  → Servidor agrega
  → Modelo global v3
  
... (repetir N rondas)
```

---

## Exportar Arquitectura

### Generar Código Python Puro

```python
from federated_learning import ModelArchitectureExporter

exporter = ModelArchitectureExporter()

# Exportar arquitectura OCR a código Python
exporter.export_keras_architecture(
    "models/ocr_crnn_best.keras",
    "ocr_model_architecture.py"
)

# Exportar configuración a JSON
exporter.export_model_config(
    "models/ocr_crnn_best.keras",
    "ocr_model_config.json"
)
```

**Archivo generado (`ocr_model_architecture.py`):**
```python
def build_ocr_crnn_model(input_shape=(48, 192, 1), num_classes=38):
    """Construye arquitectura OCR CRNN+CTC."""
    inputs = tf.keras.Input(shape=input_shape)
    x = inputs
    
    # === CNN Feature Extraction ===
    x = tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    # ... más capas ...
    
    # === RNN Sequence Modeling ===
    x = tf.keras.layers.Reshape((-1, x.shape[-1]))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)
    
    # === Character Classification ===
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    
    model = tf.keras.Model(inputs, outputs)
    return model

# Uso:
# model = build_ocr_crnn_model()
# model.load_weights('weights.h5')
```

**Ventajas:**
- Reproducible sin archivos binarios
- Fácil de adaptar
- Documentación clara de arquitectura

---

## Casos de Uso Reales

### Caso 1: Múltiples Municipios (OCR)

```python
# Municipios: Madrid, Barcelona, Valencia

modelos = {
    "madrid": "models/ocr_madrid_entrenado.keras",
    "barcelona": "models/ocr_barcelona_entrenado.keras",
    "valencia": "models/ocr_valencia_entrenado.keras"
}

from federated_learning import FederatedModelAggregator

agg = FederatedModelAggregator()
for city, path in modelos.items():
    agg.register_client_model(city, path)

agg.aggregate_mean()
agg.save_aggregated_model_keras("ocr_españa_nacional.keras")
```

**Beneficio:** Modelo nacional que funciona bien en todas las regiones

---

### Caso 2: Placas Normales + Lejanas (Detectores)

```python
from federated_learning import DetectorEnsemble

# Usar ambos detectores
detectors = [
    "models/plate_detector_best.pt",        # Cercanas
    "models/plate_detector_smallfar_best.pt" # Lejanas
]

ensemble = DetectorEnsemble(detectors)
ensemble.set_weights([0.6, 0.4])

# En streaming:
for frame in video_stream:
    detections = ensemble.predict(frame)
    for det in detections:
        cv2.rectangle(frame, (det['x1'], det['y1']), (det['x2'], det['y2']), (0,255,0), 2)
    cv2.imshow("Detecciones", frame)
```

---

### Caso 3: OCR Robusto para Streaming

```python
from federated_learning import OCREnsemble

# 3 modelos OCR con voting temporal
ocr = OCREnsemble([
    "models/ocr_crnn_best.keras",
    "models/ocr_crnn_best_a_lr3e4_w2.keras",
    "models/ocr_crnn_best_b_lr2e4_w3.keras"
], voting_window=5)

# En cada frame del stream
for frame in video_stream:
    # Detectar placa
    detections = detector.detect(frame)
    
    for det in detections:
        # Recortar placa
        plate_crop = frame[int(det['y1']):int(det['y2']), 
                           int(det['x1']):int(det['x2'])]
        
        # OCR con consensus
        texto, conf = ocr.predict(plate_crop)
        
        if conf > 0.7:
            print(f"Placa: {texto} (confianza: {conf:.2%})")
            guardar_evidencia(frame, det, texto)
```

---

## Troubleshooting

### Error: TensorFlow no instalado

```bash
pip install tensorflow
# o para CPU only:
pip install tensorflow-cpu
```

### Error: Ultralytics YOLO no encontrado

```bash
pip install ultralytics
# Descargar modelo:
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

### Modelos diferentes tamaños

```
Error: Pesos con tamaños inconsistentes
```

**Solución:** Todos los modelos deben tener la MISMA arquitectura
```python
# ✗ Incorrecto:
agg.register_client_model("a", "ocr_small.keras")  # 5M params
agg.register_client_model("b", "ocr_large.keras")  # 50M params

# ✓ Correcto:
agg.register_client_model("a", "ocr_crnn_best_a.keras")  # Mismo modelo, diferente entrenamiento
agg.register_client_model("b", "ocr_crnn_best_b.keras")  # Mismo modelo, diferente entrenamiento
```

### Memory Out (OOM)

```python
# Usar batches más pequeños
ensemble.predict(image, batch_size=1)

# O descargar uno de los modelos temporalmente
agg.client_models = {k: v for k, v in agg.client_models.items() if k != "big_model"}
```

### Pesos muy lentos

```python
# Para agregación ponderada, normalizar manualmente
import numpy as np

sizes = {"a": 1500, "b": 1200, "c": 1800}
total = sum(sizes.values())
normalized = {k: v/total for k, v in sizes.items()}

agg.aggregate_weighted(normalized)
```

---

## Referencia Rápida - Cheat Sheet

```python
# AGREGACIÓN
from federated_learning import FederatedModelAggregator
agg = FederatedModelAggregator()
agg.register_client_model("a", "model_a.keras")
agg.register_client_model("b", "model_b.keras")
agg.aggregate_mean()
agg.save_aggregated_model_keras("global.keras")

# ENSEMBLE DETECTOR
from federated_learning import DetectorEnsemble
ens = DetectorEnsemble(["model1.pt", "model2.pt"])
ens.set_weights([0.7, 0.3])
detections = ens.predict(image)

# ENSEMBLE OCR
from federated_learning import OCREnsemble
ocr = OCREnsemble(["ocr1.keras", "ocr2.keras"])
texto, conf = ocr.predict(plate_image)

# ENTRENAMIENTO FEDERADO
from federated_learning import FederatedLearningTrainer
trainer = FederatedLearningTrainer("global_model.keras")
trainer.register_client_update("a", "updated_a.keras")
trainer.aggregate_round()

# EXPORTAR
from federated_learning import ModelArchitectureExporter
exp = ModelArchitectureExporter()
exp.export_keras_architecture("model.keras", "arch.py")
```

---

## Ejemplos Completos

Ejecutar ejemplos prácticos:

```bash
# Ejemplo 1: Agregación FedAvg
python federated_learning_examples.py 1

# Ejemplo 2: Agregación Ponderada
python federated_learning_examples.py 2

# Ejemplo 3: Ensemble Detectores
python federated_learning_examples.py 3

# Ejemplo 4: Ensemble OCR
python federated_learning_examples.py 4

# Ejemplo 5: Entrenamiento Federado
python federated_learning_examples.py 5

# Ejemplo 6: Exportar Arquitectura
python federated_learning_examples.py 6

# Ejemplo 7: Crear Configuración
python federated_learning_examples.py 7

# Flujo Completo
python federated_learning_examples.py complete
```

---

## Links Útiles

- [Federated Learning Paper (McMahan et al.)](https://arxiv.org/abs/1602.05629)
- [TensorFlow Federated](https://www.tensorflow.org/federated)
- [Ultralytics YOLO](https://docs.ultralytics.com/)
- [Keras Documentation](https://keras.io/)

---

**Versión:** 2.0 (Abril 2026)  
**Compatible con:** Python 3.8+, TensorFlow 2.10+, PyTorch 1.9+
