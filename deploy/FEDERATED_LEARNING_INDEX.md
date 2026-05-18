# 🚀 Federación de Modelos - Índice Completo

## 📋 Archivos Creados

Dentro de `deploy/`, se han agregado **4 archivos nuevos** para federación de modelos:

### 1. **federated_learning.py** (2100+ líneas)
**Módulo principal de federación**

Contiene 5 clases principales:

```
FederatedModelAggregator
├── Agregación FedAvg (promedio simple)
├── Agregación ponderada (por dataset size)
├── Carga de modelos Keras y PyTorch
├── Guardado de modelos agregados
└── Historial de agregaciones

DetectorEnsemble
├── Ensemble de detectores YOLO
├── Voto mayoritario + promedio confianza
├── NMS para fusión de detecciones
└── Predicción en paralelo

OCREnsemble
├── Ensemble de modelos OCR
├── Voting temporal (consenso entre frames)
├── Decodificación CTC
└── Estabilidad de predicciones

FederatedLearningTrainer
├── Orquestación multi-ronda
├── Agregación entre rondas
├── Registro de actualizaciones
└── Guardado de resumen

ModelArchitectureExporter
├── Exportar arquitectura a Python puro
├── Generar código sin dependencias
├── Exportar configuración a JSON
└── Documentación de modelos
```

**Usar:**
```python
from federated_learning import (
    FederatedModelAggregator,
    DetectorEnsemble,
    OCREnsemble,
    FederatedLearningTrainer,
    ModelArchitectureExporter
)
```

---

### 2. **federated_learning_examples.py** (650+ líneas)
**7 Casos de uso prácticos + workflow completo**

Cada función es un caso de uso independiente:

```
use_case_1_fedavg_aggregation()         # Promedio simple de 3 modelos OCR
use_case_2_weighted_aggregation()       # Promedio ponderado (1500:1200:1800)
use_case_3_detector_ensemble()          # Ensemble YOLO (normal + small/far)
use_case_4_ocr_ensemble()               # Ensemble OCR con voting temporal
use_case_5_federated_training()         # Multi-ronda federada
use_case_6_export_architecture()        # Exportar OCR a Python puro
use_case_7_create_federated_config()    # Generar JSON de configuración
complete_workflow()                     # Todas juntas
```

**Ejecutar:**
```bash
python federated_learning_examples.py 1      # Caso 1
python federated_learning_examples.py complete # Todos
```

---

### 3. **FEDERATED_LEARNING_QUICK_REFERENCE.md** (400+ líneas)
**Guía de referencia rápida con snippets listos para copiar**

Secciones:
- ✅ Introducción a federated learning
- ✅ Instalación y verificación
- ✅ Agregación FedAvg (código ejemplo)
- ✅ Agregación Ponderada (código ejemplo)
- ✅ Ensemble de Detectores (código ejemplo)
- ✅ Ensemble de OCR (código ejemplo)
- ✅ Entrenamiento Federado (código ejemplo)
- ✅ Exportar Arquitectura (código ejemplo)
- ✅ 3 Casos de uso reales (municipios, placas, streaming)
- ✅ Troubleshooting (8 problemas + soluciones)
- ✅ Cheat Sheet (referencia rápida)

**Está diseñado para:**
- Copiar/pegar snippets directo en tu código
- Encontrar rápido cómo hacer cada cosa
- Resolver errores comunes

---

## 🎯 Cómo Usarlo - Flujo Rápido

### **Opción A: Solo Agregación (5 min)**

```python
from federated_learning import FederatedModelAggregator

# 1. Crear agregador
agg = FederatedModelAggregator(framework="keras")

# 2. Registrar modelos
agg.register_client_model("client_a", "models/ocr_crnn_best_a_lr3e4_w2.keras")
agg.register_client_model("client_b", "models/ocr_crnn_best_b_lr2e4_w3.keras")
agg.register_client_model("client_c", "models/ocr_crnn_best_c_lr4e4_w2.keras")

# 3. Agregar
agg.aggregate_mean()

# 4. Guardar
agg.save_aggregated_model_keras("federated_ocr_global.keras")
```

**Resultado:** `federated_ocr_global.keras` (modelo global con pesos promediados)

---

### **Opción B: Ensemble de Detectores (5 min)**

```python
from federated_learning import DetectorEnsemble

# 1. Cargar
ensemble = DetectorEnsemble([
    "models/plate_detector_best.pt",
    "models/plate_detector_smallfar_best.pt"
])

# 2. Configurar pesos
ensemble.set_weights([0.7, 0.3])

# 3. Predecir
detections = ensemble.predict(image, conf_threshold=0.5)

# 4. Usar resultados
for det in detections:
    print(f"Placa detectada: ({det['x1']}, {det['y1']}) - ({det['x2']}, {det['y2']})")
```

---

### **Opción C: Ensemble de OCR (5 min)**

```python
from federated_learning import OCREnsemble

# 1. Cargar
ocr = OCREnsemble([
    "models/ocr_crnn_best.keras",
    "models/ocr_crnn_best_a_lr3e4_w2.keras",
    "models/ocr_crnn_best_b_lr2e4_w3.keras"
], voting_window=3)

# 2. En cada frame
for frame_placa in video:
    texto, confianza = ocr.predict(frame_placa)
    print(f"Placa: {texto} ({confianza:.2%})")
```

---

### **Opción D: Exportar Arquitectura (3 min)**

```python
from federated_learning import ModelArchitectureExporter

exp = ModelArchitectureExporter()
exp.export_keras_architecture("models/ocr_crnn_best.keras", "ocr_arch.py")
exp.export_model_config("models/ocr_crnn_best.keras", "ocr_config.json")
```

**Resultado:**
- `ocr_arch.py` → Función `build_ocr_crnn_model()` en Python puro
- `ocr_config.json` → Configuración de capas

---

## 📊 Comparativa: Métodos de Agregación

| Método | Cuándo Usar | Ventaja | Desventaja |
|--------|-----------|---------|-----------|
| **FedAvg** | Clientes con datos similares | Simple, rápido | Ignora desequilibrio |
| **Weighted** | Clientes con datos distintos | Proporcional a data | Requiere conocer sizes |
| **Ensemble (Detector)** | Mejorar detección | Robustez + confianza | 2x latencia |
| **Ensemble (OCR)** | Mejorar OCR en video | Estabilidad temporal | Requires GPU |
| **FL Multi-ronda** | Mejora progresiva | Iterativo, mejora continua | Lento, complejo |

---

## 🔧 Parámetros Clave

### **FederatedModelAggregator**
```python
FederatedModelAggregator(
    framework="keras",      # "keras", "pytorch", o "mixed"
    verbose=True            # Imprimir logs detallados
)
```

### **DetectorEnsemble**
```python
DetectorEnsemble(
    model_paths=[...],      # Lista de rutas .pt
    framework="pytorch",    # "pytorch", "tensorflow"
    device="cpu"            # "cpu", "cuda", "mps"
)
ensemble.set_weights([0.7, 0.3])  # Pesos por modelo
```

### **OCREnsemble**
```python
OCREnsemble(
    model_paths=[...],      # Lista de rutas .keras
    voting_window=3         # Frames para consensus
)
```

### **FederatedLearningTrainer**
```python
FederatedLearningTrainer(
    global_model_path="...",  # Modelo inicial
    framework="keras"         # "keras", "pytorch"
)
trainer.aggregate_round()     # Una ronda = agregación
```

---

## 📁 Estructura de Carpetas Recomendada

```
deploy/
├── federated_learning.py                    # Módulo (NO EDITAR)
├── federated_learning_examples.py           # Ejemplos (REFERENCIA)
├── FEDERATED_LEARNING_QUICK_REFERENCE.md   # Guía (CONSULTA)
├── inference_jetson.py                      # Pipeline principal
├── ocr_inference.py                         # OCR modular
├── runtime_config.py                        # Configuración
└── user_scripts/                            # Tus scripts personalizados
    ├── my_federated_pipeline.py             # Tu código
    ├── my_ensemble_detector.py              # Tu código
    └── my_fl_training.py                    # Tu código

../models/
├── plate_detector_best.pt
├── plate_detector_smallfar_best.pt
├── ocr_crnn_best.keras
├── ocr_crnn_best_a_lr3e4_w2.keras
├── ocr_crnn_best_b_lr2e4_w3.keras
├── ocr_crnn_best_c_lr4e4_w2.keras
└── optimized/
    ├── detector_int8.tflite
    └── ocr_crnn_ctc_int8.tflite
```

---

## ⚡ Flujo Completo (Ejemplo Real)

**Municipios: Madrid, Barcelona, Valencia**

```
SEMANA 1: Entrenamiento Local
┌─────────────┬────────────┬──────────────┐
│   Madrid    │ Barcelona  │   Valencia   │
│  Entrena    │  Entrena   │   Entrena    │
│ OCR local   │ OCR local  │  OCR local   │
│ 2000 datos  │ 1500 datos │ 1800 datos   │
└─────────────┴────────────┴──────────────┘
        ↓         ↓            ↓
     Enviar modelos entrenados
        ↓         ↓            ↓
    AGREGACIÓN FEDERADA (FedAvg Ponderado)
        ↓
    Modelo Global Mejorado v1
        ↓         ↓            ↓
    Descargar   Descargar   Descargar

SEMANA 2: Entrenamiento Continuo (mismo proceso)
    Modelo Global Mejorado v2
    ...
```

**Código:**

```python
from federated_learning import FederatedLearningTrainer

trainer = FederatedLearningTrainer("models/ocr_crnn_best.keras")

# Ronda 1
for week in range(1, 5):
    print(f"\n=== SEMANA {week} ===")
    
    # Cada municipio entrena (simulado)
    trainer.register_client_update("madrid", f"output/madrid_w{week}.keras")
    trainer.register_client_update("barcelona", f"output/barcelona_w{week}.keras")
    trainer.register_client_update("valencia", f"output/valencia_w{week}.keras")
    
    # Agregación
    trainer.aggregate_round()
    
    print(f"✓ Modelo global v{week} listo")

trainer.save_training_summary("fl_españa_resumen.json")
```

---

## 🐛 Troubleshooting Rápido

**P: ¿Pedo agregar modelos de frameworks diferentes?**  
R: No fácilmente. Usa modelos Keras con Keras, PyTorch con PyTorch.

**P: ¿Funciona en Jetson Nano?**  
R: Sí, pero sin GPU es lento. Usa modelos TFLite para inferencia rápida.

**P: ¿Cuánto tiempo toma agregación?**  
R: ~30s-2min dependiendo de tamaño modelo y CPU.

**P: ¿Necesito TensorFlow o PyTorch?**  
R: Sí, al menos uno. TensorFlow para OCR, PyTorch para detectores.

---

## 📚 Documentación Relacionada

- **[README.md](README.md)** → Visión general del pipeline
- **[FEDERATED_LEARNING_QUICK_REFERENCE.md](FEDERATED_LEARNING_QUICK_REFERENCE.md)** → Snippets y ejemplos
- **[federated_learning.py](federated_learning.py)** → Código fuente (docstrings)
- **[federated_learning_examples.py](federated_learning_examples.py)** → Casos de uso

---

## 🚀 Próximos Pasos

1. **Revisar ejemplos:** `python federated_learning_examples.py complete`
2. **Consultar guía:** Abrir [FEDERATED_LEARNING_QUICK_REFERENCE.md](FEDERATED_LEARNING_QUICK_REFERENCE.md)
3. **Copiar snippet:** Seleccionar el que necesites y adaptarlo
4. **Ejecutar:** `python mi_script_federado.py`

---

## 📞 Soporte

Si hay errores, revisar:

1. **¿Modelos existen?** → Verificar rutas en `MODELS_DIR`
2. **¿Dependencias?** → `pip install tensorflow ultralytics opencv-python`
3. **¿Tamaños iguales?** → Todos modelos deben tener misma arquitectura
4. **¿RAM suficiente?** → Agregación requiere ~3x memoria del modelo

---

**Versión:** 2.0 (Federación de Modelos)  
**Fecha:** Abril 30, 2026  
**Autor:** GitHub Copilot
