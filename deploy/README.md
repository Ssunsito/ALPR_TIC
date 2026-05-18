# 🚗 Deploy - Pipeline de Detección de Placas en Tiempo Real

Carpeta de despliegue optimizada para **Jetson Nano J1010** con 2GB RAM. Contiene todos los módulos necesarios para ejecutar detección de placas vehiculares en tiempo real con OCR confirmación e captura automática de evidencias.

---

## 📁 Contenido de la Carpeta

### 🔹 **inference_jetson.py** (2342 líneas)
**Propósito:** Pipeline principal de inferencia en tiempo real.

**Características clave:**
- Stream de video desde cámara nativa con V4L2/GStreamer
- Detección de placas con YOLO v8 (o fallback automático)
- OCR inteligente con motor plugeable (RapidOCR → PaddleOCR → Tesseract)
- **Auto-capture:** Guarda evidencia (imagen anotada) cuando OCR confirma texto con confianza ≥ 0.50
- **Deduplicación dual:** Previene saturación de disco mediante cooldown por firma (2s) y por texto (6s)
- **Pequeñas placas rescue:** Detección periódica de placas lejanas/pequeñas sin sacrificar FPS
- Optimizaciones de memoria para edge: stride adaptativo, skip de movimiento, pre-procesamiento condicional

**Dependencias:**
```
opencv-python >= 4.5
numpy
ultralytics (YOLO)
tensorflow (opcional - para CTC post-processing)
pytesseract (opcional)
paddleocr (opcional)
rapidocr-onnxruntime (opcional)
```

**Punto de entrada:**
```bash
python inference_jetson.py --video 0 --runtime-profile jetson-realtime
```

---

### 🔹 **runtime_config.py** (267 líneas)
**Propósito:** Configuración centralizada y perfiles de runtime.

**Contenido:**
- **CONFIG dict:** 200+ parámetros configurables (detección, OCR, hardware)
- **Perfiles predefinidos:** Cuatro configuraciones optimizadas
  - `jetson-stable`: Máxima estabilidad, menor FPS (~2-3)
  - `jetson-quality`: Balance estabilidad/velocidad (~4-5 FPS)
  - `jetson-production`: Producción real (~6-7 FPS)
  - `jetson-realtime`: Alto rendimiento (~8-9 FPS, tolerancia media)

**Función principal:**
```python
apply_runtime_profile(profile_name, config=CONFIG)
# Ajusta todos los parámetros según el perfil seleccionado
```

**Uso:**
```python
from runtime_config import CONFIG, apply_runtime_profile

apply_runtime_profile("jetson-realtime")
detector = YOLODetector(CONFIG["detector_model"])
```

---

### 🔹 **ocr_inference.py** (1188 líneas)
**Propósito:** Motor de OCR modular e independiente.

**Características:**
- **Clase OCRInference:** Abstracción unificada para todos los motores OCR
- **Plugeable:** Soporta RapidOCR (preferido), PaddleOCR, Tesseract, clasificador char-level
- **Fallback automático:** Si un motor falla, intenta el siguiente en la cadena
- **Voting temporal:** Consenso de OCR entre frames para estabilidad (reduce falsos positivos)
- **Variantes de placa:** Maneja 6-7 caracteres españoles, dígitos-letras visuales confusas
- **CTC post-processing:** Si TensorFlow disponible, aplica capa CTC para refinamiento

**Métodos principales:**
```python
ocr = OCRInference(config=CONFIG)

# Reconocimiento básico
texto, confianza = ocr.segment_and_recognize_with_conf(
    imagen_placa_crop, 
    es_placa_lejana=False
)

# Reconocimiento con CTC (si TensorFlow)
texto, confianza = ocr.recognize_sequence_ctc(
    imagen_placa_crop, 
    topk=3
)
```

---

### 🔹 **inference_jetson_trt.py** (Variante TensorRT)
**Propósito:** Aceleración con TensorRT (desactivado actualmente por limitaciones de almacenamiento).

**Estado:** Preparado pero NO activo en producción.

**Futuro:** Se activará cuando Jetson tenga espacio para almacenar modelos TensorRT compilados (~500MB).

---

## 🚀 Inicio Rápido

### Requisitos Previos
1. **Jetson Nano J1010** con JetPack 4.6.1+ (o cualquier Linux con Python 3.8+)
2. **Dependencias instaladas:**
   ```bash
   pip install opencv-python numpy ultralytics
   pip install rapidocr-onnxruntime  # Recomendado
   # Opcional:
   pip install tensorflow paddleocr pytesseract
   ```

3. **Modelos descargados** en carpeta `../models/`:
   - `plate_detector_best.pt` (YOLO detección de placas)
   - `optimized/ocr_crnn_ctc_int8.tflite` (OCR TFLite, opcional)

4. **Cámara conectada** (default: `/dev/video0` en Linux, índice 0 en Windows)

---

## 📋 Ejemplos de Uso

### 1. **Ejecución Estándar en Jetson**
```bash
cd deploy/
python inference_jetson.py \
    --video 0 \
    --runtime-profile jetson-realtime \
    --verbose
```

**Salida esperada:**
```
[INFO] Camera initialized: /dev/video0 @ 30fps
[INFO] Detector loaded: models/plate_detector_best.pt
[INFO] OCR engine: RapidOCR (RapidOCR fallback available)
[INFO] Runtime profile: jetson-realtime (8-9 FPS target)
[INFO] Starting video stream...
Detected: AAB-1234 (conf=0.92) → Plate capture saved: plate_captures/AAB-1234_20260426_143022.jpg
```

---

### 2. **Ejecución con Archivo de Video**
```bash
python inference_jetson.py \
    --video camera_test.mp4 \
    --runtime-profile jetson-production
```

---

### 3. **Cambiar Perfil de Runtime**
```bash
# Máxima estabilidad (laboratorio, pruebas)
python inference_jetson.py --video 0 --runtime-profile jetson-stable

# Producción real (strada)
python inference_jetson.py --video 0 --runtime-profile jetson-production

# Alto rendimiento (tolerancia media de errores)
python inference_jetson.py --video 0 --runtime-profile jetson-realtime
```

---

### 4. **Interfaz Programática**
```python
from deploy.inference_jetson import VideoStreamProcessor
from deploy.runtime_config import CONFIG, apply_runtime_profile

# Cargar perfil
apply_runtime_profile("jetson-quality")

# Crear procesador
processor = VideoStreamProcessor(
    camera_index=0,
    config=CONFIG,
    verbose=True
)

# Ejecutar stream (bloqueante, presionar 'q' para salir)
processor.run_video_stream()
```

---

## ⚙️ Parámetros de Configuración

Todos los parámetros están en `runtime_config.py\:CONFIG`. Aquí los más importantes:

| Parámetro | Rango | Default | Descripción |
|-----------|-------|---------|-------------|
| `confidence_threshold` | 0.1 - 1.0 | 0.8 | Confianza mínima de detección |
| `iou_threshold` | 0.0 - 1.0 | 0.5 | NMS IOU para eliminar duplicados |
| `ocr_engine` | "rapidocr-lpr", "paddle", "tesseract" | "rapidocr-lpr" | Motor OCR preferido |
| `camera_plate_capture_enabled` | true/false | true | Guardar evidencia de placas |
| `camera_plate_capture_cooldown_frame_secs` | 1 - 30 | 2 | Cooldown por firma (segundos) |
| `camera_plate_capture_cooldown_text_secs` | 1 - 30 | 6 | Cooldown por texto (segundos) |
| `enable_small_plate_rescue` | true/false | true | Detección periódica de pequeñas |
| `camera_small_rescue_every_n_detections` | 1 - 20 | 4 | Cada N detecciones, rescatar pequeñas |

**Modificar en tiempo de ejecución:**
```python
CONFIG["confidence_threshold"] = 0.75
CONFIG["camera_plate_capture_cooldown_text_secs"] = 10
apply_runtime_profile("jetson-production")  # Reaplica el perfil
```

---

## 📸 Auto-Capture de Evidencia

### Cómo Funciona
1. **Detección:** YOLO detecta placa en frame (conf ≥ 0.8)
2. **OCR:** Lee texto con motor configurado (conf ≥ 0.50)
3. **Guardado:** Si ambos pasan, guarda imagen anotada en `plate_captures/`
4. **Deduplicación:**
   - No guarda nuevamente la MISMA placa dentro de 2s (por firma)
   - No guarda el MISMO TEXTO dentro de 6s (por OCR)

### Carpeta de Salida
```
plate_captures/
    AAB-1234_20260426_143022.jpg  (imagen + bounding box + OCR)
    AAB-1234_20260426_143024.jpg  (OCR ≠ → nueva captura)
    ABC-5678_20260426_144010.jpg
    ...
```

### Deshabilitación
```python
CONFIG["camera_plate_capture_enabled"] = False
```

---

## 🎯 Pequeñas Placas - Rescue Strategy

### Problema
Placas lejanas/pequeñas no detectadas por YOLO en stride normal (necesitaría 2-3MB extra para tiling).

### Solución
**Rescate periódico:** Cada N detecciones exitosas, ejecutar detector con tiling en imagen completa.

### Flujo
```
Frame 1: YOLO stride normal → sin detección
Frame 2: YOLO stride normal → sin detección  
Frame 3: YOLO stride normal → sin detección
Frame 4: YOLO stride normal → PLACA DETECTADA ✓
Frame 5: YOLO stride normal → sin detección
Frame 6: YOLO con TILING + upscale → Rescata placa lejana ✓
```

### Parámetros
```python
"enable_small_plate_rescue": true,           # Habilitar
"camera_small_rescue_every_n_detections": 4, # Cada 4 detecciones
"small_plate_rescue_upscale": 1.8,          # 1.8x amplificación
"small_plate_rescue_tiling": true,          # Grid 2x2 con overlap
```

### Deshabilitación
```python
CONFIG["enable_small_plate_rescue"] = False
```

---

## 🔍 Troubleshooting

| Problema | Causa | Solución |
|----------|-------|----------|
| **FPS muy bajo (~1-2)** | Stride muy agresivo, no hay CPU suficiente | Usar `jetson-realtime` profile, desactivar `enable_small_plate_rescue` |
| **OCR instable (cambia cada frame)** | Falta voting temporal | Aumentar `ocr_voting_window_size` en CONFIG |
| **Cámara no inicializa** | Puerto incorrecto, sin permisos | `ls -la /dev/video*`, usar `--video 0` o `/dev/video0` |
| **Out of memory (OOM)** | Cache de frames acumulado | Reducir `fps_buffer_size` de 30 a 10 |
| **TensorFlow no encontrado pero requerido** | Falta instalación TF | `pip install tensorflow-lite` o usar engines sin TF (RapidOCR) |
| **Muy pocas placas detectadas** | Threshold demasiado alto | Reducir `confidence_threshold` de 0.8 a 0.6 |
| **Muchos falsos positivos** | Threshold muy bajo | Aumentar `confidence_threshold` o refinar modelo |
| **OCR lee mal (AAB → AA8, etc.)** | Confusión visual (0→O, 1→I) | Validada en `ocr_inference.py` → revisar `DIGIT_TO_LETTER_VISUAL` |

---

## 📊 Monitoreo y Debugging

### Ejecutar con Verbose
```bash
python inference_jetson.py --video 0 --verbose
```

**Salidas:**
```
[DEBUG] Frame 42 → 3 detections
[DEBUG] Detection 0: confidence=0.92, area=2340px²
[DEBUG] OCR result: AAB-1234, confidence=0.88, voting_score=0.92
[DEBUG] Capture saved: plate_captures/AAB-1234_20260426_143022.jpg (2.3MB)
```

### Monitoreo de Recursos (Jetson)
```bash
# En otra terminal
watch -n 1 'nvidia-smi; echo "---"; ps aux | grep python'
```

**Métrica objetivo:** RAM < 1.5GB, GPU < 80%, CPU < 80% (Jetson realtime)

---

## 🔧 Extensiones y Personalizaciones

### 1. **Agregar Motor OCR Nuevo**
Editar `ocr_inference.py`, método `segment_and_recognize_with_conf()`:
```python
# Agregar engine antes del fallback final
if engine_name == "mi_ocr_nuevo":
    texto, conf = self._ocr_mi_engine_nuevo(crop_img, params)
    if texto and conf >= self.config["ocr_rapid_min_conf"]:
        return texto, conf
```

### 2. **Cambiar Modelo Detector**
```python
CONFIG["detector_model"] = "models/plate_detector_v2.pt"
CONFIG["detector_fallback_model"] = "models/plate_detector_v1.pt"
apply_runtime_profile("jetson-production")
```

### 3. **Post-procesamiento de OCR Personalizado**
Editar `ocr_inference.py`, método `recognize_sequence_ctc()`:
```python
# Agregar validación de negocio (ej: solo placas españolas)
if not re.match(r'^[A-Z]{0,3}\d{0,4}$', texto):
    return None, 0.0
```

---

## 📈 Roadmap

| Fase | Descripción | Estado |
|------|-------------|--------|
| 1. Detección base | YOLO + OCR simple | ✅ Activo |
| 2. Consenso OCR | Voting temporal | ✅ Activo |
| 3. Pequeñas placas | Rescue periódico | ✅ Activo |
| 4. Auto-capture | Guardado evidencia | ✅ Activo |
| 5. TensorRT | Aceleración GPU | 🟡 Preparado (sin almacenamiento) |
| 6. Modelos custom | Transfer learning | 📋 Futuro |

---

## 📝 Logs y Evidencia

- **Logs:** Se imprimen en stdout (capturar con `script /tmp/session.log` en Linux)
- **Evidencia capturada:** `plate_captures/*.jpg` (imagen + boxes + OCR)
- **Modelos:** `../models/` (YOLO, OCR, TFLite)
- **Configuración aplicada:** Visible con `--verbose` al inicio

---

## 🆘 Contacto / Reporte de Bugs

Si encuentras problemas:
1. Verifica requisitos previos (cámara, modelos, dependencias)
2. Ejecuta con `--verbose` y captura logs
3. Revisa sección **Troubleshooting** arriba
4. Consulta `../docs/IMPLEMENTACION_CON_ERRORES.md` para problemas conocidos

---

## 📄 Licencia y Referencias

- **YOLO v8:** Ultralytics → https://github.com/ultralytics/ultralytics
- **RapidOCR:** ONNX Runtime → https://github.com/RapidAI/RapidOCR
- **PaddleOCR:** PaddlePaddle → https://github.com/PaddlePaddle/PaddleOCR
- **Documentación adicional:** Ver `../docs/` (IMPLEMENTACION_FINAL_Y_ROADMAP.md, IMPLEMENTACION_CON_ERRORES.md)

---

**Última actualización:** 26 de Abril de 2026  
**Versión deploy:** 2.0 (Modularizado, optimizado para Jetson Nano)
