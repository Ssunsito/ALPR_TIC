# MAPA DE CORRESPONDENCIA: ARCHIVOS DEL PROYECTO → CAPÍTULOS DE TESIS
## Guía Visual para Entender dónde Escribir sobre Cada Componente

---

## VISTA GENERAL (Estructura Jerárquica)

```
TESIS
│
├─ CAPÍTULO 2: MARCO TEÓRICO
│  └─ Incluye: Conceptos, estado del arte, problemas conocidos
│     (No depende de archivos específicos - investigación)
│
├─ CAPÍTULO 3: METODOLOGÍA (CENTRALIZADO)
│  │
│  ├─ 3.1 Arquitectura General
│  │  └─ ARCHIVOS:
│  │     • deploy/inference_jetson.py → flujo detector → OCR
│  │     • deploy/inference_jetson_pc.py → versión PC (transferencia)
│  │     • DIAGRAMA: Flujo completo (crear visual)
│  │
│  ├─ 3.2 Módulo Detector YOLO
│  │  └─ ARCHIVOS:
│  │     • deploy/inference_jetson.py (línea ~200: _load_detector)
│  │     • deploy/runtime_config.py (línea ~10: modelo YOLO)
│  │     • models/plate_detector_best.pt (el modelo)
│  │     DATOS: detecciones 100%, confianza 0.915
│  │
│  ├─ 3.3 CONTRIBUCIÓN 1: ROI Bias System ⭐⭐⭐ SECCIÓN ESTRELLA
│  │  └─ ARCHIVOS:
│  │     • deploy/runtime_config.py
│  │       ✓ línea ~92: "ocr_roi_y_bias_close": 0.30
│  │       ✓ línea ~93: "ocr_roi_y_bias_medium": 0.20
│  │       ✓ línea ~94: "ocr_roi_y_bias_small": 0.12
│  │       ✓ comentario: "Aumentado bias vertical para OCR"
│  │     • deploy/inference_jetson.py (función extract_roi_with_bias)
│  │     • ACCIONES_REALIZADAS_v2.md (línea ~20: problema diagnosticado)
│  │     DATOS: 0% → 99% validez, calibración empírica
│  │
│  ├─ 3.4 Módulo OCR (Dual Backend)
│  │  └─ ARCHIVOS:
│  │     • deploy/ocr_inference.py (backends RapidOCR + Tesseract)
│  │     • deploy/runtime_config.py (línea ~14: modelo OCR TFLite)
│  │     • models/optimized/ocr_crnn_ctc_int8.tflite (modelo cuantizado)
│  │     DATOS: latencia OCR ~3500ms (bottleneck identificado)
│  │
│  ├─ 3.5 CONTRIBUCIÓN 2: Postprocessing + Retry Logic ⭐⭐⭐
│  │  └─ ARCHIVOS:
│  │     • deploy/ocr_postprocessor.py (ARCHIVO CLAVE)
│  │       ✓ validate_plate_format() - línea ~25
│  │       ✓ postprocess_ocr_text() - línea ~60
│  │       ✓ adjust_roi_for_invalid_ocr() - línea ~100
│  │       ✓ detect_header_anomaly() - línea ~140
│  │     • deploy/inference_jetson.py (integración retry logic)
│  │     DATOS: 0% → 99% formato válido, 1 error genuino
│  │
│  ├─ 3.6 Sistema de Configuración Centralizada
│  │  └─ ARCHIVOS:
│  │     • deploy/runtime_config.py (ARCHIVO MAESTRO)
│  │       ✓ CONFIG diccionario (todas las opciones)
│  │       ✓ apply_runtime_profile() función
│  │       ✓ perfiles: jetson-stable, jetson-quality, etc.
│  │
│  └─ 3.7 Sistema de Evaluación y Métricas
│     └─ ARCHIVOS:
│        • reportes/resultados_metricas_20260512/generar_reporte_metricas_v2_fast.py (SCRIPT PRINCIPAL)
│          ✓ run_image_analyzer() - procesa 100 imágenes
│          ✓ _char_confusion_matrix() - matriz confusión
│          ✓ generate_plots() - 5 gráficas
│          ✓ build_docx() - reporte DOCX
│        • dataset_alpr/images/val/ (100 imágenes input)
│
├─ CAPÍTULO 4: RESULTADOS Y ANÁLISIS
│  │
│  ├─ 4.1 Tabla Resumen (100 Imágenes)
│  │  └─ DATOS FUENTE:
│  │     • reportes/.../metricas_resumen.json
│  │     • reportes/.../Informe_metricas_y_graficas_v3_100.docx
│  │     NÚMEROS: 100%, 99%, 33%, 7264ms, 0.14 FPS
│  │
│  ├─ 4.2 Análisis de Errores
│  │  └─ DATOS FUENTE:
│  │     • reportes/.../tablas/metricas_detalle_imagen.csv
│  │     • reportes/.../tablas/metricas_detalle_imagen_valid.csv
│  │     (100 filas, columnas: filename, gt, pred, exact_match, cer)
│  │
│  ├─ 4.3 Matriz de Confusión
│  │  └─ VISUALIZACIÓN:
│  │     • reportes/.../graficas/06_confusion_char_heatmap.png
│  │
│  ├─ 4.4 Latencia y FPS
│  │  └─ VISUALIZACIONES:
│  │     • reportes/.../graficas/04_boxplot_fps_latencia.png
│  │     • reportes/.../graficas/05_area_latency_timeline.png
│  │
│  └─ 4.5 Limitaciones + Transición FL
│     └─ ANÁLISIS: (derivado de resultados, no de archivo específico)
│
└─ CAPÍTULO 5: PROPUESTA FEDERATED LEARNING
   └─ ARCHIVOS (para adelanto/futuro):
      • deploy/federated_learning.py (si existe, estructura FL)
      • deploy/federated_learning_examples.py (ejemplos)

```

---

## CORRESPONDENCIA ARCHIVO ↔ SECCIÓN DE TESIS (Tabla Detallada)

| Archivo | Ubicación | Sección de Tesis | Propósito | Líneas Clave |
|---------|-----------|------------------|----------|-------------|
| **runtime_config.py** | `deploy/` | 3.3, 3.6 | Parámetros ROI bias, configuración | 92-94, 200+ |
| **ocr_postprocessor.py** | `deploy/` | 3.5 | Validación formato, anomalía detection | 25, 60, 100, 140 |
| **inference_jetson.py** | `deploy/` | 3.1, 3.2, 3.5 | Orquestador principal, integración | Clase PipelineJetson |
| **ocr_inference.py** | `deploy/` | 3.4 | Motor OCR dual | Backend selection |
| **inference_jetson_pc.py** | `deploy/` | 3.1 | Entrypoint PC (validación) | Reuso PipelineJetson |
| **generar_reporte_metricas_v2_fast.py** | `reportes/.../` | 3.7, 4.1-4.4 | Generador métricas, gráficas | run_image_analyzer(), generate_plots() |
| **dataset_alpr/images/val/** | `dataset_alpr/` | 3.7, 4.1 | 100 imágenes validación | Dataset specification |
| **metricas_resumen.json** | `reportes/.../` | 4.1 | Números agregados (100%, 99%, 33%) | JSON con stats |
| **tablas/metricas_detalle_imagen.csv** | `reportes/.../` | 4.2 | Desglose errores por imagen | 100 filas |
| **graficas/*.png** | `reportes/.../` | 4.2-4.4 | 5 gráficas profesionales | 5 PNG files |
| **Informe_metricas_y_graficas_v3_100.docx** | `reportes/.../` | 4.1 | Reporte final con gráficas | DOCX con figuras |
| **ACCIONES_REALIZADAS_v2.md** | Raíz | 2.4, 3.3 | Diagnóstico del problema (header reading) | Línea ~20 |

---

## FLUJO DE LECTURA PARA REDACCIÓN (En qué Orden Explorar)

### PASO 1: Entender el Problema (30 minutos)
```
Leer → ACCIONES_REALIZADAS_v2.md (todo el archivo)
↓
Comprensión: ¿Por qué OCR lee "ECUADOR"?
Conclusión: Encabezado naranja incluido en bounding box
```

### PASO 2: Entender la Solución ROI Bias (1 hora)
```
Leer → deploy/runtime_config.py líneas 92-94
       (parámetros ocr_roi_y_bias_*)
↓
Buscar → deploy/inference_jetson.py función extract_roi_with_bias
         (cómo se aplica el sesgo)
↓
Comprensión: Desplazamiento vertical excluye encabezado
```

### PASO 3: Entender Postprocessing (1 hora)
```
Leer → deploy/ocr_postprocessor.py (archivo completo)
       funciones: validate_plate_format(), postprocess_ocr_text()
↓
Buscar → deploy/inference_jetson.py (línea ~500: integración)
↓
Comprensión: Validación formato + reintentos automáticos
```

### PASO 4: Entender Resultados (1 hora)
```
Abrir → reportes/resultados_metricas_20260512/
       (ver carpeta completa)
↓
Leer → metricas_resumen.json (números: 100%, 99%, 33%)
↓
Ver → tablas/metricas_detalle_imagen.csv (100 filas, desglose)
↓
Ver → graficas/*.png (5 gráficas)
       03_donut_resultados.png ← distribución
       04_boxplot_fps_latencia.png ← latencia
       06_confusion_char_heatmap.png ← errores OCR
↓
Abrir → Informe_metricas_y_graficas_v3_100.docx (contexto completo)
```

---

## COMANDO PARA EXPLORAR PROYECTO (Terminal)

```powershell
# 1. Ver estructura completa
Tree /F c:\Users\santi\OneDrive\Escritorio\Dataset\Intento2

# 2. Ver archivos más importantes
cd c:\Users\santi\OneDrive\Escritorio\Dataset\Intento2

# 3. Abrir archivos de configuración
code deploy/runtime_config.py
code deploy/ocr_postprocessor.py

# 4. Ver resultados
cd reportes/resultados_metricas_20260512
ls -la
Get-Content metricas_resumen.json | ConvertFrom-Json | Format-Table

# 5. Ver métricas en Excel/CSV
Invoke-Item tablas/metricas_detalle_imagen.csv

# 6. Ver gráficas
Invoke-Item graficas/

# 7. Ver reporte DOCX
Invoke-Item Informe_metricas_y_graficas_v3_100.docx
```

---

## PREGUNTAS A RESPONDER MIENTRAS REDACTAS (Checklist)

### Capítulo 3.3: ROI Bias System

- [ ] ¿Dónde están los valores 0.30, 0.20, 0.12? 
  → `deploy/runtime_config.py` líneas 92-94

- [ ] ¿Cómo se aplica el sesgo en el código?
  → `deploy/inference_jetson.py` función `extract_roi_with_bias()`

- [ ] ¿Cuál fue el problema original?
  → `ACCIONES_REALIZADAS_v2.md` sección "Diagnóstico del Problema"

- [ ] ¿Cuál es el impacto numérico?
  → Validez de formato: 0% → 99%
  → Exactitud: 0% → 33%

### Capítulo 3.5: Postprocessing + Retry Logic

- [ ] ¿Dónde está la validación de formato?
  → `deploy/ocr_postprocessor.py` función `validate_plate_format()`

- [ ] ¿Cuál es el regex exacto?
  → `^[A-Z]{3}-\d{3,4}$` (ver línea ~25)

- [ ] ¿Cómo funciona el retry?
  → `deploy/ocr_postprocessor.py` función `postprocess_ocr_prediction()` (línea ~60)

- [ ] ¿Cómo se detecta "ECUADOR"?
  → `detect_header_anomaly()` función

### Capítulo 4: Resultados

- [ ] ¿Cuál es la exactitud final?
  → 33% (33/100 exact matches)

- [ ] ¿Cuál es la validez de formato?
  → 99% (99/100 predicciones con formato correcto)

- [ ] ¿Cuál es la latencia?
  → 7264 ms promedio

- [ ] ¿Cuántos errores de cada tipo?
  → Detección: 0 | Formato: 1 | Carácter: 66

- [ ] ¿Cuáles son los caracteres más confundidos?
  → Ver gráfica `06_confusion_char_heatmap.png`

---

## ESTRATEGIA DE COPIAR-PEGAR (Datos Exactos de Archivo)

### Números de Línea Importantes (Para Citas)

**En `deploy/runtime_config.py`:**
```python
# Línea ~92-94: Parámetros ROI Bias
"ocr_roi_y_bias_close": 0.30,     # Placa cercana
"ocr_roi_y_bias_medium": 0.20,    # Placa media
"ocr_roi_y_bias_small": 0.12,     # Placa lejana
```

**En `deploy/ocr_postprocessor.py`:**
```python
# Línea ~25: Validación formato
pattern = r"^[A-Z]{3}-\d{3,4}$"
return bool(re.match(pattern, text.strip().upper()))

# Línea ~60: Postprocesamiento completo
def postprocess_ocr_text(text: str) -> Tuple[str, bool]:
    ...
```

---

## MATRIZ DE DECISIÓN: "¿De dónde Saco Esto?"

| Necesito Escribir Sobre | Archivo que Debo Abrir | ¿Qué Buscar? |
|-------------------------|----------------------|-------------|
| ROI Bias Values | `deploy/runtime_config.py` | `ocr_roi_y_bias_` |
| Validación Formato | `deploy/ocr_postprocessor.py` | `validate_plate_format` o regex |
| Retry Logic | `deploy/ocr_postprocessor.py` | función `postprocess_ocr_prediction` |
| Arquitectura Completa | `deploy/inference_jetson.py` | Clase `PipelineJetson` (main pipeline) |
| Números de Resultados | `reportes/.../metricas_resumen.json` | JSON con stats (100%, 99%, 33%, etc.) |
| Errores por Tipo | `reportes/.../tablas/metricas_detalle_imagen.csv` | CSV con columnas error_type |
| Matriz Confusión Visual | `reportes/.../graficas/06_confusion_char_heatmap.png` | PNG heatmap |
| Latencia Distribución | `reportes/.../graficas/04_boxplot_fps_latencia.png` | PNG boxplot |
| Problema Original | `ACCIONES_REALIZADAS_v2.md` | Sección "Diagnóstico del Problema" |

---

## LISTA DE FIGURAS A INCLUIR (Y de Dónde Vienen)

### Figuras que YA EXISTEN (solo incluir en tesis)

```markdown
1. Donut Chart - Distribución de Resultados
   Archivo: reportes/.../graficas/03_donut_resultados.png
   Sección: Capítulo 4.1
   
2. Boxplot - FPS y Latencia
   Archivo: reportes/.../graficas/04_boxplot_fps_latencia.png
   Sección: Capítulo 4.4

3. Area Chart - Timeline de Latencia
   Archivo: reportes/.../graficas/05_area_latency_timeline.png
   Sección: Capítulo 4.4

4. Heatmap - Matriz Confusión OCR
   Archivo: reportes/.../graficas/06_confusion_char_heatmap.png
   Sección: Capítulo 4.3

5. Barras - Análisis por Bloques
   Archivo: reportes/.../graficas/07_grouped_batches.png
   Sección: Capítulo 4.1
```

### Figuras que NECESITAS CREAR

```markdown
1. Diagrama Arquitectura (Flujo completo)
   Información: deploy/inference_jetson.py (clase PipelineJetson)
   Formato: Recomendado: Diagrama ASCII o Lucidchart
   Sección: Capítulo 3.1

2. ROI Antes/Después (Comparativa Visual)
   Información: Comparar ROI sin sesgo vs con sesgo 0.20
   Formato: 2 imágenes lado a lado
   Sección: Capítulo 3.3
   Nota: Usar imagen anotada del DOCX de métrica

3. Flujo de Decisión Postprocessing
   Información: deploy/ocr_postprocessor.py
   Formato: Diagrama de flujo con condicionales
   Sección: Capítulo 3.5

4. Tabla Comparativa Centralizado vs FL
   Información: Análisis (Capítulo 4.5)
   Formato: Tabla con filas (latencia, exactitud, etc.)
   Sección: Capítulo 4.5-4.6
```

---

## CÓMO CITAR ARCHIVOS EN TESIS (Formato Académico)

```markdown
### Formato 1: Referencia a Archivo del Proyecto
"Según se implementa en deploy/inference_jetson.py (línea 120-135), 
el sistema extrae la ROI con sesgo vertical..."

### Formato 2: Referencia a Configuración
"Los parámetros de sesgo se definen en runtime_config.py como:
  - ocr_roi_y_bias_close = 0.30
  - ocr_roi_y_bias_medium = 0.20
  - ocr_roi_y_bias_small = 0.12
(Véase Apéndice A)"

### Formato 3: Referencia a Resultados
"En la evaluación de 100 imágenes (dataset_alpr/images/val/), 
se logró 99% de validez de formato, como se detalla en 
reportes/resultados_metricas_20260512/metricas_resumen.json"

### Formato 4: Referencia a Figuras
"La Figura X (generada por generar_reporte_metricas_v2_fast.py) 
muestra la distribución de resultados..."

### Formato 5: Referencia a Reproducibilidad
"El experimento se reproduce ejecutando:
  python deploy/inference_jetson_pc.py --input dataset_alpr/images/val
  (Véase Sección 3.7 para detalles)"
```

---

## TIMELINE RECOMENDADO DE REDACCIÓN (Con Referencias)

**Semana 1:**
- Lunes: Leer ACCIONES_REALIZADAS_v2.md + entender problema
- Martes: Redactar Capítulo 2 (Marco Teórico)
- Miércoles: Redactar Capítulo 3.1-3.2 (Arquitectura, YOLO)
- Jueves: Redactar Capítulo 3.3 (ROI Bias) - SECCIÓN CLAVE
- Viernes: Completar Capítulo 3.3 con figuras y ecuaciones

**Semana 2:**
- Lunes: Redactar Capítulo 3.4-3.5 (OCR + Postprocessing) - OTRA CLAVE
- Martes: Redactar Capítulo 3.6-3.7 (Config, Métricas)
- Miércoles: Redactar Capítulo 4.1-4.2 (Resultados, errores)
- Jueves: Redactar Capítulo 4.3-4.4 (Confusión, latencia)
- Viernes: Redactar Capítulo 4.5-4.6 + Capítulo 5

**Semana 3:**
- Lunes-Viernes: Revisión, integración de figuras, validación reproducibilidad

---

## CHECKLIST DE "VÍ ESTE ARCHIVO" (Validación)

- [ ] Abrí `deploy/runtime_config.py` y vi los valores 0.30, 0.20, 0.12
- [ ] Abrí `deploy/ocr_postprocessor.py` y entiendo la validación + retry
- [ ] Abrí `reportes/.../metricas_resumen.json` y vi 100%, 99%, 33%
- [ ] Abrí `reportes/.../tablas/metricas_detalle_imagen.csv` y cuenté 100 filas
- [ ] Abrí las 5 gráficas PNG y entiendo cada una
- [ ] Abrí `ACCIONES_REALIZADAS_v2.md` y entiendo el diagnóstico
- [ ] Corrí `generar_reporte_metricas_v2_fast.py` (o revisé salida)
- [ ] Vi `Informe_metricas_y_graficas_v3_100.docx` completo

---

**Documento preparado:** MAPA_ARCHIVOS_A_CAPITULOS_TESIS.md
**Uso:** Referencia visual mientras redactas cada capítulo
**Actualizado:** Mayo 16, 2026

