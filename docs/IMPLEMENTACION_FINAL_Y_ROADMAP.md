# Implementacion Final y Roadmap

## Estado final actual
- Pipeline principal: estable en Jetson Nano J1010.
- Script principal: inference_jetson.py.
- Rama paralela de aceleracion: inference_jetson_trt.py (sin activar en produccion).
- Captura de evidencia: guardado automatico de imagen con box + OCR confirmado.
- Perfil recomendado: jetson-realtime.

## Arquitectura final operativa
1. Captura de camara CSI (GStreamer/V4L2) con normalizacion para Y16/GREY.
2. Deteccion YOLO con stride adaptativo y salto en escena estatica.
3. Rescate de placa pequena periodico (bajo costo).
4. OCR asincrono con mejora de ROI y voto temporal.
5. Persistencia de evidencia en imagen anotada cuando OCR se confirma.

## Parametros clave de produccion (realtime)
- yolo_imgsz bajo para sostener FPS.
- camera_detection_stride adaptativo por objetivo de FPS.
- motion skip para reducir inferencias innecesarias.
- temporal vote OCR para estabilidad de texto.
- capture cooldown para evitar duplicados y crecimiento excesivo de evidencias.

## Roadmap historico del proyecto

### Fase 1: Base y entrenamiento
- Integracion de dataset y scripts de entrenamiento/export.
- Definicion de modelos detector/OCR y primeras validaciones.

### Fase 2: Deployment inicial Jetson
- Transferencia de artefactos, entorno Python y pruebas de arranque.
- Correccion de dependencias y rutas.

### Fase 3: Estabilizacion de video y OCR
- Correccion de frames negros en camara monocroma.
- OCR worker asincrono, cooldown y filtros para ruido.

### Fase 4: Optimizacion para tiempo real
- Runtime profiles (stable/quality/production/realtime).
- Ajustes de deteccion para sostener rendimiento en Nano 2GB.
- Consenso temporal OCR y guardado de capturas.

### Fase 5: Small plate boost
- Rescate periodico para placas pequenas y lejanas.
- Ajuste conservador para no bajar el umbral de FPS esperado.

### Fase 6: TensorRT (pendiente condicionado)
- Reanudar cuando exista almacenamiento suficiente.
- Flujo previsto: ONNX -> Engine FP16 -> Integracion backend TRT -> benchmark A/B.

## Checklist de operacion actual
- Ejecutar: python inference_jetson.py --video 0 --runtime-profile jetson-realtime
- Verificar: deteccion estable, OCR estable, capturas guardadas.
- Monitorear: FPS promedio, temperatura y uso de memoria.

## Plan de evolucion (sin romper produccion)
1. Mantener inference_jetson.py como baseline estable.
2. Probar cambios experimentales solo en ramas/copia separada.
3. Medir siempre A/B: precision OCR, recall de placas pequenas, FPS.
4. Promover a produccion solo si mejora o mantiene metricas objetivo.

## Estructura de repositorio historico (objetivo)
- docs/: documentacion consolidada (este archivo + troubleshooting).
- archive/temporal/: evidencia, logs, notebooks y documentacion previa.
- scripts principales en raiz: inferencia operativa y utilidades clave.
- datasets y modelos: preservados para reproducibilidad.
