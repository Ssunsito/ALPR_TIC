# Intento2

Proyecto ALPR para detección y reconocimiento de placas vehiculares, con una evolución por etapas que va desde el entrenamiento y validación, hasta el despliegue en Jetson, el postprocesamiento OCR, la generación de métricas y la preparación para federated learning.

## Etapas del proyecto

1. Entrenamiento y ajuste de detectores de placas.
2. Integración del pipeline de inferencia en Jetson.
3. Postprocesamiento OCR, validación de formato y retry logic.
4. Versión PC / imágenes y generación de métricas.
5. Documentación final y roadmap hacia federated learning.

## Estructura principal

- `train_detector.py` y `train_detector_small_far_yolo.py`: entrenamiento de detectores.
- `deploy/`: inferencia, OCR y configuración runtime.
- `reportes/`: scripts de métricas y reportes.
- `docs/`: roadmap y documentación de implementación.
- `models/`: ubicación de los pesos locales del proyecto.

## Nota sobre archivos pesados

Los datasets, resultados generados, artefactos de entrenamiento y pesos binarios no se incluyen en GitHub por tamaño y mantenimiento. Si hace falta, pueden versionarse aparte con Git LFS o como descargas externas.
