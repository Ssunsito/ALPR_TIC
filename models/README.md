# Modelos

Esta carpeta se reserva para los pesos y exportaciones del proyecto.

## Organización esperada

- `plate_detector_best.pt`: detector principal de placas.
- `optimized/ocr_crnn_ctc_int8.tflite`: modelo OCR cuantizado.
- Otros pesos experimentales pueden mantenerse solo de forma local.

## Nota

Los binarios grandes no se versionan en este repositorio para evitar superar límites de GitHub. Si necesitas reproducibilidad completa, añade los pesos manualmente o usa Git LFS.
