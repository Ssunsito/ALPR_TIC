# Configuración Jetson

Este documento debe completar la información real del dispositivo usado para pruebas en edge.

## Script de captura y monitoreo real

Usa `experiments/monitor_jetson_runtime.py` para dos cosas en una sola ejecución:

1. Capturar la configuración exacta del dispositivo en un JSON.
2. Medir CPU, RAM, GPU y temperatura mientras corre `deploy/inference_jetson.py`.

El script actual está escrito sin anotaciones de tipo y con declaración UTF-8 en la cabecera para evitar errores de sintaxis en instalaciones antiguas de Python en Jetson.
Ahora el monitor ejecuta la inferencia de forma embebida dentro del mismo proceso Python, así evita el choque entre `python` y `python3` al lanzar `deploy/inference_jetson.py`.

Si ves un error en una línea como `def _safe_run(command: List[str]) -> Optional[str]:`, significa que estás ejecutando una copia antigua del script.

Si ves `SyntaxError: Non-ASCII character ... but no encoding declared`, estás ejecutando una versión sin la cabecera `# -*- coding: utf-8 -*-`.

### Ejemplo

```bash
python3 experiments/monitor_jetson_runtime.py \
	--repo-root . \
	--video 0 \
	--runtime-profile jetson-realtime \
	--no-window \
	--use-tegrastats
```

### Salidas esperadas

- `results/jetson_monitoring_YYYYMMDD_HHMMSS/jetson_system_snapshot.json`
- `results/jetson_monitoring_YYYYMMDD_HHMMSS/jetson_runtime_metrics.csv`
- `results/jetson_monitoring_YYYYMMDD_HHMMSS/inference_command.txt`
- `results/jetson_monitoring_YYYYMMDD_HHMMSS/jetson_runtime_summary.json`

## Pendiente de documentar

- Versión de Ubuntu.
- Versión de JetPack / L4T.
- Versión de Python.
- OpenCV instalado.
- PyTorch / TensorRT.
- TensorFlow Lite.
- RapidOCR y sus dependencias.
- Memoria RAM disponible y restricciones reales del dispositivo.

## Datos que sí conviene registrar con el script

- `nv_tegra_release`
- `uname -a`
- versión de Python
- versiones de `cv2`, `numpy`, `ultralytics`, `tensorflow`, `pytesseract`, `rapidocr_onnxruntime`, `paddleocr`, `psutil`
- total/usable de RAM
- zonas térmicas activas
- uso real durante inferencia

## Uso esperado

Debe acompañar los resultados de `results/latency_results.csv` y cualquier medición de CPU, RAM, GPU o temperatura.
