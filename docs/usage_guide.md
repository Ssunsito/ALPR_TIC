# Guía de Uso

Este repositorio contiene el flujo completo para detección de placas, OCR, postprocesamiento y evaluación. La idea de esta guía es que una persona nueva pueda clonar el proyecto, instalar dependencias, preparar los datos y ejecutar las pruebas principales sin revisar el historial del proyecto.

## Estructura pública

- `src/`: código principal del sistema.
- `models/`: solo los modelos finales usados por `src/main_pipeline.py`.
- `data/`: metadatos del dataset y particiones preparadas en `data/splits/`.
- `deploy/`: backend de inferencia usado por `src/main_pipeline.py`.
- `experiments/`: scripts de evaluación.
- `results/`: resultados finales generados por las evaluaciones.
- `docs/`: guía de uso y diagramas.
- `environment/`: configuración para Jetson y preparación del entorno.

## Instalación

### 1. Crear y activar un entorno

```bash
python -m venv .venv
```

En Windows:

```bash
.venv\Scripts\activate
```

En Linux o Jetson:

```bash
source .venv/bin/activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

Si vas a ejecutar en Jetson, revisa además [environment/jetson_setup.md](../environment/jetson_setup.md) para dependencias del sistema y compatibilidad con cámara, TensorFlow, Tesseract y `tegrastats`.

## Datos

Los datos públicos ya están organizados en `data/splits/`.

La distribución aplicada es:

- `train`: 60%
- `val`: 30%
- `evaluation`: 10%

La partición se generó de forma determinista agrupando cada imagen con sus variantes augmentadas para que no se mezclen entre conjuntos.

La carpeta `data/README_dataset.md` resume el origen del dataset y `data/splits/` contiene las imágenes y etiquetas listas para uso.

## Modelos finales

Para inferencia se usan únicamente estos modelos:

- `models/plate_detector_best.pt`
- `models/ocr_crnn.tflite`

Los modelos intermedios y variantes experimentales se guardaron en `personal/` para no contaminar la entrega pública.

## Ejecución del sistema

### En escritorio o servidor

```bash
python src/main_pipeline.py
```

Este comando lanza la inferencia desde `deploy/inference_jetson.py` a través de `src/main_pipeline.py`.

### En Jetson

Lee primero [environment/jetson_setup.md](../environment/jetson_setup.md). El flujo de inferencia y monitoreo usa el script del monitor para ejecutar la inferencia y capturar métricas en segundo plano.

Comando base para la prueba en Jetson:

```bash
python3 experiments/monitor_jetson_runtime.py \
	--repo-root . \
	--video 0 \
	--runtime-profile jetson-realtime \
	--no-window \
	--use-tegrastats
```

## Evaluaciones

Los scripts de evaluación disponibles están en `experiments/`:

- `experiments/run_detection_tests.py`
- `experiments/run_ocr_tests.py`
- `experiments/run_pipeline_tests.py`

Comandos exactos:

```bash
python experiments/run_detection_tests.py
python experiments/run_ocr_tests.py
python experiments/run_pipeline_tests.py
```

Los resultados generados por esas ejecuciones se centralizan en `results/`.

## Resultados esperados

Después de correr las evaluaciones deberías encontrar, entre otros, estos archivos:

- `results/detection_results.csv`
- `results/ocr_results.csv`
- `results/pipeline_results.csv`
- `results/latency_results.csv`
- `results/figures/`

## Flujo recomendado

1. Instala dependencias.
2. Verifica que `models/` solo tenga los dos modelos finales.
3. Comprueba que `data/splits/` tenga `train`, `val` y `evaluation`.
4. Ejecuta las pruebas de `experiments/`.
5. Revisa `results/` y exporta los gráficos o tablas que necesites para el informe.

## Notas de mantenimiento

- Si agregas nuevos artefactos experimentales, colócalos en `personal/` antes de subir a GitHub.
- Si cambias los modelos finales, actualiza `deploy/runtime_config.py` para que apunte a los archivos correctos.
- Si trabajas en Jetson, mantén la guía de [environment/jetson_setup.md](../environment/jetson_setup.md) alineada con el intérprete y las dependencias reales del dispositivo.
