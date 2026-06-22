# ALPR_TIC

Plataforma ALPR para Jetson Nano organizada para reproducir inferencia, evaluación y documentación de presentación.

Este repositorio está preparado para publicarse en GitHub con el nombre `ALPR_TIC`.

## Resumen

El sistema realiza captura, preprocesamiento, detección de placas, OCR y postprocesamiento local en un dispositivo edge. La versión pública del árbol conserva solo lo necesario para ejecutar el flujo, revisar resultados y entender la estructura del proyecto.

## Configuración usada en Jetson Nano

| Elemento | Configuración usada | Función |
| --- | --- | --- |
| Dispositivo edge | reComputer J1010 basado en NVIDIA Jetson Nano 2 GB | Ejecutar inferencia local del sistema |
| Sistema operativo | Ubuntu 18.04 LTS con JetPack SDK | Proveer entorno base de ejecución |
| Cámara | Arducam IMX219 | Captura de imágenes o video de placas |
| Lenguaje de programación | Python 3.8 | Ejecutar el pipeline ALPR |
| Captura y preprocesamiento | OpenCV + V4L2 | Capturar fotogramas y preparar imágenes |
| Detector de placas | YOLO / Ultralytics | Localizar la placa dentro del fotograma |
| OCR principal | CRNN / CTC optimizado | Reconocer la secuencia textual |
| OCR de apoyo | RapidOCR | Apoyar la lectura en condiciones difíciles |
| Postprocesamiento | Reglas de formato y correcciones textuales | Validar y normalizar el resultado |

## Estructura pública

- `src/`: implementación principal del sistema.
- `deploy/`: backend operativo de inferencia, OCR y postprocesamiento.
- `experiments/`: scripts de evaluación del detector, OCR y pipeline.
- `results/`: resultados finales generados por los experimentos.
- `data/`: documentación del dataset y particiones listas para usar.
- `models/`: solo los modelos finales de inferencia.
- `docs/`: guía de uso y diagramas de soporte.
- `environment/`: configuración para Jetson y preparación del entorno.

## Modelo de trabajo

El punto de entrada de ejecución es `src/main_pipeline.py`, que llama al backend de inferencia en `deploy/inference_jetson.py`. Para monitoreo en Jetson, el flujo recomendado es el script de captura de métricas en `experiments/monitor_jetson_runtime.py`.

## Instalación rápida

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

```bash
pip install -r requirements.txt
```

## Ejecución

Inferencia principal:

```bash
python src/main_pipeline.py
```

Evaluaciones:

```bash
python experiments/run_detection_tests.py
python experiments/run_ocr_tests.py
python experiments/run_pipeline_tests.py
```

Jetson con monitoreo:

```bash
python3 experiments/monitor_jetson_runtime.py \
	--repo-root . \
	--video 0 \
	--runtime-profile jetson-realtime \
	--no-window \
	--use-tegrastats
```

## Datos y modelos

- `data/splits/` contiene la partición lista para uso.
- `models/plate_detector_best.pt` es el detector final.
- `models/ocr_crnn.tflite` es el OCR final.

## Resultados

Los resultados generados por las evaluaciones se concentran en `results/`.

## Documentación clave

- [docs/usage_guide.md](docs/usage_guide.md)
- [environment/jetson_setup.md](environment/jetson_setup.md)
- [data/README_dataset.md](data/README_dataset.md)

## Publicación en GitHub

Para publicar el repositorio con el nombre `ALPR_TIC`, crea el repositorio remoto con ese nombre y luego sube este árbol local. Desde aquí ya quedó listo el contenido y la portada.
