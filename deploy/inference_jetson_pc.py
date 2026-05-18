# -*- coding: utf-8 -*-
"""
Inference para PC sobre imágenes usando la misma pipeline de `inference_jetson.py`.
No abre cámara. Procesa un archivo o un directorio de imágenes y guarda anotaciones.

Uso:
    python deploy/inference_jetson_pc.py --input dataset_alpr/images/val --output-dir outputs/pc
    python deploy/inference_jetson_pc.py --image dataset_alpr/images/val/AAB-4475.png --output outputs/pc/AAB-4475_annotated.png
"""

import argparse
from pathlib import Path
from typing import Optional

# Reutiliza la implementación principal existente
from inference_jetson import PipelineJetson
from runtime_config import CONFIG, apply_runtime_profile

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def build_config(args):
    cfg = dict(CONFIG)
    apply_runtime_profile(cfg, args.runtime_profile)

    if args.detector_model:
        cfg["detector_model"] = args.detector_model
    if args.ocr_model:
        cfg["ocr_model"] = args.ocr_model
    if args.ocr_engine:
        cfg["ocr_engine"] = args.ocr_engine
    if args.ocr_profile:
        cfg["ocr_profile"] = args.ocr_profile
    if args.no_window:
        cfg["show_window"] = False

    # PC: prioriza calidad razonable y ROI OCR más bajo para evitar encabezados.
    cfg["ocr_roi_y_bias_close"] = float(cfg.get("ocr_roi_y_bias_close", 0.30))
    cfg["ocr_roi_y_bias_medium"] = float(cfg.get("ocr_roi_y_bias_medium", 0.20))
    cfg["ocr_roi_y_bias_small"] = float(cfg.get("ocr_roi_y_bias_small", 0.12))
    cfg["camera_ocr_use_worker"] = False
    cfg["camera_ocr_use_detector_box"] = True
    cfg["camera_ocr_detector_box_expand"] = float(args.camera_ocr_expand or cfg.get("camera_ocr_detector_box_expand", 1.18))
    cfg["camera_display_width"] = int(args.camera_display_width or cfg.get("camera_display_width", 800))
    return cfg


def iter_input_images(path: Path):
    if path.is_file():
        yield path
        return
    for item in sorted(path.iterdir()):
        if item.is_file() and item.suffix.lower() in IMAGE_EXTS:
            yield item


def process_image(pipeline: PipelineJetson, image_path: Path, output_path: Optional[Path], show_window: bool):
    annotated, detections = pipeline.run_single_image(str(image_path), str(output_path) if output_path else None, show_window=show_window)
    return detections


def main():
    parser = argparse.ArgumentParser(description="Inference para PC sobre imágenes usando la pipeline de placas")
    parser.add_argument("--input", type=str, default=None, help="Imagen o directorio de imágenes")
    parser.add_argument("--image", type=str, default=None, help="Imagen única")
    parser.add_argument("--output", type=str, default=None, help="Salida para imagen única anotada")
    parser.add_argument("--output-dir", type=str, default="outputs/pc", help="Directorio de salida para modo carpeta")
    parser.add_argument("--limit", type=int, default=0, help="Límite de imágenes en modo directorio")
    parser.add_argument("--runtime-profile", type=str, default="jetson-quality", choices=["current", "jetson-stable", "jetson-quality", "jetson-production", "jetson-realtime"], help="Perfil runtime")
    parser.add_argument("--detector-model", type=str, default=None, help="Ruta del detector")
    parser.add_argument("--ocr-model", type=str, default=None, help="Ruta del OCR")
    parser.add_argument("--ocr-engine", type=str, default=None, choices=["rapidocr-lpr", "paddle-latin", "ctc", "simple", "tesseract"], help="Motor OCR")
    parser.add_argument("--ocr-profile", type=str, default=None, choices=["balanced", "far"], help="Perfil OCR")
    parser.add_argument("--camera-ocr-expand", type=float, default=None, help="Expansión del bbox para OCR")
    parser.add_argument("--camera-display-width", type=int, default=None, help="Ancho de visualización")
    parser.add_argument("--no-window", action="store_true", help="No abrir ventana OpenCV")
    args = parser.parse_args()

    source = args.image or args.input
    if not source:
        raise SystemExit("Debes indicar --image o --input")

    source_path = Path(source)
    if not source_path.exists():
        raise SystemExit(f"No existe la ruta de entrada: {source_path}")

    cfg = build_config(args)
    pipeline = PipelineJetson(cfg)
    show_window = bool(cfg.get("show_window", True)) and not args.no_window

    if source_path.is_file() or args.image:
        output_path = Path(args.output) if args.output else None
        detections = process_image(pipeline, source_path, output_path, show_window)
        print(f"Procesada 1 imagen. Detecciones: {len(detections)}")
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images = list(iter_input_images(source_path))
    if args.limit and args.limit > 0:
        images = images[:args.limit]

    print(f"Procesando {len(images)} imágenes desde {source_path}")
    total_detections = 0
    for idx, image_path in enumerate(images, 1):
        out_path = output_dir / f"{image_path.stem}_annotated{image_path.suffix}"
        detections = process_image(pipeline, image_path, out_path, show_window)
        total_detections += len(detections)
        print(f"[{idx}/{len(images)}] {image_path.name} -> {len(detections)} detecciones")

    print(f"Listo. Imágenes procesadas: {len(images)}. Detecciones totales: {total_detections}")


if __name__ == "__main__":
    main()
