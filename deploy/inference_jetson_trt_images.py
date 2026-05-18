"""
Inference wrapper (TensorRT-style config) for processing image files.
Duplicates the advanced TRTensorRT pipeline config and runs detection+OCR on images.
Usage:
    python inference_jetson_trt_images.py --input_dir path/to/images --out_dir outputs/annotated
"""
import argparse
import os
from pathlib import Path
import shutil
import time

# Reuse existing pipeline implementation
from inference_jetson import PipelineJetson
from runtime_config import CONFIG as BASE_CONFIG, apply_runtime_profile


def build_config():
    cfg = dict(BASE_CONFIG)
    # TensorRT-style/advanced settings (duplicate from inference_jetson_trt.py)
    cfg.update({
        "use_gpu": True,
        "yolo_imgsz": 640,
        "camera_detection_stride": 2,
        "camera_preview_width": 800,
        "camera_display_width": 800,
        # Ensure OCR ROI bias uses the downward-shifted values
        "ocr_roi_y_bias_close": cfg.get("ocr_roi_y_bias_close", 0.30),
        "ocr_roi_y_bias_medium": cfg.get("ocr_roi_y_bias_medium", 0.20),
        "ocr_roi_y_bias_small": cfg.get("ocr_roi_y_bias_small", 0.12),
        "camera_ocr_roi_candidates_max": 3,
        "camera_ocr_use_detector_box": True,
    })
    return cfg


def process_images(input_dir: Path, out_dir: Path, cfg, limit: int = 0):
    pipeline = PipelineJetson(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in input_dir.iterdir() if p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.webp'}])
    if limit > 0:
        images = images[:limit]

    stats = []
    for i, img_path in enumerate(images, 1):
        t0 = time.time()
        frame, detections = pipeline.run_single_image(str(img_path), show_window=False)
        t1 = time.time()
        out_path = out_dir / f"{img_path.stem}_annotated{img_path.suffix}"
        try:
            # frame is BGR numpy
            import cv2
            cv2.imwrite(str(out_path), frame)
        except Exception:
            pass
        stats.append({"image": img_path.name, "detections": len(detections), "latency_ms": (t1 - t0) * 1000})
        print(f"[{i}/{len(images)}] {img_path.name} -> {len(detections)} detections, {stats[-1]['latency_ms']:.0f} ms")
    return stats


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True)
    parser.add_argument('--out_dir', default='outputs/annotated')
    parser.add_argument('--limit', type=int, default=0)
    args = parser.parse_args()

    inp = Path(args.input_dir)
    out = Path(args.out_dir)
    if not inp.exists():
        raise SystemExit(f"Input dir not found: {inp}")

    cfg = build_config()
    apply_runtime_profile(cfg, 'jetson-quality')
    print('Running with config overrides: ocr_roi_y_bias_close=', cfg.get('ocr_roi_y_bias_close'))
    stats = process_images(inp, out, cfg, limit=args.limit)
    print('Done. Processed', len(stats), 'images')
