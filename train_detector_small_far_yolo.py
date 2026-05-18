from __future__ import annotations

import argparse
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from train_plate_yolo import ensure_base_model


@dataclass
class PlateSample:
    image_path: Path
    label_path: Path
    line: str
    area_ratio: float


def iter_images(images_dir: Path):
    for p in images_dir.iterdir():
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".avif"}:
            yield p


def parse_first_bbox(label_path: Path):
    if not label_path.exists():
        return None

    for raw in label_path.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = raw.strip().split()
        if len(parts) < 5:
            continue
        cls_id = parts[0]
        x, y, w, h = map(float, parts[1:5])
        area = max(0.0, w) * max(0.0, h)
        return cls_id, x, y, w, h, area, raw.strip()

    return None


def collect_samples(root: Path, split: str):
    images_dir = root / "images" / split
    labels_dir = root / "labels" / split

    samples: list[PlateSample] = []
    for image_path in iter_images(images_dir):
        label_path = labels_dir / f"{image_path.stem}.txt"
        parsed = parse_first_bbox(label_path)
        if parsed is None:
            continue
        _, _, _, _, _, area, line = parsed
        samples.append(PlateSample(image_path, label_path, line, area))
    return samples


def degrade_far_plate(image):
    h, w = image.shape[:2]
    scale = random.uniform(0.35, 0.65)
    small_w = max(24, int(w * scale))
    small_h = max(24, int(h * scale))
    resized = cv2.resize(image, (small_w, small_h), interpolation=cv2.INTER_AREA)
    up = cv2.resize(resized, (w, h), interpolation=cv2.INTER_LINEAR)

    if random.random() < 0.6:
        k = random.choice([3, 5])
        up = cv2.GaussianBlur(up, (k, k), 0)

    if random.random() < 0.5:
        noise = random.random() * 8.0
        gauss = np.random.normal(0.0, noise, (h, w)).astype("float32")
        for c in range(3):
            up[..., c] = cv2.add(up[..., c].astype("float32"), gauss, dtype=cv2.CV_8U)

    return up


def write_label(path: Path, line: str):
    path.write_text(line + "\n", encoding="utf-8")


def copy_base_split(source_root: Path, target_root: Path, split: str):
    src_images = source_root / "images" / split
    src_labels = source_root / "labels" / split
    dst_images = target_root / "images" / split
    dst_labels = target_root / "labels" / split
    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    for image_path in iter_images(src_images):
        shutil.copy2(image_path, dst_images / image_path.name)
        src_label = src_labels / f"{image_path.stem}.txt"
        dst_label = dst_labels / f"{image_path.stem}.txt"
        if src_label.exists():
            shutil.copy2(src_label, dst_label)
        else:
            dst_label.write_text("", encoding="utf-8")


def prepare_small_far_dataset(
    source_root: Path,
    target_root: Path,
    small_area_thr: float,
    repeat_small: int,
    synth_far_each: int,
):
    if target_root.exists():
        shutil.rmtree(target_root)

    copy_base_split(source_root, target_root, "train")
    copy_base_split(source_root, target_root, "val")

    train_samples = collect_samples(source_root, "train")
    small_samples = [s for s in train_samples if s.area_ratio <= small_area_thr]

    dst_images = target_root / "images" / "train"
    dst_labels = target_root / "labels" / "train"

    added_repeats = 0
    added_synth = 0

    for idx, sample in enumerate(small_samples):
        image = cv2.imread(str(sample.image_path))
        if image is None:
            continue

        for r in range(max(0, repeat_small - 1)):
            name = f"{sample.image_path.stem}_smallboost_{idx}_{r}{sample.image_path.suffix}"
            out_img = dst_images / name
            out_lbl = dst_labels / f"{Path(name).stem}.txt"
            cv2.imwrite(str(out_img), image)
            write_label(out_lbl, sample.line)
            added_repeats += 1

        for s in range(max(0, synth_far_each)):
            name = f"{sample.image_path.stem}_faraug_{idx}_{s}{sample.image_path.suffix}"
            out_img = dst_images / name
            out_lbl = dst_labels / f"{Path(name).stem}.txt"
            aug = degrade_far_plate(image)
            cv2.imwrite(str(out_img), aug)
            write_label(out_lbl, sample.line)
            added_synth += 1

    data_yaml = target_root / "data.yaml"
    data_yaml.write_text(
        f"path: {target_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "\n"
        "names:\n"
        "  0: plate\n",
        encoding="utf-8",
    )

    stats = {
        "train_samples": len(train_samples),
        "small_samples": len(small_samples),
        "small_ratio": (len(small_samples) / len(train_samples)) if train_samples else 0.0,
        "added_repeats": added_repeats,
        "added_synth_far": added_synth,
        "final_train_size": len(list(iter_images(dst_images))),
    }
    return data_yaml, stats


def train(args):
    base_model = ensure_base_model(Path(args.base_model))
    data_yaml, stats = prepare_small_far_dataset(
        source_root=Path(args.data_root),
        target_root=Path(args.boosted_root),
        small_area_thr=args.small_area_thr,
        repeat_small=args.repeat_small,
        synth_far_each=args.synth_far_each,
    )

    print("Small/Far dataset stats:")
    for k, v in stats.items():
        print(f"  - {k}: {v}")

    model = YOLO(str(base_model))

    results = model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        patience=args.patience,
        project=args.project,
        name=args.name,
        pretrained=True,
        optimizer="AdamW",
        lr0=args.lr0,
        lrf=0.01,
        cos_lr=True,
        close_mosaic=10,
        freeze=args.freeze,
        workers=args.workers,
        single_cls=True,
        rect=False,
        cache=False,
        degrees=0.0,
        perspective=0.0,
        scale=0.35,
        translate=0.08,
    )

    best_weights = Path(results.save_dir) / "weights" / "best.pt"
    if not best_weights.exists():
        raise FileNotFoundError(f"No se encontró best.pt en {best_weights.parent}")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_weights, output)
    print(f"Modelo detector small/far guardado en: {output}")


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune YOLO detector for small/far plates")
    p.add_argument("--base-model", default="models/lp_yolov8n.pt")
    p.add_argument("--data-root", default="dataset_publico_placas")
    p.add_argument("--boosted-root", default="dataset_publico_placas_smallfar")
    p.add_argument("--output", default="models/plate_detector_smallfar_best.pt")

    p.add_argument("--small-area-thr", type=float, default=0.02)
    p.add_argument("--repeat-small", type=int, default=4)
    p.add_argument("--synth-far-each", type=int, default=2)

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default=0)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--project", default="runs/plate_detector")
    p.add_argument("--name", default="yolov8n_plate_smallfar")
    p.add_argument("--lr0", type=float, default=0.0012)
    p.add_argument("--freeze", type=int, default=8)
    p.add_argument("--workers", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
