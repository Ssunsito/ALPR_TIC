"""
🚗 License Plate Detection & OCR Pipeline para Jetson Nano
Framework: TensorFlow/Keras optimizado para edge computing
"""

import os
import cv2
import numpy as np
import json
from pathlib import Path
from collections import defaultdict
import re

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

CONFIG = {
    "dataset_home": "dataset_alpr",  # o "dataset_publico_placas"
    "detector_input_size": (320, 320),
    "ocr_input_size": (32, 32),
    "batch_size": 8,
    "epochs_detector": 50,
    "epochs_ocr": 30,
    "validation_split": 0.15,
    "test_split": 0.15,
    "output_dir": "outputs",
    "models_dir": "models",
    "seed": 42,
}

# ============================================================================
# 1. VALIDACIÓN Y CARGA DE DATOS
# ============================================================================

def validate_datasets():
    """Valida integridad de ambos datasets"""
    print("\n" + "="*60)
    print("📋 VALIDACIÓN DE DATASETS")
    print("="*60)
    
    for dataset_name in ["dataset_publico_placas", "dataset_alpr"]:
        dataset_path = Path(dataset_name)
        if not dataset_path.exists():
            print(f"⚠️  {dataset_name} NO ENCONTRADO")
            continue
            
        print(f"\n✓ {dataset_name}")
        
        # Contar imágenes
        img_path = dataset_path / "images" / "train"
        if img_path.exists():
            imgs = list(img_path.glob("*.[jp][pn]g"))
            print(f"  - Imágenes train: {len(imgs)}")
        
        # Contar labels
        lbl_path = dataset_path / "labels" / "train"
        if lbl_path.exists():
            lbls = list(lbl_path.glob("*.txt"))
            print(f"  - Labels train: {len(lbls)}")
            
            # Verificar labels vacíos
            empty_labels = 0
            for lbl_file in lbls:
                if lbl_file.stat().st_size == 0:
                    empty_labels += 1
            if empty_labels > 0:
                print(f"  ⚠️  Labels vacíos: {empty_labels}")


def analyze_bbox_distribution(dataset_path="dataset_alpr"):
    """Analiza distribución de tamaños de bboxes"""
    print("\n" + "="*60)
    print("📊 ANÁLISIS DE BBOXES")
    print("="*60)
    
    sizes = defaultdict(int)
    lbl_path = Path(dataset_path) / "labels" / "train"
    
    if not lbl_path.exists():
        print("❌ No se encontraron labels")
        return
    
    for lbl_file in lbl_path.glob("*.txt"):
        try:
            with open(lbl_file) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        bbox_w = float(parts[3])
                        bbox_h = float(parts[4])
                        size_key = f"{int(bbox_w*100)}x{int(bbox_h*100)}"
                        sizes[size_key] += 1
        except:
            pass
    
    print(f"Total BBOXes: {sum(sizes.values())}")
    print("Top 5 tamaños:")
    for size, count in sorted(sizes.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {size}: {count}")


# ============================================================================
# 2. PREPARACIÓN DE DATOS
# ============================================================================

def prepare_dataset_splits(dataset_path="dataset_alpr"):
    """Crea train/val/test splits"""
    print("\n" + "="*60)
    print("🔧 PREPARACIÓN DE SPLITS")
    print("="*60)
    
    dataset_path = Path(dataset_path)
    img_dir = dataset_path / "images" / "train"
    lbl_dir = dataset_path / "labels" / "train"
    
    if not img_dir.exists():
        print("❌ Dataset no encontrado")
        return
    
    # Recolectar pares imagen-label
    pairs = []
    for img_file in img_dir.glob("*.[jp][pn]g"):
        lbl_file = lbl_dir / (img_file.stem + ".txt")
        if lbl_file.exists():
            pairs.append((img_file, lbl_file))
    
    print(f"Total pares imagen-label: {len(pairs)}")
    
    # Statistics
    np.random.seed(CONFIG["seed"])
    np.random.shuffle(pairs)
    
    n_total = len(pairs)
    n_test = int(n_total * CONFIG["test_split"])
    n_val = int((n_total - n_test) * CONFIG["validation_split"])
    n_train = n_total - n_test - n_val
    
    train_pairs = pairs[:n_train]
    val_pairs = pairs[n_train:n_train+n_val]
    test_pairs = pairs[n_train+n_val:]
    
    print(f"\n✓ Split realizado:")
    print(f"  Train: {len(train_pairs)} ({100*len(train_pairs)/n_total:.1f}%)")
    print(f"  Val:   {len(val_pairs)} ({100*len(val_pairs)/n_total:.1f}%)")
    print(f"  Test:  {len(test_pairs)} ({100*len(test_pairs)/n_total:.1f}%)")
    
    return train_pairs, val_pairs, test_pairs


def load_image_and_bbox(img_path, lbl_path, img_size=(320, 320)):
    """Carga imagen y bbox normalizado"""
    try:
        img = cv2.imread(str(img_path))
        if img is None:
            return None, None
        
        # Resize
        h_orig, w_orig = img.shape[:2]
        img = cv2.resize(img, img_size)
        
        # Cargar bbox
        with open(lbl_path) as f:
            line = f.readline().strip()
            if line:
                parts = line.split()
                cls, x, y, w, h = [float(p) for p in parts[:5]]
                return img / 255.0, np.array([cls, x, y, w, h])
        
        return img / 255.0, None
    except Exception as e:
        print(f"Error cargando {img_path}: {e}")
        return None, None


# ============================================================================
# 3. MODELOS (ARQUITECTURAS LIGERAS)
# ============================================================================

def create_detector_model(input_size=(320, 320)):
    """
    Crea detector MobileNetV2 SSD ligero.
    Output: bboxes (x, y, w, h, conf)
    """
    try:
        import tensorflow as tf
    except ImportError:
        print("❌ TensorFlow no instalado. Instala: pip install tensorflow")
        return None
    
    inputs = tf.keras.Input(shape=(*input_size, 3))
    
    # MobileNetV2 backbone
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(*input_size, 3),
        include_top=False,
        weights='imagenet',
        alpha=0.75  # 0.75x ancho para edge
    )
    backbone.trainable = False
    
    x = backbone(inputs)
    
    # Detector head
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    # Output: [bbox (4) + confidence (1)]
    outputs = tf.keras.layers.Dense(5, activation='sigmoid')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='mse',
        metrics=['mae']
    )
    
    print("✓ Detector model created (MobileNetV2 0.75x)")
    return model


def create_ocr_model(input_size=(32, 32), num_classes=36):
    """
    Crea modelo OCR compacto para clasificación de caracteres.
    Input: imagen de 32x32 de UN carácter
    Output: clase (A-Z, 0-9)
    """
    try:
        import tensorflow as tf
    except ImportError:
        return None
    
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(*input_size, 3)),
        
        # Conv block 1
        tf.keras.layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.2),
        
        # Conv block 2
        tf.keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Dropout(0.2),
        
        # Dense
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"✓ OCR model created ({num_classes} clases)")
    return model


# ============================================================================
# 4. UTILIDADES DE ENTRENAMIENTO
# ============================================================================

def setup_directories():
    """Crea directorios necesarios"""
    for dir_name in [CONFIG["output_dir"], CONFIG["models_dir"]]:
        Path(dir_name).mkdir(exist_ok=True)
    print(f"✓ Directorios creados: {CONFIG['output_dir']}, {CONFIG['models_dir']}")


def print_model_summary(model, name="Model"):
    """Imprime resumen del modelo optimizado para Jetson"""
    print(f"\n{'='*60}")
    print(f"📦 {name} - RESUMEN OPTIMIZADO")
    print(f"{'='*60}")
    
    total_params = model.count_params()
    print(f"Total parámetros: {total_params:,}")
    print(f"Tamaño estimado: {total_params * 4 / 1e6:.1f} MB (float32)")
    print(f"Tamaño INT8: {total_params / 1e6:.1f} MB (cuantizado)")
    
    # Primeras capas
    print("\nPrimeras capas:")
    for i, layer in enumerate(model.layers[:5]):
        print(f"  {i}. {layer.__class__.__name__}")


# ============================================================================
# 5. VALIDACIÓN Y BENCHMARKS
# ============================================================================

def estimate_memory_usage(model_params, batch_size=1):
    """Estima consumo de RAM"""
    # 4 bytes por parámetro (float32)
    weights_mb = model_params * 4 / 1e6
    # Activaciones aproximadamente 2x el tamaño de pesos
    activations_mb = weights_mb * 2
    # Batch overhead
    batch_overhead = batch_size * 10  # MB
    
    total = weights_mb + activations_mb + batch_overhead
    return total


def estimate_inference_time(input_size, model_name="unknown"):
    """Estima tiempo de inferencia en Jetson Nano"""
    # Benchmarks aproximados para Jetson Nano (GPU)
    pixels = input_size[0] * input_size[1]
    
    # ~0.05 ms por megapixel en GPU Jetson
    time_ms = (pixels / 1e6) * 50
    
    print(f"\n⚡ Benchmark: {model_name}")
    print(f"  Input: {input_size[0]}x{input_size[1]} ({pixels} px)")
    print(f"  Tiempo inferencia: ~{time_ms:.0f} ms")
    print(f"  FPS: ~{1000/time_ms:.1f} fps")
    
    return time_ms


# ============================================================================
# 6. FLUJO PRINCIPAL
# ============================================================================

def main():
    """Pipeline completo"""
    print("\n" + "🚀"*20)
    print(" LICENSE PLATE DETECTION & OCR PARA JETSON NANO 2GB")
    print("🚀"*20 + "\n")
    
    # Setup
    setup_directories()
    
    # Validación
    validate_datasets()
    analyze_bbox_distribution()
    
    # Preparación
    train_pairs, val_pairs, test_pairs = prepare_dataset_splits()
    
    # Modelos
    print("\n" + "="*60)
    print("🔨 CREACIÓN DE MODELOS")
    print("="*60)
    
    detector = create_detector_model()
    ocr_model = create_ocr_model()
    
    if detector:
        print_model_summary(detector, "DETECTOR")
        
        # Estimaciones
        detector_params = detector.count_params()
        detector_ram = estimate_memory_usage(detector_params, batch_size=1)
        print(f"\n💾 RAM Detector (batch_size=1): {detector_ram:.1f} MB")
        
        detector_time = estimate_inference_time(
            CONFIG["detector_input_size"],
            "Detector (MobileNetV2 0.75x)"
        )
    
    if ocr_model:
        print_model_summary(ocr_model, "OCR")
        
        ocr_params = ocr_model.count_params()
        ocr_ram = estimate_memory_usage(ocr_params, batch_size=32)
        print(f"\n💾 RAM OCR (batch_size=32): {ocr_ram:.1f} MB")
        
        ocr_time = estimate_inference_time(
            CONFIG["ocr_input_size"],
            "OCR (36 clases)"
        )
    
    # Resumen total
    print("\n" + "="*60)
    print("📈 RESUMEN FINAL")
    print("="*60)
    
    total_params = (detector_params if detector else 0) + (ocr_params if ocr_model else 0)
    total_ram = (detector_ram if detector else 0) + (ocr_ram if ocr_model else 0)
    total_time = (detector_time if detector else 0) + (ocr_time if ocr_model else 0)
    fps_total = 1000 / max(total_time, 1)
    
    print(f"\n✓ Parámetros totales: {total_params:,}")
    print(f"✓ RAM requerida: {total_ram:.1f} MB")
    print(f"✓ Latencia total: ~{total_time:.0f} ms")
    print(f"✓ FPS esperado: ~{fps_total:.1f} fps")
    print(f"\n⚠️  COMPATIBLE CON JETSON NANO 2GB: {'SÍ ✓' if total_ram < 800 else 'NO ✗'}")
    
    print("\n" + "="*60)
    print("🎯 SIGUIENTES PASOS:")
    print("="*60)
    print("""
    1. [ ] Instalar: pip install tensorflow opencv-python numpy
    2. [ ] Ejecutar: python -c "import tensorflow; print(tf.__version__)"
    3. [ ] Entrenar detector: python train_detector.py
    4. [ ] Entrenar OCR: python train_ocr.py
    5. [ ] Optimizar modelos: python quantize_models.py
    6. [ ] Probar en Jetson: scp models/* jetson:~/lp_detection/models/
    """)
    
    # Guardar configuración
    with open(Path(CONFIG["output_dir"]) / "config.json", "w") as f:
        json.dump(CONFIG, f, indent=2)
    print(f"\n✓ Configuración guardada en {CONFIG['output_dir']}/config.json")


if __name__ == "__main__":
    main()
