"""
🎯 ENTRENAMIENTO DEL DETECTOR LIGERO PARA JETSON NANO
"""

import os
import json
import cv2
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

CONFIG = {
    "input_size": (416, 416),              # Alineado con inferencia y mejor localización
    "batch_size": 8,                        # Más seguro para 416x416 en GTX 1650
    "epochs": 40,
    "learning_rate": 5e-5,
    "data_augmentation": True,
    "early_stopping_patience": 8,
    "bbox_loss_weight": 3.0,
    "confidence_loss_weight": 0.25,
    "bbox_shrink_factor": 0.45,
}

# ============================================================================
# GENERADOR DE DATOS OPTIMIZADO
# ============================================================================

class LicensePlateDataGenerator(tf.keras.utils.Sequence):
    """Generador de datos optimizado para memoria limitada"""
    
    def __init__(self, image_paths, label_paths, batch_size=8, img_size=(416, 416),
                 augment=True, shuffle=True):
        self.image_paths = np.array(image_paths)
        self.label_paths = np.array(label_paths)
        self.batch_size = batch_size
        self.img_size = img_size
        self.augment = augment
        self.shuffle = shuffle
        self.indices = np.arange(len(self.image_paths))
        
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))
    
    def __getitem__(self, idx):
        batch_indices = self.indices[
            idx * self.batch_size:(idx + 1) * self.batch_size
        ]
        
        images = []
        bboxes = []
        
        for i in batch_indices:
            img = cv2.imread(str(self.image_paths[i]))
            if img is None:
                continue
            
            # Resize
            img = cv2.resize(img, self.img_size)
            img = img / 255.0
            
            # Cargar bbox (soporta labels YOLO txt y PascalVOC xml)
            bbox = self._load_bbox(self.label_paths[i])
            if bbox is not None:
                # Augmentation
                if self.augment:
                    img, bbox = self._augment(img, bbox)

                images.append(img)
                bboxes.append(bbox)
        
        if not images:
            # Retornar batch vacío si hay error
            return np.zeros((self.batch_size, *self.img_size, 3)), \
                   np.zeros((self.batch_size, 5))
        
        # Pad a batch_size
        while len(images) < self.batch_size:
            images.append(np.zeros((*self.img_size, 3)))
            bboxes.append(np.zeros(5))
        
        return np.array(images[:self.batch_size]), np.array(bboxes[:self.batch_size])
    
    def _augment(self, img, bbox):
        """Augmentación ligera de datos"""
        # Flip horizontal con ajuste correcto de la caja.
        if np.random.rand() > 0.5:
            img = cv2.flip(img, 1)
            bbox = bbox.copy()
            bbox[0] = 1.0 - bbox[0]
        
        # Contraste/brillo
        if np.random.rand() > 0.5:
            alpha = np.random.uniform(0.8, 1.2)
            beta = np.random.uniform(-20, 20)
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta/255.0)
            img = np.clip(img, 0, 1)
        
        # Ruido gaussiano
        if np.random.rand() > 0.7:
            noise = np.random.normal(0, 0.01, img.shape)
            img = np.clip(img + noise, 0, 1)
        
        return img, bbox

    def _load_bbox(self, label_path):
        """Carga bbox normalizado [x,y,w,h,conf] desde txt o xml."""
        label_path = Path(label_path)

        try:
            if label_path.suffix.lower() == ".txt":
                with open(label_path, "r", encoding="utf-8", errors="ignore") as f:
                    line = f.readline().strip()
                if not line:
                    return None
                parts = line.split()
                if len(parts) < 5:
                    return None
                _, x, y, w, h = [float(p) for p in parts[:5]]
                w *= CONFIG["bbox_shrink_factor"]
                h *= CONFIG["bbox_shrink_factor"]
                w = max(0.02, min(1.0, w))
                h = max(0.02, min(1.0, h))
                return np.array([x, y, w, h, 1.0], dtype=np.float32)

            if label_path.suffix.lower() == ".xml":
                root = ET.parse(label_path).getroot()

                size = root.find("size")
                if size is None:
                    return None

                img_w = float(size.findtext("width", default="0"))
                img_h = float(size.findtext("height", default="0"))
                if img_w <= 0 or img_h <= 0:
                    return None

                obj = root.find("object")
                if obj is None:
                    return None

                bnd = obj.find("bndbox")
                if bnd is None:
                    return None

                xmin = float(bnd.findtext("xmin", default="0"))
                ymin = float(bnd.findtext("ymin", default="0"))
                xmax = float(bnd.findtext("xmax", default="0"))
                ymax = float(bnd.findtext("ymax", default="0"))

                if xmax <= xmin or ymax <= ymin:
                    return None

                x = ((xmin + xmax) / 2.0) / img_w
                y = ((ymin + ymax) / 2.0) / img_h
                w = (xmax - xmin) / img_w
                h = (ymax - ymin) / img_h

                w *= CONFIG["bbox_shrink_factor"]
                h *= CONFIG["bbox_shrink_factor"]
                w = max(0.02, min(1.0, w))
                h = max(0.02, min(1.0, h))

                return np.array([x, y, w, h, 1.0], dtype=np.float32)

        except Exception:
            return None

        return None


def detector_loss(y_true, y_pred):
    """Da más peso a la localización que a la confianza."""
    bbox_true = y_true[:, :4]
    bbox_pred = y_pred[:, :4]
    conf_true = y_true[:, 4:5]
    conf_pred = y_pred[:, 4:5]

    bbox_loss = tf.reduce_mean(tf.square(bbox_true - bbox_pred), axis=-1)
    conf_loss = tf.reduce_mean(tf.square(conf_true - conf_pred), axis=-1)

    return (
        CONFIG["bbox_loss_weight"] * bbox_loss
        + CONFIG["confidence_loss_weight"] * conf_loss
    )


# ============================================================================
# MODELO DETECTOR
# ============================================================================

def create_detector_model(input_size=(416, 416)):
    """Detector MobileNetV2 optimizado"""
    
    inputs = tf.keras.Input(shape=(*input_size, 3))
    
    # Backbone MobileNetV2 (alpha=1.0 con GPU potente)
    backbone = tf.keras.applications.MobileNetV2(
        input_shape=(*input_size, 3),
        include_top=False,
        weights='imagenet',
        alpha=1.0  # ↑ Modelo completo (mejor precisión)
    )
    
    # Fine-tuning del backbone (últimas capas)
    backbone.trainable = True
    for layer in backbone.layers[:-20]:
        layer.trainable = False
    
    # Encoder
    x = backbone(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    
    # Dense layers ampliados
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    
    x = tf.keras.layers.Dense(256, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Output: [x, y, w, h, confidence]
    outputs = tf.keras.layers.Dense(
        5, activation='sigmoid', dtype='float32', name='bbox_output'
    )(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    return model


def collect_pairs(images_dir: Path, labels_dir: Path):
    """Recolecta pares imagen-label válidos de un split."""
    pairs = []
    for img_file in images_dir.glob("*.[jp][pn]g"):
        txt_file = labels_dir / (img_file.stem + ".txt")
        xml_file = labels_dir / (img_file.stem + ".xml")

        # Prioridad: txt no vacío; fallback xml.
        if txt_file.exists() and txt_file.stat().st_size > 0:
            pairs.append((img_file, txt_file))
        elif xml_file.exists() and xml_file.stat().st_size > 0:
            pairs.append((img_file, xml_file))
    return pairs


# ============================================================================
# ENTRENAMIENTO
# ============================================================================

def train_detector():
    """Entrena detector ligero"""
    
    print("\n" + "="*60)
    print("🎯 ENTRENAMIENTO DEL DETECTOR")
    print("="*60)

    # Evita preasignar toda la VRAM para mayor estabilidad.
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✓ GPU detectada: {len(gpus)}")
    else:
        print("⚠️ No se detectó GPU; entrenamiento en CPU")
    
    # DETECCIÓN: usar exclusivamente dataset_publico_placas
    dataset_path = Path("dataset_publico_placas")
    train_img_dir = dataset_path / "images" / "train"
    train_lbl_dir = dataset_path / "labels" / "train"
    val_img_dir = dataset_path / "images" / "val"
    val_lbl_dir = dataset_path / "labels" / "val"
    
    if not train_img_dir.exists() or not train_lbl_dir.exists():
        print("❌ Dataset de detector no encontrado en dataset_publico_placas")
        return

    train_pairs = collect_pairs(train_img_dir, train_lbl_dir)
    val_pairs = collect_pairs(val_img_dir, val_lbl_dir)

    print(
        f"ℹ️ dataset_publico_placas -> train pairs: {len(train_pairs)}, "
        f"val pairs: {len(val_pairs)}"
    )

    if len(train_pairs) == 0:
        print("❌ No hay pares válidos en train")
        return

    # Si no hay split val en dataset_publico_placas, se hace split local.
    if len(val_pairs) == 0:
        np.random.seed(42)
        np.random.shuffle(train_pairs)
        split_idx = int(len(train_pairs) * 0.85)
        val_pairs = train_pairs[split_idx:]
        train_pairs = train_pairs[:split_idx]
        print("⚠️ Split val no encontrado; usando 85/15 desde train")
    
    print(f"✓ Train: {len(train_pairs)}, Val: {len(val_pairs)}")
    
    # Generadores
    train_gen = LicensePlateDataGenerator(
        [p[0] for p in train_pairs],
        [p[1] for p in train_pairs],
        batch_size=CONFIG["batch_size"],
        img_size=CONFIG["input_size"],
        augment=CONFIG["data_augmentation"]
    )
    
    val_gen = LicensePlateDataGenerator(
        [p[0] for p in val_pairs],
        [p[1] for p in val_pairs],
        batch_size=CONFIG["batch_size"],
        img_size=CONFIG["input_size"],
        augment=False
    )
    
    # Modelo
    model = create_detector_model(CONFIG["input_size"])
    
    model.compile(
        optimizer=tf.keras.optimizers.AdamW(
            learning_rate=CONFIG["learning_rate"],
            weight_decay=1e-5
        ),
        loss=detector_loss,
        metrics=['mae']
    )
    
    print("\n📦 Modelo Detector:")
    print(f"  Parámetros: {model.count_params():,}")
    print(f"  RAM (batch=1): ~{model.count_params() * 4 / 1e6 * 3:.1f} MB")
    
    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=CONFIG["early_stopping_patience"],
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            'models/detector_best.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        ),
    ]
    
    # Entrenar
    print("\n🚀 Iniciando entrenamiento...")
    print(f"  Batches por época: {len(train_gen)}")
    print(f"  Épocas: {CONFIG['epochs']}")
    print(f"  Batch size: {CONFIG['batch_size']}")
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=CONFIG["epochs"],
        callbacks=callbacks,
        verbose=1
    )
    
    # Guardar
    model.save('models/detector_final.h5')
    print("\n✅ Modelo guardado en: models/detector_final.h5")
    
    # Guardar historial
    with open('outputs/training_history.json', 'w') as f:
        json.dump({
            'loss': [float(l) for l in history.history['loss']],
            'val_loss': [float(l) for l in history.history['val_loss']],
        }, f, indent=2)
    
    return model, train_gen, val_gen


def evaluate_detector(model, val_gen):
    """Evalúa el detector"""
    
    print("\n" + "="*60)
    print("📊 EVALUACIÓN DEL DETECTOR")
    print("="*60)
    
    val_loss, val_mae = model.evaluate(val_gen, verbose=0)
    
    print(f"✓ Validation Loss: {val_loss:.4f}")
    print(f"✓ Validation MAE:  {val_mae:.4f}")
    
    # Predecir en un batch
    sample_batch, sample_targets = val_gen[0]
    predictions = model.predict(sample_batch, verbose=0)
    
    # Analizar confianzas
    confidences = predictions[:, 4]
    mean_conf = np.mean(confidences)
    
    print(f"\n✓ Confianzas promedio: {mean_conf:.3f}")
    print(f"✓ Min-Max: [{np.min(confidences):.3f}, {np.max(confidences):.3f}]")
    
    if mean_conf > 0.75:
        print("✅ Confianza > 0.75 ✓")
    else:
        print("⚠️  Confianza < 0.75 - Requiere más entrenamiento")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("🚗 DETECTOR TRAINING PIPELINE\n")
    
    # Setup
    Path('models').mkdir(exist_ok=True)
    Path('outputs').mkdir(exist_ok=True)
    
    # Entrenar
    model, train_gen, val_gen = train_detector()
    
    # Evaluar
    evaluate_detector(model, val_gen)
    
    print("\n" + "="*60)
    print("✨ ENTRENAMIENTO COMPLETADO")
    print("="*60)
    print("""
    Siguientes pasos:
    1. Revisar outputs/training_history.json
    2. Si confianza < 0.75: ajusta epochs, learning_rate
    3. Entrenar OCR: python train_ocr.py
    4. Cuantizar: python quantize_models.py
    """)
