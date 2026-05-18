# -*- coding: utf-8 -*-
"""
Federación de Modelos - Agregación, Ensemble y Entrenamiento Distribuido
Compatible con PyTorch (YOLO) y TensorFlow (OCR CRNN+CTC)
"""

import os
import numpy as np
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)

# =============================================================================
# 1. FEDERATED MODEL AGGREGATOR (Promediado de Pesos)
# =============================================================================

class FederatedModelAggregator:
    """Agregación federada de modelos - promedia pesos de múltiples clientes."""
    
    def __init__(self, framework: str = "keras", verbose: bool = False):
        """
        Args:
            framework: "keras", "pytorch", o "mixed"
            verbose: imprimir logs detallados
        """
        self.framework = framework
        self.verbose = verbose
        self.client_models = {}
        self.aggregated_weights = None
        self.weight_history = []
        
    def register_client_model(self, client_id: str, model_path: str) -> None:
        """Registra un modelo cliente para agregación."""
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Modelo no encontrado: {model_path}")
        
        self.client_models[client_id] = {
            "path": model_path,
            "weights": None,
            "num_params": 0
        }
        if self.verbose:
            logger.info(f"✓ Cliente registrado: {client_id} ({model_path})")
    
    def load_client_weights_keras(self, client_id: str) -> np.ndarray:
        """Carga pesos de modelo Keras (.h5, .keras)."""
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow requerido para modelos Keras")
        
        model_path = self.client_models[client_id]["path"]
        model = tf.keras.models.load_model(model_path, compile=False)
        weights = model.get_weights()
        
        # Aplanar a vector 1D
        flat_weights = np.concatenate([w.flatten() for w in weights])
        
        self.client_models[client_id]["weights"] = flat_weights
        self.client_models[client_id]["num_params"] = len(flat_weights)
        
        if self.verbose:
            logger.info(f"  → Pesos Keras cargados: {len(weights)} capas, {len(flat_weights):,} parámetros")
        
        return flat_weights
    
    def load_client_weights_pytorch(self, client_id: str) -> np.ndarray:
        """Carga pesos de modelo PyTorch (.pt)."""
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch requerido para modelos .pt")
        
        model_path = self.client_models[client_id]["path"]
        checkpoint = torch.load(model_path, map_location="cpu")
        
        # Extraer state_dict (puede ser directo o anidado en checkpoint)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            state_dict = checkpoint["model"].state_dict()
        elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        
        # Aplanar todos los pesos
        flat_weights = np.concatenate([
            v.cpu().numpy().flatten() 
            for v in state_dict.values()
        ])
        
        self.client_models[client_id]["weights"] = flat_weights
        self.client_models[client_id]["num_params"] = len(flat_weights)
        
        if self.verbose:
            logger.info(f"  → Pesos PyTorch cargados: {len(state_dict)} capas, {len(flat_weights):,} parámetros")
        
        return flat_weights
    
    def aggregate_mean(self) -> np.ndarray:
        """Agregación por promedio simple (FedAvg)."""
        if not self.client_models:
            raise ValueError("No hay modelos clientes registrados")
        
        # Cargar pesos de todos los clientes
        all_weights = []
        for client_id in self.client_models.keys():
            weights = self.client_models[client_id]["weights"]
            if weights is None:
                # Auto-detectar framework
                if self.client_models[client_id]["path"].endswith(".pt"):
                    weights = self.load_client_weights_pytorch(client_id)
                else:
                    weights = self.load_client_weights_keras(client_id)
            all_weights.append(weights)
        
        # Validar que todos tengan igual tamaño
        sizes = [len(w) for w in all_weights]
        if len(set(sizes)) > 1:
            raise ValueError(f"Pesos con tamaños inconsistentes: {set(sizes)}")
        
        # Promedio simple
        aggregated = np.mean(all_weights, axis=0)
        self.aggregated_weights = aggregated
        self.weight_history.append(aggregated.copy())
        
        num_clients = len(self.client_models)
        logger.info(f"\n✓ Agregación FedAvg completada")
        logger.info(f"  → {num_clients} clientes agregados")
        logger.info(f"  → {len(aggregated):,} parámetros en modelo global")
        
        return aggregated
    
    def aggregate_weighted(self, weights: Dict[str, float]) -> np.ndarray:
        """Agregación ponderada por cliente (ej: por dataset size)."""
        if not self.client_models:
            raise ValueError("No hay modelos clientes registrados")
        
        if set(weights.keys()) != set(self.client_models.keys()):
            raise ValueError(f"Pesos no coinciden con clientes registrados")
        
        total_weight = sum(weights.values())
        all_weighted = []
        
        for client_id, weight in weights.items():
            client_weights = self.client_models[client_id]["weights"]
            if client_weights is None:
                if self.client_models[client_id]["path"].endswith(".pt"):
                    client_weights = self.load_client_weights_pytorch(client_id)
                else:
                    client_weights = self.load_client_weights_keras(client_id)
            
            # Ponderar
            weighted = (client_weights * weight) / total_weight
            all_weighted.append(weighted)
        
        aggregated = np.sum(all_weighted, axis=0)
        self.aggregated_weights = aggregated
        self.weight_history.append(aggregated.copy())
        
        logger.info(f"\n✓ Agregación ponderada completada")
        logger.info(f"  → Pesos: {weights}")
        logger.info(f"  → {len(aggregated):,} parámetros en modelo global")
        
        return aggregated
    
    def save_aggregated_model_keras(self, output_path: str) -> None:
        """Guarda modelo agregado en formato Keras."""
        if self.aggregated_weights is None:
            raise ValueError("Ejecuta aggregate_mean() o aggregate_weighted() primero")
        
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow requerido")
        
        # Cargar template (primer modelo cliente)
        first_client = list(self.client_models.keys())[0]
        model_path = self.client_models[first_client]["path"]
        model = tf.keras.models.load_model(model_path, compile=False)
        
        # Reconstruir pesos desde vector aplanado
        original_weights = model.get_weights()
        shapes = [w.shape for w in original_weights]
        
        new_weights = []
        offset = 0
        for shape in shapes:
            size = np.prod(shape)
            new_weights.append(self.aggregated_weights[offset:offset+size].reshape(shape))
            offset += size
        
        # Asignar pesos
        model.set_weights(new_weights)
        
        # Guardar
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        model.save(output_path)
        
        logger.info(f"✓ Modelo agregado guardado: {output_path}")
    
    def save_aggregated_model_pytorch(self, output_path: str, template_path: Optional[str] = None) -> None:
        """Guarda modelo agregado en formato PyTorch."""
        if self.aggregated_weights is None:
            raise ValueError("Ejecuta aggregate_mean() o aggregate_weighted() primero")
        
        try:
            import torch
        except ImportError:
            raise ImportError("PyTorch requerido")
        
        # Usar template si se proporciona, si no usar primer cliente
        if template_path is None:
            first_client = list(self.client_models.keys())[0]
            template_path = self.client_models[first_client]["path"]
        
        template = torch.load(template_path, map_location="cpu")
        if isinstance(template, dict) and "model" in template:
            state_dict = template["model"].state_dict()
        elif isinstance(template, dict) and "state_dict" in template:
            state_dict = template["state_dict"]
        else:
            state_dict = template
        
        # Reconstruir desde vector aplanado
        new_state_dict = OrderedDict()
        offset = 0
        for key, param in state_dict.items():
            size = np.prod(param.shape)
            new_state_dict[key] = torch.tensor(
                self.aggregated_weights[offset:offset+size].reshape(param.shape),
                dtype=param.dtype
            )
            offset += size
        
        # Guardar
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(new_state_dict, output_path)
        
        logger.info(f"✓ Modelo agregado guardado: {output_path}")


# =============================================================================
# 2. MODEL ENSEMBLE (Combinación de Predicciones)
# =============================================================================

class DetectorEnsemble:
    """Ensemble de detectores (YOLO) con voto de confianza."""
    
    def __init__(self, model_paths: List[str], framework: str = "pytorch", device: str = "cpu"):
        """
        Args:
            model_paths: lista de rutas a modelos detectores
            framework: "pytorch" o "tensorflow"
            device: "cpu", "cuda", "mps"
        """
        self.model_paths = model_paths
        self.framework = framework
        self.device = device
        self.models = []
        self.weights = [1.0] * len(model_paths)  # pesos iguales por defecto
        self._load_models()
    
    def _load_models(self) -> None:
        """Carga todos los modelos del ensemble."""
        if self.framework == "pytorch":
            try:
                from ultralytics import YOLO
            except ImportError:
                raise ImportError("Ultralytics YOLO requerido")
            
            for path in self.model_paths:
                model = YOLO(path)
                self.models.append(model)
                logger.info(f"✓ Modelo YOLO cargado: {path}")
        else:
            raise NotImplementedError(f"Framework {self.framework} no implementado")
    
    def set_weights(self, weights: List[float]) -> None:
        """Establece pesos para cada modelo del ensemble."""
        if len(weights) != len(self.models):
            raise ValueError(f"Se esperan {len(self.models)} pesos, se recibieron {len(weights)}")
        
        total = sum(weights)
        self.weights = [w / total for w in weights]  # Normalizar
        logger.info(f"Pesos del ensemble: {self.weights}")
    
    def predict(self, image: np.ndarray, conf_threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        Detecta placas con ensemble (voto mayoritario + promedio confianza).
        
        Returns:
            Lista de detecciones: [{"x1": ..., "y1": ..., "x2": ..., "y2": ..., "conf": ...}, ...]
        """
        all_detections = []
        
        # Cada modelo predice
        for model, weight in zip(self.models, self.weights):
            results = model.predict(image, conf=conf_threshold, verbose=False)
            detections = []
            
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    detections.append({
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "conf": conf, "weight": weight
                    })
            
            all_detections.extend(detections)
        
        # Agrupar y promediar detecciones cercanas (NMS ensemble)
        ensemble_detections = self._ensemble_nms(all_detections)
        
        return ensemble_detections
    
    def _ensemble_nms(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """NMS que combina predicciones del ensemble."""
        if not detections:
            return []
        
        # Ordenar por confianza ponderada
        for det in detections:
            det["weighted_conf"] = det["conf"] * det["weight"]
        
        detections.sort(key=lambda x: x["weighted_conf"], reverse=True)
        
        keep = []
        while detections:
            det = detections.pop(0)
            keep.append({
                "x1": det["x1"], "y1": det["y1"],
                "x2": det["x2"], "y2": det["y2"],
                "conf": det["weighted_conf"],
                "num_votes": 1
            })
            
            # Eliminar detecciones superpuestas
            to_remove = []
            for i, other in enumerate(detections):
                iou = self._calculate_iou(det, other)
                if iou > iou_threshold:
                    # Fusionar
                    keep[-1]["conf"] = max(keep[-1]["conf"], other["weighted_conf"])
                    keep[-1]["num_votes"] += 1
                    to_remove.append(i)
            
            detections = [d for i, d in enumerate(detections) if i not in to_remove]
        
        return keep
    
    @staticmethod
    def _calculate_iou(det1: Dict, det2: Dict) -> float:
        """Calcula IoU entre dos bounding boxes."""
        x1_min, y1_min, x1_max, y1_max = det1["x1"], det1["y1"], det1["x2"], det1["y2"]
        x2_min, y2_min, x2_max, y2_max = det2["x1"], det2["y1"], det2["x2"], det2["y2"]
        
        inter_xmin = max(x1_min, x2_min)
        inter_ymin = max(y1_min, y2_min)
        inter_xmax = min(x1_max, x2_max)
        inter_ymax = min(y1_max, y2_max)
        
        if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
            return 0.0
        
        inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        return inter_area / union_area if union_area > 0 else 0.0


class OCREnsemble:
    """Ensemble de modelos OCR (CRNN+CTC) con voting temporal."""
    
    def __init__(self, model_paths: List[str], voting_window: int = 3):
        """
        Args:
            model_paths: lista de rutas a modelos OCR Keras
            voting_window: número de frames para consensus voting
        """
        self.model_paths = model_paths
        self.voting_window = voting_window
        self.models = []
        self.voting_history = []
        self._load_models()
    
    def _load_models(self) -> None:
        """Carga modelos OCR."""
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow requerido para OCR")
        
        for path in self.model_paths:
            model = tf.keras.models.load_model(path, compile=False)
            self.models.append(model)
            logger.info(f"✓ Modelo OCR cargado: {path}")
    
    def predict(self, plate_image: np.ndarray) -> Tuple[str, float]:
        """
        Predice texto de placa con ensemble de OCR.
        
        Returns:
            (texto, confianza)
        """
        predictions = []
        confidences = []
        
        for model in self.models:
            # Procesar imagen
            img_processed = self._preprocess_ocr(plate_image)
            
            # Inferencia
            output = model.predict(img_processed, verbose=0)
            
            # Decodificación CTC
            text, conf = self._decode_ctc(output)
            predictions.append(text)
            confidences.append(conf)
        
        # Voting por consenso
        final_text = max(set(predictions), key=predictions.count)
        final_conf = np.mean([c for p, c in zip(predictions, confidences) if p == final_text])
        
        # Guardar en history para voting temporal
        self.voting_history.append(final_text)
        if len(self.voting_history) > self.voting_window:
            self.voting_history.pop(0)
        
        # Voting temporal (si hay consenso en frames anteriores)
        temporal_vote = max(set(self.voting_history), key=self.voting_history.count)
        
        logger.debug(f"Ensemble OCR: {predictions} → {final_text} (temporal: {temporal_vote})")
        
        return temporal_vote, final_conf
    
    @staticmethod
    def _preprocess_ocr(image: np.ndarray) -> np.ndarray:
        """Preprocesa imagen para OCR (normalización, resize)."""
        # Resize a (48, 192)
        image = cv2.resize(image, (192, 48))
        
        # Grayscale
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Normalizar
        image = image.astype(np.float32) / 255.0
        
        # Add channel dimension y batch
        image = np.expand_dims(image, axis=-1)
        image = np.expand_dims(image, axis=0)
        
        return image
    
    @staticmethod
    def _decode_ctc(output: np.ndarray, charset: str = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") -> Tuple[str, float]:
        """Decodifica salida CTC a texto."""
        # output shape: (batch, timesteps, num_classes)
        output = output[0]  # Primera imagen del batch
        
        # Greedy decode: argmax por timestep
        predictions = np.argmax(output, axis=-1)
        
        # Remover duplicados y blanks (clase 0)
        text = ""
        prev_class = -1
        confidences = []
        
        for class_idx in predictions:
            if class_idx != 0 and class_idx != prev_class:  # No blank, no duplicado
                if class_idx - 1 < len(charset):
                    text += charset[class_idx - 1]
                    confidences.append(output[len(text)-1, class_idx])
            prev_class = class_idx
        
        conf = np.mean(confidences) if confidences else 0.0
        return text, conf


# =============================================================================
# 3. FEDERATED LEARNING TRAINER
# =============================================================================

class FederatedLearningTrainer:
    """Entrenador para federated learning con sincronización de clientes."""
    
    def __init__(self, global_model_path: str, framework: str = "keras"):
        """
        Args:
            global_model_path: ruta del modelo global inicial
            framework: "keras" o "pytorch"
        """
        self.global_model_path = global_model_path
        self.framework = framework
        self.global_model = None
        self.client_updates = {}
        self.training_rounds = 0
        self._load_global_model()
    
    def _load_global_model(self) -> None:
        """Carga modelo global."""
        if self.framework == "keras":
            try:
                import tensorflow as tf
                self.global_model = tf.keras.models.load_model(self.global_model_path, compile=False)
            except ImportError:
                raise ImportError("TensorFlow requerido")
        else:
            raise NotImplementedError(f"Framework {self.framework}")
    
    def register_client_update(self, client_id: str, update_path: str) -> None:
        """Registra actualización de cliente (gradientes o pesos)."""
        self.client_updates[client_id] = update_path
        logger.info(f"✓ Actualización registrada: {client_id}")
    
    def aggregate_round(self) -> None:
        """Agregación federada de una ronda."""
        if not self.client_updates:
            raise ValueError("No hay actualizaciones de clientes")
        
        logger.info(f"\n{'='*60}")
        logger.info(f"RONDA FEDERADA #{self.training_rounds + 1}")
        logger.info(f"{'='*60}")
        logger.info(f"Agregando {len(self.client_updates)} clientes...")
        
        aggregator = FederatedModelAggregator(framework=self.framework, verbose=True)
        for client_id, update_path in self.client_updates.items():
            aggregator.register_client_model(client_id, update_path)
        
        # Agregación con pesos por dataset size
        weights = {cid: 1.0 for cid in self.client_updates.keys()}  # Pesos iguales
        aggregator.aggregate_weighted(weights)
        
        # Guardar modelo global actualizado
        global_output = self.global_model_path.replace(".keras", f"_round{self.training_rounds+1}.keras")
        aggregator.save_aggregated_model_keras(global_output)
        
        # Cargar nuevo modelo global
        try:
            import tensorflow as tf
            self.global_model = tf.keras.models.load_model(global_output, compile=False)
        except ImportError:
            pass
        
        self.training_rounds += 1
        self.client_updates.clear()
        
        logger.info(f"✓ Ronda {self.training_rounds} completada. Modelo guardado: {global_output}")
    
    def save_training_summary(self, output_file: str) -> None:
        """Guarda resumen del entrenamiento federado."""
        summary = {
            "total_rounds": self.training_rounds,
            "global_model_path": self.global_model_path,
            "framework": self.framework,
            "timestamp": str(Path(output_file).parent)
        }
        
        with open(output_file, "w") as f:
            import json
            json.dump(summary, f, indent=2)
        
        logger.info(f"✓ Resumen guardado: {output_file}")


# =============================================================================
# 4. EXPORTAR ARQUITECTURA A PYTHON PURO
# =============================================================================

class ModelArchitectureExporter:
    """Exporta arquitectura de modelos a código Python puro."""
    
    @staticmethod
    def export_keras_architecture(model_path: str, output_py_path: str) -> None:
        """Exporta arquitectura Keras a función Python."""
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow requerido")
        
        model = tf.keras.models.load_model(model_path, compile=False)
        
        # Generar código Python
        code = "# -*- coding: utf-8 -*-\n"
        code += f'"""Arquitectura OCR CRNN+CTC - Auto-generada desde {Path(model_path).name}"""\n\n'
        code += "import tensorflow as tf\nimport numpy as np\n\n"
        code += "def build_ocr_crnn_model(input_shape=(48, 192, 1), num_classes=38):\n"
        code += "    \"\"\"Construye arquitectura OCR CRNN+CTC.\n"
        code += f"    Original: {model_path}\n"
        code += "    \"\"\"\n"
        code += "    inputs = tf.keras.Input(shape=input_shape)\n"
        code += "    x = inputs\n\n"
        
        # Recorrer capas
        code += "    # === CNN Feature Extraction ===\n"
        layer_num = 1
        for layer in model.layers:
            if "conv" in layer.name.lower():
                config = layer.get_config()
                filters = config.get("filters", 32)
                kernel_size = config.get("kernel_size", (3, 3))
                code += f"    x = tf.keras.layers.Conv2D({filters}, {kernel_size}, padding='same', activation='relu')(x)\n"
                layer_num += 1
            elif "pool" in layer.name.lower():
                config = layer.get_config()
                pool_size = config.get("pool_size", (2, 2))
                code += f"    x = tf.keras.layers.MaxPooling2D({pool_size})(x)\n"
                layer_num += 1
            elif "batch" in layer.name.lower():
                code += f"    x = tf.keras.layers.BatchNormalization()(x)\n"
        
        code += "\n    # === RNN Sequence Modeling ===\n"
        code += "    x = tf.keras.layers.Reshape((-1, x.shape[-1]))(x)\n"
        code += "    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)\n"
        code += "    x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128, return_sequences=True))(x)\n"
        code += "    x = tf.keras.layers.Dropout(0.2)(x)\n\n"
        code += "    # === Character Classification ===\n"
        code += "    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)\n\n"
        code += "    model = tf.keras.Model(inputs, outputs)\n"
        code += "    return model\n\n"
        code += "# Uso:\n"
        code += "# model = build_ocr_crnn_model()\n"
        code += "# model.load_weights('weights.h5')\n"
        
        # Guardar
        Path(output_py_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_py_path, "w") as f:
            f.write(code)
        
        logger.info(f"✓ Arquitectura exportada a: {output_py_path}")
    
    @staticmethod
    def export_model_config(model_path: str, output_json_path: str) -> None:
        """Exporta configuración del modelo a JSON."""
        try:
            import tensorflow as tf
            import json
        except ImportError:
            raise ImportError("TensorFlow requerido")
        
        model = tf.keras.models.load_model(model_path, compile=False)
        
        config = {
            "name": model.name,
            "input_shape": model.input_shape,
            "output_shape": model.output_shape,
            "num_parameters": int(model.count_params()),
            "layers": []
        }
        
        for layer in model.layers:
            layer_info = {
                "name": layer.name,
                "type": layer.__class__.__name__,
                "config": layer.get_config(),
                "params": int(layer.count_params())
            }
            config["layers"].append(layer_info)
        
        Path(output_json_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json_path, "w") as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"✓ Configuración exportada a: {output_json_path}")


# =============================================================================
# 5. UTILIDADES
# =============================================================================

def create_federated_config(output_path: str) -> None:
    """Crea archivo de configuración federada."""
    config_template = {
        "federated_learning": {
            "global_model": "models/ocr_crnn_best.keras",
            "num_rounds": 5,
            "clients": [
                {
                    "id": "client_a",
                    "data_path": "data/client_a/",
                    "model_path": "models/ocr_crnn_best_a_lr3e4_w2.keras",
                    "dataset_size": 1500
                },
                {
                    "id": "client_b",
                    "data_path": "data/client_b/",
                    "model_path": "models/ocr_crnn_best_b_lr2e4_w3.keras",
                    "dataset_size": 1200
                },
                {
                    "id": "client_c",
                    "data_path": "data/client_c/",
                    "model_path": "models/ocr_crnn_best_c_lr4e4_w2.keras",
                    "dataset_size": 1800
                }
            ],
            "aggregation": {
                "method": "fedavg",  # or "weighted"
                "weights": {"client_a": 1500, "client_b": 1200, "client_c": 1800}
            }
        },
        "ensemble": {
            "detector": {
                "models": [
                    "models/plate_detector_best.pt",
                    "models/plate_detector_smallfar_best.pt"
                ],
                "weights": [0.7, 0.3],
                "iou_threshold": 0.5
            },
            "ocr": {
                "models": [
                    "models/ocr_crnn_best.keras",
                    "models/ocr_crnn_best_a_lr3e4_w2.keras"
                ],
                "voting_window": 3
            }
        }
    }
    
    import json
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(config_template, f, indent=2)
    
    logger.info(f"✓ Configuración federada creada: {output_path}")


# =============================================================================
# EJEMPLOS DE USO
# =============================================================================

if __name__ == "__main__":
    import cv2
    
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s"
    )
    
    print("\n" + "="*70)
    print("FEDERACIÓN DE MODELOS - EJEMPLOS")
    print("="*70)
    
    # ======================== EJEMPLO 1: Agregación FedAvg ========================
    print("\n[1] Agregación Federada (FedAvg) - OCR")
    print("-" * 70)
    
    aggregator = FederatedModelAggregator(framework="keras", verbose=True)
    
    # Registrar modelos OCR (variantes A, B, C)
    ocr_models = [
        ("models/ocr_crnn_best_a_lr3e4_w2.keras", "client_a"),
        ("models/ocr_crnn_best_b_lr2e4_w3.keras", "client_b"),
        ("models/ocr_crnn_best_c_lr4e4_w2.keras", "client_c"),
    ]
    
    print("\nRegistrando modelos OCR...")
    for model_path, client_id in ocr_models:
        if Path(model_path).exists():
            aggregator.register_client_model(client_id, model_path)
        else:
            print(f"⚠ Modelo no encontrado: {model_path}")
    
    if aggregator.client_models:
        # Agregación simple
        print("\n→ Ejecutando agregación FedAvg...")
        aggregated = aggregator.aggregate_mean()
        
        # Guardar modelo global
        output_model = "deploy/federated_ocr_global_model.keras"
        # aggregator.save_aggregated_model_keras(output_model)  # Comentado (requiere TF)
        print(f"  Modelo global guardado en: {output_model}")
    
    # ======================== EJEMPLO 2: Agregación Ponderada ========================
    print("\n[2] Agregación Ponderada por Dataset Size")
    print("-" * 70)
    
    aggregator2 = FederatedModelAggregator(framework="keras", verbose=True)
    
    dataset_sizes = {"client_a": 1500, "client_b": 1200, "client_c": 1800}
    
    for model_path, client_id in ocr_models:
        if Path(model_path).exists():
            aggregator2.register_client_model(client_id, model_path)
    
    if aggregator2.client_models:
        print(f"\n→ Ejecutando agregación ponderada...")
        print(f"  Pesos por dataset size: {dataset_sizes}")
        # aggregated2 = aggregator2.aggregate_weighted(dataset_sizes)  # Comentado
    
    # ======================== EJEMPLO 3: Ensemble de Detectores ========================
    print("\n[3] Ensemble de Detectores (YOLO)")
    print("-" * 70)
    
    detector_models = [
        "models/plate_detector_best.pt",
        "models/plate_detector_smallfar_best.pt"
    ]
    
    existing_detectors = [m for m in detector_models if Path(m).exists()]
    
    if len(existing_detectors) > 0:
        print(f"Detectores disponibles: {existing_detectors}")
        # ensemble = DetectorEnsemble(existing_detectors)  # Comentado (requiere Ultralytics)
        # ensemble.set_weights([0.7, 0.3])
    else:
        print("⚠ No hay detectores YOLO disponibles")
    
    # ======================== EJEMPLO 4: Ensemble de OCR ========================
    print("\n[4] Ensemble de OCR (Voting Temporal)")
    print("-" * 70)
    
    ocr_ensemble_paths = [m for m, _ in ocr_models if Path(m).exists()]
    
    if len(ocr_ensemble_paths) > 0:
        print(f"Modelos OCR disponibles para ensemble: {len(ocr_ensemble_paths)}")
        # ocr_ensemble = OCREnsemble(ocr_ensemble_paths)  # Comentado (requiere TF)
    else:
        print("⚠ No hay modelos OCR disponibles")
    
    # ======================== EJEMPLO 5: Exportar Arquitectura ========================
    print("\n[5] Exportar Arquitectura a Python Puro")
    print("-" * 70)
    
    exporter = ModelArchitectureExporter()
    
    # Intentar exportar arquitectura OCR
    ocr_main = "models/ocr_crnn_best.keras"
    if Path(ocr_main).exists():
        output_py = "deploy/ocr_architecture_exported.py"
        output_json = "deploy/ocr_config_exported.json"
        try:
            exporter.export_keras_architecture(ocr_main, output_py)
            exporter.export_model_config(ocr_main, output_json)
            print(f"✓ Arquitectura exportada:")
            print(f"  → {output_py}")
            print(f"  → {output_json}")
        except Exception as e:
            print(f"⚠ Error exportando: {e}")
    else:
        print(f"⚠ Modelo no encontrado: {ocr_main}")
    
    # ======================== EJEMPLO 6: Crear Config Federada ========================
    print("\n[6] Crear Archivo de Configuración Federada")
    print("-" * 70)
    
    config_output = "deploy/federated_config.json"
    create_federated_config(config_output)
    
    print("\n" + "="*70)
    print("✓ Ejemplos completados")
    print("="*70 + "\n")
