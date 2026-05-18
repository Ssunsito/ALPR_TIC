# -*- coding: utf-8 -*-
"""
EJEMPLOS PRÁCTICOS - Federación de Modelos OCR y Detectores
Adaptados para tu dataset ALPR (Automatic License Plate Recognition)
"""

import numpy as np
from pathlib import Path
import logging

# Importar desde federated_learning
from federated_learning import (
    FederatedModelAggregator,
    DetectorEnsemble,
    OCREnsemble,
    FederatedLearningTrainer,
    ModelArchitectureExporter,
    create_federated_config
)

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# =============================================================================
# RUTAS DE MODELOS (Adapta según tu estructura)
# =============================================================================

MODELS_DIR = Path("../models")

OCR_MODELS = {
    "client_a": MODELS_DIR / "ocr_crnn_best_a_lr3e4_w2.keras",
    "client_b": MODELS_DIR / "ocr_crnn_best_b_lr2e4_w3.keras",
    "client_c": MODELS_DIR / "ocr_crnn_best_c_lr4e4_w2.keras",
    "main": MODELS_DIR / "ocr_crnn_best.keras",
}

DETECTOR_MODELS = {
    "standard": MODELS_DIR / "plate_detector_best.pt",
    "small_far": MODELS_DIR / "plate_detector_smallfar_best.pt",
}

OPTIMIZED_MODELS = {
    "detector_tflite": MODELS_DIR / "optimized" / "detector_int8.tflite",
    "ocr_tflite": MODELS_DIR / "optimized" / "ocr_crnn_ctc_int8.tflite",
}


# =============================================================================
# CASO DE USO 1: AGREGACIÓN FedAvg (Múltiples clientes OCR)
# =============================================================================

def use_case_1_fedavg_aggregation():
    """
    Escenario: Tienes 3 clientes con modelos OCR entrenados localmente
    (A, B, C) y quieres combinarlos en un modelo global federado.
    
    Aplicación real: Cada oficina municipal entrena OCR con sus imágenes
    locales, y se crea un modelo global combinado.
    """
    print("\n" + "="*70)
    print("CASO 1: Agregación Federada (FedAvg) - OCR")
    print("="*70)
    
    aggregator = FederatedModelAggregator(framework="keras", verbose=True)
    
    # Registrar modelos de clientes
    print("\n[Paso 1] Registrando modelos de clientes...")
    for client_id, model_path in OCR_MODELS.items():
        if client_id != "main" and model_path.exists():
            aggregator.register_client_model(client_id, str(model_path))
            print(f"  ✓ {client_id}: {model_path.name}")
        elif client_id == "main" and model_path.exists():
            print(f"  ℹ {client_id}: Reservado para modelo global")
    
    if not aggregator.client_models:
        print("⚠ No hay modelos disponibles")
        return
    
    # Agregación
    print("\n[Paso 2] Agregación simple (FedAvg)...")
    print("  Cada cliente tiene igual peso (1.0)")
    try:
        aggregated_weights = aggregator.aggregate_mean()
        print(f"  ✓ Pesos agregados: {len(aggregated_weights):,} parámetros")
    except ImportError as e:
        print(f"  ⚠ TensorFlow no disponible: {e}")
        return
    
    # Guardar
    print("\n[Paso 3] Guardando modelo global agregado...")
    output_path = Path("federated_ocr_global_v1.keras")
    try:
        aggregator.save_aggregated_model_keras(str(output_path))
        print(f"  ✓ Modelo guardado: {output_path}")
    except Exception as e:
        print(f"  ⚠ Error: {e}")


# =============================================================================
# CASO DE USO 2: AGREGACIÓN PONDERADA (por tamaño de dataset)
# =============================================================================

def use_case_2_weighted_aggregation():
    """
    Escenario: Los 3 clientes tienen datasets de diferentes tamaños.
    Cliente A: 1500 imágenes
    Cliente B: 1200 imágenes
    Cliente C: 1800 imágenes
    
    → Pondera agregación por tamaño (cliente C más peso)
    """
    print("\n" + "="*70)
    print("CASO 2: Agregación Ponderada (por Dataset Size)")
    print("="*70)
    
    aggregator = FederatedModelAggregator(framework="keras", verbose=True)
    
    # Dataset sizes (ejemplo)
    dataset_sizes = {
        "client_a": 1500,
        "client_b": 1200,
        "client_c": 1800,
    }
    
    # Registrar
    print("\n[Paso 1] Registrando modelos con dataset sizes...")
    for client_id, model_path in OCR_MODELS.items():
        if client_id != "main" and model_path.exists():
            aggregator.register_client_model(client_id, str(model_path))
            size = dataset_sizes.get(client_id, 0)
            print(f"  ✓ {client_id}: {size} imágenes")
    
    if not aggregator.client_models:
        print("⚠ No hay modelos disponibles")
        return
    
    # Agregación ponderada
    print("\n[Paso 2] Agregación ponderada...")
    total_size = sum(dataset_sizes.values())
    weights_normalized = {
        cid: size / total_size 
        for cid, size in dataset_sizes.items()
    }
    print(f"  Pesos: {weights_normalized}")
    
    try:
        aggregated = aggregator.aggregate_weighted(dataset_sizes)
        print(f"  ✓ Pesos agregados")
    except ImportError as e:
        print(f"  ⚠ TensorFlow no disponible: {e}")
        return
    
    # Guardar
    print("\n[Paso 3] Guardando modelo ponderado...")
    output_path = Path("federated_ocr_global_weighted.keras")
    try:
        aggregator.save_aggregated_model_keras(str(output_path))
        print(f"  ✓ Modelo guardado: {output_path}")
    except Exception as e:
        print(f"  ⚠ Error: {e}")


# =============================================================================
# CASO DE USO 3: ENSEMBLE DE DETECTORES (Voto mayoritario)
# =============================================================================

def use_case_3_detector_ensemble():
    """
    Escenario: Tienes 2 detectores YOLO
    - plate_detector_best: Para placas normales
    - plate_detector_smallfar: Para placas pequeñas/lejanas
    
    Usa ambos en paralelo, combine predicciones por voto.
    """
    print("\n" + "="*70)
    print("CASO 3: Ensemble de Detectores (Voto)")
    print("="*70)
    
    detector_paths = [
        str(DETECTOR_MODELS["standard"]),
        str(DETECTOR_MODELS["small_far"])
    ]
    
    existing = [p for p in detector_paths if Path(p).exists()]
    
    if len(existing) < 2:
        print(f"⚠ Se necesitan 2 detectores, disponibles: {len(existing)}")
        return
    
    print(f"\n[Paso 1] Cargando detectores...")
    try:
        ensemble = DetectorEnsemble(existing, framework="pytorch", device="cpu")
        print(f"  ✓ {len(ensemble.models)} detectores cargados")
    except ImportError as e:
        print(f"  ⚠ Ultralytics YOLO no disponible: {e}")
        return
    
    # Configurar pesos
    print(f"\n[Paso 2] Configurando pesos...")
    weights = [0.7, 0.3]  # 70% detector estándar, 30% small/far
    ensemble.set_weights(weights)
    
    # Predicción en dummy image
    print(f"\n[Paso 3] Predicción en imagen de prueba...")
    dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    try:
        detections = ensemble.predict(dummy_image, conf_threshold=0.5)
        print(f"  ✓ Detecciones: {len(detections)}")
        for det in detections[:3]:
            print(f"    - Box: ({det['x1']:.0f}, {det['y1']:.0f}) - ({det['x2']:.0f}, {det['y2']:.0f}), "
                  f"Conf: {det['conf']:.2f}, Votos: {det.get('num_votes', 1)}")
    except Exception as e:
        print(f"  ℹ No se pueden hacer predicciones aquí: {type(e).__name__}")
        print(f"    Usa con imagen real: ensemble.predict(cv2.imread('placa.jpg'))")


# =============================================================================
# CASO DE USO 4: ENSEMBLE DE OCR (Voting Temporal)
# =============================================================================

def use_case_4_ocr_ensemble():
    """
    Escenario: Múltiples modelos OCR (variantes A, B, C)
    
    → OCR ensemble con voting temporal
      - Cada modelo predice el texto
      - Vota por consenso (mayoría gana)
      - Historial de N frames para estabilidad
    """
    print("\n" + "="*70)
    print("CASO 4: Ensemble de OCR (Voting Temporal)")
    print("="*70)
    
    ocr_paths = [
        str(OCR_MODELS["main"]),
        str(OCR_MODELS["client_a"]),
        str(OCR_MODELS["client_b"]),
    ]
    
    existing = [p for p in ocr_paths if Path(p).exists()]
    
    if len(existing) < 2:
        print(f"⚠ Se necesitan 2+ modelos OCR, disponibles: {len(existing)}")
        return
    
    print(f"\n[Paso 1] Cargando modelos OCR...")
    try:
        ocr_ensemble = OCREnsemble(existing, voting_window=3)
        print(f"  ✓ {len(ocr_ensemble.models)} modelos OCR cargados")
        print(f"  ℹ Ventana de voting temporal: 3 frames")
    except ImportError as e:
        print(f"  ⚠ TensorFlow no disponible: {e}")
        return
    
    # Simular predicciones en 5 frames
    print(f"\n[Paso 2] Simulando predicciones en 5 frames...")
    dummy_plate = np.random.randint(0, 255, (48, 192, 1), dtype=np.uint8)
    
    for frame in range(1, 6):
        try:
            text, conf = ocr_ensemble.predict(dummy_plate)
            print(f"  Frame {frame}: {text} (conf: {conf:.2f})")
        except Exception as e:
            print(f"  Frame {frame}: Error - {type(e).__name__}")
            break
    
    print(f"\n[Info] Historial de voting: {ocr_ensemble.voting_history}")


# =============================================================================
# CASO DE USO 5: FEDERATED LEARNING TRAINING
# =============================================================================

def use_case_5_federated_training():
    """
    Escenario: Entrenamiento federado multi-ronda
    
    Ronda 1: Clientes A, B, C entrenan localmente → Agregan
    Ronda 2: Clientes A, B, C descargan modelo global, entrenan → Agregan
    ...
    """
    print("\n" + "="*70)
    print("CASO 5: Entrenamiento Federado Multi-Ronda")
    print("="*70)
    
    global_model = str(OCR_MODELS["main"])
    
    if not Path(global_model).exists():
        print(f"⚠ Modelo global no encontrado: {global_model}")
        return
    
    print(f"\n[Paso 1] Inicializando entrenador federado...")
    try:
        trainer = FederatedLearningTrainer(
            global_model_path=global_model,
            framework="keras"
        )
        print(f"  ✓ Entrenador listo")
    except ImportError as e:
        print(f"  ⚠ TensorFlow no disponible: {e}")
        return
    
    # Simular rondas
    print(f"\n[Paso 2] Simulando 2 rondas de entrenamiento...")
    
    for round_num in range(1, 3):
        print(f"\n  --- RONDA {round_num} ---")
        
        # Clientes entrenan localmente y suben actualizaciones
        print(f"  Clientes entrenando localmente...")
        for client_id, model_path in OCR_MODELS.items():
            if client_id != "main" and model_path.exists():
                # En práctica: Cliente entrena en su data local
                # Aquí simulamos que actualiza su modelo
                trainer.register_client_update(client_id, str(model_path))
                print(f"    ✓ {client_id} actualización registrada")
        
        # Agregación
        print(f"  Agregando actualizaciones...")
        try:
            trainer.aggregate_round()
        except Exception as e:
            print(f"    ⚠ Error: {e}")
            break


# =============================================================================
# CASO DE USO 6: EXPORTAR ARQUITECTURA A PYTHON PURO
# =============================================================================

def use_case_6_export_architecture():
    """
    Escenario: Exportar arquitectura OCR a código Python puro
    
    → Genera archivo .py con función build_model()
    → Compatible sin dependencias externas (solo NumPy)
    """
    print("\n" + "="*70)
    print("CASO 6: Exportar Arquitectura OCR a Python Puro")
    print("="*70)
    
    model_path = str(OCR_MODELS["main"])
    
    if not Path(model_path).exists():
        print(f"⚠ Modelo no encontrado: {model_path}")
        return
    
    exporter = ModelArchitectureExporter()
    
    print(f"\n[Paso 1] Exportando arquitectura Keras...")
    output_py = Path("ocr_model_architecture.py")
    output_json = Path("ocr_model_config.json")
    
    try:
        exporter.export_keras_architecture(model_path, str(output_py))
        exporter.export_model_config(model_path, str(output_json))
        
        print(f"  ✓ Arquitectura Python: {output_py}")
        print(f"  ✓ Configuración JSON: {output_json}")
        
        # Mostrar contenido
        print(f"\n[Paso 2] Vista previa (primeras líneas):")
        with open(output_py, "r") as f:
            lines = f.readlines()[:15]
            for line in lines:
                print(f"  {line.rstrip()}")
        
    except ImportError as e:
        print(f"  ⚠ TensorFlow no disponible: {e}")


# =============================================================================
# CASO DE USO 7: CREAR CONFIGURACIÓN FEDERADA COMPLETA
# =============================================================================

def use_case_7_create_federated_config():
    """
    Escenario: Generar archivo de configuración JSON para gestionar
    todo el sistema federado.
    """
    print("\n" + "="*70)
    print("CASO 7: Generar Configuración Federada Completa")
    print("="*70)
    
    config_path = Path("federated_config_complete.json")
    
    print(f"\n[Paso 1] Creando archivo de configuración...")
    try:
        create_federated_config(str(config_path))
        
        print(f"  ✓ Archivo creado: {config_path}")
        
        # Mostrar contenido
        import json
        with open(config_path, "r") as f:
            config = json.load(f)
        
        print(f"\n[Paso 2] Contenido (resumen):")
        print(f"  - Rondas de FL: {config['federated_learning']['num_rounds']}")
        print(f"  - Clientes: {len(config['federated_learning']['clients'])}")
        for client in config['federated_learning']['clients']:
            print(f"    • {client['id']}: {client['dataset_size']} imágenes")
        
        print(f"  - Detectores en ensemble: {len(config['ensemble']['detector']['models'])}")
        print(f"  - Modelos OCR en ensemble: {len(config['ensemble']['ocr']['models'])}")
        
    except Exception as e:
        print(f"  ⚠ Error: {e}")


# =============================================================================
# FLUJO COMPLETO: Agregación + Ensemble + Exportación
# =============================================================================

def complete_workflow():
    """
    Flujo completo:
    1. Agregación FedAvg de 3 modelos OCR
    2. Ensemble de detectores
    3. Exportar arquitectura
    4. Crear configuración
    """
    print("\n" + "="*80)
    print("FLUJO COMPLETO: Federación + Ensemble + Exportación")
    print("="*80)
    
    print("\n[Fase 1/4] Agregación Federada...")
    use_case_1_fedavg_aggregation()
    
    print("\n[Fase 2/4] Ensemble de Detectores...")
    use_case_3_detector_ensemble()
    
    print("\n[Fase 3/4] Exportar Arquitectura...")
    use_case_6_export_architecture()
    
    print("\n[Fase 4/4] Crear Configuración...")
    use_case_7_create_federated_config()
    
    print("\n" + "="*80)
    print("✓ Flujo completo finalizado")
    print("="*80)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys
    
    print("\n" + "█"*70)
    print("FEDERATED LEARNING - EJEMPLOS PRÁCTICOS")
    print("█"*70)
    
    print("\nCasos de uso disponibles:")
    print("  1. Agregación FedAvg (OCR)")
    print("  2. Agregación Ponderada (OCR)")
    print("  3. Ensemble de Detectores")
    print("  4. Ensemble de OCR (Voting)")
    print("  5. Entrenamiento Federado Multi-Ronda")
    print("  6. Exportar Arquitectura a Python")
    print("  7. Crear Configuración Federada")
    print("  complete. Ejecutar flujo completo")
    
    # Verificar argumentos
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = "complete"
    
    # Mapeo
    cases = {
        "1": use_case_1_fedavg_aggregation,
        "2": use_case_2_weighted_aggregation,
        "3": use_case_3_detector_ensemble,
        "4": use_case_4_ocr_ensemble,
        "5": use_case_5_federated_training,
        "6": use_case_6_export_architecture,
        "7": use_case_7_create_federated_config,
        "complete": complete_workflow,
    }
    
    if choice in cases:
        print(f"\nEjecutando: {choice}\n")
        cases[choice]()
    else:
        print(f"\n⚠ Opción no válida: {choice}")
        print("Usa: python federated_learning_examples.py [1-7|complete]")
    
    print("\n" + "█"*70 + "\n")
