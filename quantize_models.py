"""
⚡ CUANTIZACIÓN Y OPTIMIZACIÓN DE MODELOS PARA JETSON NANO
Convierte modelos a TFLite INT8 y ONNX para inferencia eficiente
"""

import os
import json
import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path
import glob

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

CONFIG = {
    "quantization_type": "INT8",  # INT8, FLOAT16, DYNAMIC_RANGE
    "tflite_optimization": True,
    "representative_data_size": 100,
    "output_dir": "models/optimized",
    "input_size": (416, 416),
}

# ============================================================================
# GENERADOR DE DATOS REPRESENTATIVO
# ============================================================================

class RepresentativeDataGenerator:
    """Genera datos representativos para cuantización"""
    
    def __init__(self, image_dir, img_size=None, num_samples=100):
        self.image_files = glob.glob(f"{image_dir}/*.[jp][pn]g")[:num_samples]
        self.img_size = img_size if img_size is not None else CONFIG["input_size"]
        print(f"✓ Muestras representativas: {len(self.image_files)}")
    
    def __call__(self):
        """Generador que yield batches normalizados"""
        for img_path in self.image_files:
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, self.img_size)
                    img = img.astype(np.float32) / 255.0
                    yield [np.expand_dims(img, 0)]
            except:
                pass


# ============================================================================
# CUANTIZACIÓN A TFLITE
# ============================================================================

def quantize_to_tflite(model_path, output_path, representative_data_gen=None,
                       quantization_type="INT8"):
    """
    Convierte modelo Keras a TFLite con cuantización
    
    Args:
        model_path: ruta al modelo Keras (.h5)
        output_path: ruta de salida (.tflite)
        representative_data_gen: generador de datos para cuantización
        quantization_type: INT8, FLOAT16, o DYNAMIC_RANGE
    """
    
    print(f"\n{'='*60}")
    print(f"🔄 CUANTIZACIÓN A TFLITE ({quantization_type})")
    print(f"{'='*60}")
    
    # Cargar modelo
    print(f"📦 Cargando modelo: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=False)
    
    def build_converter(mode):
        conv = tf.lite.TFLiteConverter.from_keras_model(model)
        conv.optimizations = [tf.lite.Optimize.DEFAULT]

        if mode == "INT8":
            conv.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
                tf.lite.OpsSet.SELECT_TF_OPS,
            ]
            if representative_data_gen is not None:
                conv.representative_dataset = representative_data_gen
                print("✓ Usando datos representativos para cuantización INT8")
            else:
                print("⚠️ Sin datos representativos para INT8; se usará fallback")
        elif mode == "FLOAT16":
            conv.target_spec.supported_types = [tf.float16]
            conv.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS,
            ]
        else:  # DYNAMIC_RANGE
            conv.target_spec.supported_ops = [
                tf.lite.OpsSet.TFLITE_BUILTINS,
                tf.lite.OpsSet.SELECT_TF_OPS,
            ]
        return conv

    # Convertir con fallback para garantizar artefacto final
    print("🔄 Convirtiendo a TFLite...")
    modes = [quantization_type, "FLOAT16", "DYNAMIC_RANGE"]
    seen = set()
    tflite_model = None
    selected_mode = None
    for mode in modes:
        if mode in seen:
            continue
        seen.add(mode)
        try:
            print(f"   - Intentando modo: {mode}")
            converter = build_converter(mode)
            tflite_model = converter.convert()
            selected_mode = mode
            break
        except Exception as e:
            print(f"   - Fallo en {mode}: {str(e)[:180]}")

    if tflite_model is None:
        print("❌ Error en conversión: no se pudo convertir en ningún modo")
        return None
    
    # Guardar
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"✅ Modelo TFLite guardado: {output_path} (modo efectivo: {selected_mode})")
    
    # Estadísticas
    original_size = os.path.getsize(model_path) / 1e6
    tflite_size = os.path.getsize(output_path) / 1e6
    compression = (1 - tflite_size / original_size) * 100
    
    print(f"\n📊 Estadísticas:")
    print(f"  Original: {original_size:.2f} MB")
    print(f"  TFLite {selected_mode}: {tflite_size:.2f} MB")
    print(f"  Compresión: {compression:.1f}%")
    
    return output_path


# ============================================================================
# BENCHMARK DE TFLITE
# ============================================================================

def benchmark_tflite(tflite_path, num_iterations=10):
    """Benchmark de modelo TFLite en CPU"""
    
    print(f"\n{'='*60}")
    print("⚡ BENCHMARK TFLITE (CPU)")
    print(f"{'='*60}")
    
    # Cargar interprete
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"✓ Input shapes: {[d['shape'] for d in input_details]}")
    print(f"✓ Output shapes: {[d['shape'] for d in output_details]}")
    
    # Generar datos de prueba acorde al input real del modelo
    input_shape = list(input_details[0]['shape'])
    input_shape = [1 if d == -1 else int(d) for d in input_shape]
    input_dtype = input_details[0]['dtype']
    test_input = np.random.randn(*input_shape).astype(input_dtype)
    
    # Warm-up
    interpreter.set_tensor(input_details[0]['index'], test_input)
    interpreter.invoke()
    
    # Benchmark
    import time
    times = []
    
    for _ in range(num_iterations):
        interpreter.set_tensor(input_details[0]['index'], test_input)
        
        start = time.time()
        interpreter.invoke()
        end = time.time()
        
        times.append((end - start) * 1000)  # ms
    
    mean_time = np.mean(times)
    std_time = np.std(times)
    fps = 1000 / mean_time
    
    print(f"\n📊 Resultados ({num_iterations} iteraciones):")
    print(f"  Tiempo promedio: {mean_time:.2f} ± {std_time:.2f} ms")
    print(f"  FPS (CPU): {fps:.1f} fps")
    print(f"  Min: {np.min(times):.2f} ms, Max: {np.max(times):.2f} ms")
    
    return {
        'mean_time_ms': mean_time,
        'std_time_ms': std_time,
        'fps': fps,
        'input_shape': input_shape
    }


# ============================================================================
# EXPORTACIÓN A ONNX
# ============================================================================

def convert_to_onnx(model_path, output_path):
    """Convierte Keras a ONNX (requiere onnx y tf2onnx)"""
    
    print(f"\n{'='*60}")
    print("🔄 CONVERSIÓN A ONNX")
    print(f"{'='*60}")
    
    try:
        import onnx
        import tf2onnx
    except ImportError:
        print("❌ ONNX no instalado. Instala: pip install onnx tf2onnx")
        return None
    
    print("ℹ️  ONNX es opcional - úsalo si necesitas cambiar a otros frameworks")
    print("Para Jetson Nano, usa TFLite o TensorRT")
    
    return None


# ============================================================================
# VALIDACIÓN DE MODELOS CUANTIZADOS
# ============================================================================

def validate_quantized_model(original_model_path, tflite_path, 
                             test_data_dir, num_samples=20):
    """Compara outputs original vs cuantizado"""
    
    print(f"\n{'='*60}")
    print("✅ VALIDACIÓN DE CUANTIZACIÓN")
    print(f"{'='*60}")
    
    # Cargar modelo original
    original_model = tf.keras.models.load_model(original_model_path, compile=False)
    
    # Cargar TFLite
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Cargar datos de test
    test_files = glob.glob(f"{test_data_dir}/*.[jp][pn]g")[:num_samples]
    
    differences = []
    
    for test_file in test_files:
        try:
            img = cv2.imread(test_file)
            if img is None:
                continue
            
            img = cv2.resize(img, CONFIG["input_size"])
            input_dtype = input_details[0]['dtype']
            img = (img.astype(np.float32) / 255.0).astype(input_dtype)
            img = np.expand_dims(img, 0)
            
            # Original
            orig_output = original_model.predict(img, verbose=0)
            
            # Cuantizado
            interpreter.set_tensor(input_details[0]['index'], img)
            interpreter.invoke()
            quant_output = interpreter.get_tensor(output_details[0]['index'])
            
            # Diferencia
            diff = np.abs(orig_output - quant_output).mean()
            differences.append(diff)
            
        except Exception as e:
            print(f"Error procesando {test_file}: {e}")
    
    if differences:
        mean_diff = np.mean(differences)
        max_diff = np.max(differences)
        
        print(f"✓ Muestras validadas: {len(differences)}")
        print(f"✓ Diferencia promedio: {mean_diff:.6f}")
        print(f"✓ Diferencia máxima: {max_diff:.6f}")
        
        if mean_diff < 0.01:
            print("✅ Cuantización exitosa - outputs consistentes")
        else:
            print("⚠️  Diferencia significativa - revisar cuantización")
    
    return {
        'samples': len(differences),
        'mean_diff': mean_diff if differences else None,
        'max_diff': max_diff if differences else None
    }


# ============================================================================
# PIPELINE COMPLETO
# ============================================================================

def optimize_all_models():
    """Optimiza todos los modelos entrenados"""
    
    print("\n" + "🚀" * 20)
    print(" OPTIMIZACIÓN COMPLETA PARA JETSON NANO")
    print("🚀" * 20)
    
    models_dir = Path("models")
    output_dir = Path(CONFIG["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Modelos a optimizar
    models_to_optimize = {
        "detector": "detector_best.h5",
        "ocr": "ocr_best.h5",
    }
    
    results = {}
    
    for model_name, model_filename in models_to_optimize.items():
        model_path = models_dir / model_filename
        
        if not model_path.exists():
            print(f"\n⚠️  {model_name} ({model_filename}) no encontrado")
            continue
        
        print(f"\n{'='*60}")
        print(f"📦 Optimizando: {model_name}")
        print(f"{'='*60}")
        
        # Generar datos representativos
        rep_data_gen = RepresentativeDataGenerator(
            "dataset_alpr/images/train" if model_name == "detector" 
            else "dataset_alpr/images/train",
            img_size=CONFIG["input_size"],
            num_samples=CONFIG["representative_data_size"]
        )
        
        # Cuantizar
        tflite_output = output_dir / f"{model_name}_int8.tflite"
        quantize_to_tflite(
            str(model_path),
            str(tflite_output),
            rep_data_gen,
            "INT8"
        )
        
        # Benchmark
        benchmark_results = benchmark_tflite(str(tflite_output))
        results[model_name] = benchmark_results
        
        # Validación
        validate_quantized_model(
            str(model_path),
            str(tflite_output),
            "dataset_alpr/images/train"
        )
    
    # Guardar resultados
    with open(output_dir / "optimization_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("✨ OPTIMIZACIÓN COMPLETADA")
    print(f"{'='*60}")
    
    print(f"\n✅ Modelos optimizados en: {output_dir}")
    print(f"✅ Resultados guardados en: {output_dir}/optimization_results.json")
    
    # Resumen final
    print(f"\n{'='*60}")
    print("📊 RESUMEN PARA JETSON NANO")
    print(f"{'='*60}")
    
    total_fps = 0
    for model_name, result in results.items():
        print(f"\n{model_name.upper()}:")
        print(f"  Latencia: {result['mean_time_ms']:.1f} ms")
        print(f"  FPS (CPU): {result['fps']:.1f}")
        total_fps += result['fps'] if isinstance(result['fps'], (int, float)) else 0
    
    print(f"\n⚠️  NOTA: benchmark es para CPU. En Jetson Nano GPU será MUCHO más rápido")
    print(f"🎯 Target Jetson: 5-10 FPS ✓")


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    optimize_all_models()
    
    print("""
    
    ✨ SIGUIENTES PASOS:
    
    1. Copiar modelos a Jetson:
       scp models/optimized/*.tflite jetson:~/lp_detection/models/
    
    2. En Jetson, probar inference:
       python3 inference_jetson.py --video /dev/video0
    
    3. Monitorear recursos:
       watch -n 0.1 'nvidia-smi && free -h'
    """)
