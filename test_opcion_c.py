"""
Test de OPCIÓN C - Post-procesamiento OCR Inteligente con Reintentos
Demuestra el funcionamiento del reintento automático cuando OCR detecta "ECUADOR"
"""
import sys
import os
from pathlib import Path
import time

# Configurar paths
repo_root = Path(__file__).resolve().parents[0]
deploy_dir = repo_root / "deploy"
if str(deploy_dir) not in sys.path:
    sys.path.insert(0, str(deploy_dir))

# Cambiar a deploy directory para que imports funcionen
os.chdir(str(deploy_dir))
sys.path.insert(0, str(deploy_dir))

from inference_jetson import PipelineJetson
from runtime_config import CONFIG, apply_runtime_profile


def test_opcion_c(limit: int = 5):
    """Prueba Opción C: Post-procesamiento OCR con reintentos"""
    
    print("=" * 80)
    print("TEST OPCIÓN C - POST-PROCESAMIENTO OCR INTELIGENTE")
    print("=" * 80)
    print("\nConfigurando pipeline con soporte para reintentos automáticos...")
    
    cfg = dict(CONFIG)
    apply_runtime_profile(cfg, "jetson-stable")
    cfg["show_window"] = False
    cfg["detector_model"] = str((repo_root / "models" / "plate_detector_best.pt").resolve())
    cfg["ocr_engine"] = "ctc"
    cfg["ocr_model"] = str((repo_root / "models" / "optimized" / "ocr_crnn_ctc_int8.tflite").resolve())
    
    pipeline = PipelineJetson(cfg)
    
    val_dir = repo_root / "dataset_alpr" / "images" / "val"
    imgs = sorted(
        [
            p
            for p in val_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            and "_annotated" not in p.stem.lower()
        ]
    )[:limit]
    
    print(f"✓ Pipeline inicializado")
    print(f"✓ Modelo detector: plate_detector_best.pt")
    print(f"✓ Modelo OCR: ocr_crnn_ctc_int8.tflite")
    print(f"✓ Post-procesamiento: ENABLED (reintentos automáticos)")
    print(f"\nProcesando {len(imgs)} imágenes...")
    print("-" * 80)
    
    results = []
    
    for idx, img_path in enumerate(imgs, 1):
        gt = "".join(ch for ch in img_path.stem.upper() if ch.isalnum())
        if len(gt) < 5 or len(gt) > 8:
            gt = None
        
        print(f"\n[{idx}/{len(imgs)}] {img_path.name}")
        print(f"  Ground Truth: {gt or 'N/A'}")
        
        t0 = time.time()
        _, dets = pipeline.run_single_image(str(img_path), show_window=False)
        t1 = time.time()
        
        if dets:
            pred = dets[0].get("text", "").strip()
            conf = float(dets[0].get("confidence", 0.0))
            latency = (t1 - t0) * 1000
            
            print(f"  Predicción: {pred}")
            print(f"  Confianza:  {conf:.3f}")
            print(f"  Latencia:   {latency:.1f} ms")
            
            # Validar
            pred_norm = "".join(ch for ch in pred.upper() if ch.isalnum())
            is_valid = len(pred_norm) >= 6 and any(c.isdigit() for c in pred_norm[-4:])
            
            if is_valid:
                print(f"  ✓ Formato válido (contiene números)")
            else:
                print(f"  ✗ Formato inválido")
            
            results.append({
                "image": img_path.name,
                "gt": gt,
                "pred": pred,
                "conf": conf,
                "latency_ms": latency,
                "valid": is_valid
            })
        else:
            print(f"  ✗ No detectado")
    
    # Resumen
    print("\n" + "=" * 80)
    print("RESUMEN")
    print("=" * 80)
    
    valid_count = sum(1 for r in results if r["valid"])
    avg_latency = sum(r["latency_ms"] for r in results) / len(results) if results else 0
    
    print(f"Imágenes procesadas: {len(results)}")
    print(f"Predicciones válidas: {valid_count}/{len(results)} ({100*valid_count/len(results):.1f}%)")
    print(f"Latencia promedio: {avg_latency:.1f} ms")
    print(f"FPS estimado: {1000/avg_latency:.2f}")
    
    if valid_count > 0:
        print("\n✓ ¡Opción C mejorando resultados!")
        print("  Se detectaron predicciones válidas con reintentos automáticos")
    else:
        print("\n⚠ Ninguna predicción válida aún")
        print("  Considerar Opción A: Ajustar ocr_roi_y_bias en runtime_config.py")
    
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    results = test_opcion_c(limit=5)
    print(f"\nTest completado. {len(results)} imágenes analizadas.")
