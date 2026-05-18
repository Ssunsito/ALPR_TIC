# -*- coding: utf-8 -*-
"""
Script Simple - Convertir Modelos OCR a Python
Usa este script para convertir tus modelos .keras a archivos .py compartibles
"""

import sys
from pathlib import Path

# Agregar deploy al path
sys.path.insert(0, str(Path(__file__).parent))

from model_to_python_converter import convert_ocr_model_to_python
import logging

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    """Interfaz principal para convertir modelos."""
    
    print("\n" + "="*70)
    print("CONVERTIDOR: Modelo Keras → Archivo Python Compartible")
    print("="*70)
    
    # Opciones predefinidas
    modelos_disponibles = {
        "1": ("../models/ocr_crnn_best.keras", "ocr_crnn_best.py"),
        "2": ("../models/ocr_crnn_best_a_lr3e4_w2.keras", "ocr_crnn_best_a.py"),
        "3": ("../models/ocr_crnn_best_b_lr2e4_w3.keras", "ocr_crnn_best_b.py"),
        "4": ("../models/ocr_crnn_best_c_lr4e4_w2.keras", "ocr_crnn_best_c.py"),
    }
    
    print("\nModelos OCR disponibles:")
    for key, (modelo_path, output_name) in modelos_disponibles.items():
        ruta = Path(modelo_path)
        existe = "✓" if ruta.exists() else "✗"
        print(f"  {key}. {existe} {Path(modelo_path).name}")
    
    print("\n  5. Ruta personalizada")
    print("  0. Salir")
    
    choice = input("\nSelecciona modelo (0-5): ").strip()
    
    if choice == "0":
        print("Cancelado.\n")
        return
    
    if choice in modelos_disponibles:
        modelo_path, output_name = modelos_disponibles[choice]
        output_path = Path(__file__).parent / output_name
    elif choice == "5":
        modelo_path = input("\nRuta modelo (.keras): ").strip()
        output_name = input("Nombre salida (.py): ").strip()
        output_path = Path(__file__).parent / output_name
    else:
        print("❌ Opción no válida\n")
        return
    
    # Verificar modelo existe
    modelo_path = Path(modelo_path)
    if not modelo_path.exists():
        print(f"❌ Modelo no encontrado: {modelo_path}\n")
        return
    
    # Resolver ruta relativa
    if not modelo_path.is_absolute():
        modelo_path = (Path(__file__).parent / modelo_path).resolve()
    
    print(f"\n{'='*70}")
    print(f"Convertiendo...")
    print(f"  Modelo: {modelo_path.name}")
    print(f"  Salida: {output_path.name}")
    print(f"{'='*70}")
    
    try:
        # Mostrar opciones
        include_weights = input("\n¿Incluir pesos en archivo? (s/n, default: s): ").strip().lower()
        include_weights = include_weights != "n"
        
        if include_weights:
            print("\n⚠ Archivo será GRANDE (~30-100 MB)")
            print("   Pero será completamente independiente (sin necesidad de archivo .keras)")
        else:
            print("\n✓ Archivo será pequeño (~5 MB)")
            print("   Pero necesitará archivo .keras aparte para pesos")
        
        confirm = input("\nProceder? (s/n): ").strip().lower()
        if confirm != "s":
            print("Cancelado.\n")
            return
        
        print("\n⏳ Procesando...")
        convert_ocr_model_to_python(
            str(modelo_path),
            str(output_path),
            include_weights=include_weights
        )
        
        file_size = output_path.stat().st_size / (1024*1024)
        print(f"\n✓ ÉXITO")
        print(f"  Archivo: {output_path}")
        print(f"  Tamaño: {file_size:.1f} MB")
        
        print(f"\nAhora puedes:")
        print(f"  1. Compartir el archivo: {output_path.name}")
        print(f"  2. Usar en otro proyecto:")
        print(f"     from {output_path.stem} import load_ocr_crnn")
        print(f"     model = load_ocr_crnn()")
        print(f"     predicciones = model.predict(imagen_normalizada)")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nCancelado por usuario.\n")
    except Exception as e:
        print(f"\n❌ Error fatal: {e}\n")
