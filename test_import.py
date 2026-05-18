import sys
from pathlib import Path

# Setup paths
repo_root = Path(__file__).resolve().parents[0]
deploy_dir = repo_root / "deploy"
sys.path.insert(0, str(deploy_dir))

print(f"Testing imports from: {deploy_dir}")
print()

# Test import
try:
    from ocr_postprocessor import postprocess_ocr_prediction, validate_plate_format
    print("✓ ocr_postprocessor imported successfully")
    print(f"  postprocess_ocr_prediction: {postprocess_ocr_prediction}")
    print(f"  validate_plate_format: {validate_plate_format}")
except ImportError as e:
    print(f"✗ ImportError: {e}")
except Exception as e:
    print(f"✗ Exception: {e}")

print()

# Test function
try:
    result = validate_plate_format("AAB-4475")
    print(f"✓ validate_plate_format('AAB-4475') = {result}")
    
    result = postprocess_ocr_prediction("ECUADOR")
    print(f"✓ postprocess_ocr_prediction('ECUADOR') = {result}")
except Exception as e:
    print(f"✗ Error calling functions: {e}")
