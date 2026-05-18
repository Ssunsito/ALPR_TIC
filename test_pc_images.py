import sys
from pathlib import Path
import importlib

repo = Path(__file__).resolve().parent
sys.path.insert(0, str(repo / 'deploy'))

mod = importlib.import_module('inference_jetson_pc')

sys.argv = [
    'inference_jetson_pc.py',
    '--input', str(repo / 'dataset_alpr' / 'images' / 'val'),
    '--output-dir', str(repo / 'outputs' / 'pc_test_py'),
    '--limit', '3',
    '--no-window',
]

mod.main()
