from pathlib import Path
import runpy


def main():
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / 'reportes' / 'resultados_metricas_20260607' / 'evaluar_pipeline_35.py'
    runpy.run_path(str(target), run_name='__main__')


if __name__ == '__main__':
    main()
