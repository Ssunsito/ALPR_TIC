from pathlib import Path
import runpy


def main():
    repo_root = Path(__file__).resolve().parents[1]
    target = repo_root / 'deploy' / 'inference_jetson.py'
    runpy.run_path(str(target), run_name='__main__')


if __name__ == '__main__':
    main()
