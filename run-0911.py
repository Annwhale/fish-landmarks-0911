#!/usr/bin/env python3
# ><(((o>  公开命令入口；内部模块名保持兼容。
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COMMANDS = {
    'quickstart': 'examples/quickstart.py',
    'train-yolo': 'scripts/train_yolo_pose.py',
    'train-heatmap': 'scripts/train_heatmap_pose.py',
    'evaluate-yolo': 'scripts/evaluate_yolo_pose.py',
    'evaluate-heatmap': 'scripts/evaluate_heatmap_pose.py',
    'evaluate-locateanything': 'scripts/evaluate_locateanything_zero_shot.py',
    'build-benchmark': 'scripts/build_benchmark.py',
    'export-benchmark': 'scripts/export_benchmark_package.py',
}


def attach(package):
    package = package.resolve(strict=True)
    validation = package / 'CODE_AND_REPRODUCIBILITY/validation'
    dataset = package / 'DATA_REPOSITORY_UPLOAD/benchmark_v1'
    links = {'data/benchmark_v1': dataset, 'data/derived': dataset / 'tables',
             'qa': validation / 'qa', 'evidence': validation / 'evidence',
             'models/evaluation': validation / 'model_evaluation',
             'data/experiments/protocols': validation / 'protocols'}
    for name, source in links.items():
        target = ROOT / name
        if not source.is_dir():
            raise SystemExit(f'Missing input: {source}')
        if (target.exists() or target.is_symlink()) and target.resolve() != source:
            raise SystemExit(f'Existing path preserved: {target}')
    for name, source in links.items():
        target = ROOT / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.is_symlink():
            target.symlink_to(source, target_is_directory=True)
        print(name)


def main():
    parser = argparse.ArgumentParser(description='Fish landmark tools')
    parser.add_argument('command', choices=['attach', *COMMANDS])
    parser.add_argument('arguments', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.command == 'attach':
        if len(args.arguments) != 1:
            parser.error('attach requires the extracted submission package directory')
        attach(Path(args.arguments[0]))
        return
    env = os.environ.copy()
    env['PYTHONPATH'] = str(ROOT / 'src') + os.pathsep + env.get('PYTHONPATH', '')
    raise SystemExit(subprocess.call([sys.executable, str(ROOT / COMMANDS[args.command]), *args.arguments], env=env))


if __name__ == '__main__':
    main()
