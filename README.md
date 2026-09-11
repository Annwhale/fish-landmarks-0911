![Welcome, and thank you for reviewing my data and code. Have a wonderful day!](assets/welcome-0911.gif)

# Fish Landmarks

Tools for fish landmark curation, evaluation and visualization.

2,564 images · 11 landmarks · 9 taxon groups

## Setup

Python 3.11 with PyTorch.

```bash
python -m pip install -e .
python -m pip install pytest scipy scikit-learn seaborn pymupdf
```

See `environment.yml` for the experiment environment.

## Quick start

Place the dataset in `data/benchmark_v1`, or link an extracted submission package:

```bash
python run-0911.py attach /path/to/FishLandmarks_Scientific_Data_Submission_v1
python run-0911.py quickstart data/benchmark_v1
python -m pytest -q
```

Data-dependent tests skip when inputs are missing. Set `FISH_SOURCE_ROOT` for source-annotation tests.

## Commands

```bash
python run-0911.py train-yolo --help
python run-0911.py train-heatmap --help
python run-0911.py evaluate-yolo --help
python run-0911.py evaluate-heatmap --help
```

## Contents

- `src/fishlandmark` — core tools
- `scripts` — processing and evaluation
- `figures` — plotting code and source tables
- `config` — experiment settings
- `docs` — [landmark definitions](docs/landmark-definitions-0911.csv) and data dictionary

## License

Project code: [MIT](LICENSE). See [third-party notices](docs/third-party-notices.md) and [development notes](docs/development-notes.md).
