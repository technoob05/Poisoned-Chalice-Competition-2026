## Poisoned Chalice

Membership inference attacks for code models and small utilities to run Loss/PAC/Min-K% attacks against a causal LM and save results.

This repository contains attack implementations and a simple CLI entrypoint `run.py` for running experiments.

The dataset is shared on the [HuggingFace Hub](https://huggingface.co/datasets/AISE-TUDelft/Poisoned-Chalice) and supports 5 languages. Each language has a train and test split. Note that we do not train or tune these attacks so in this example we only use the test splits.

## Contents

- `run.py` — CLI wrapper that loads a model, datasets (configurable), runs selected attacks, and writes results to `results/`.
- `process.py` - CLI wrapper that processes the results, creates plots and calculates the metrics
- `Pac.py`, `Loss.py`, `MinKProbAttack.py`, `MIAttack.py` — attack implementations and helpers.
- `Dockerfile`, `requirements.txt` — container and dependency manifest to run the project.

## Prerequisites
- Docker (for the recommended containerized workflow).
- If you want to run outside Docker: Python 3.10+ and the packages from `requirements.txt`.

## Quickstart — Docker (recommended)

1. Build the image (from the project root):

```bash
docker build -t poisoned-chalice:latest .
```

2. Run the container, passing the required CLI arguments.
To generate results:

```bash
docker run --rm -v "$(pwd)/":/app/ poisoned-chalice:latest run.py --model_name <MODEL_PATH_OR_HF_NAME> --output_dir results
```

Replace `<MODEL_PATH_OR_HF_NAME>` with either a Hugging Face model identifier (`bigcode/starcoder2-3b` or `HuggingFaceTB/SmolLM3-3B-Base`) or a path to a local model directory mounted into the container.

To plot them and get the scores:

```bash
docker run --rm -v "$(pwd)/":/app/ poisoned-chalice:latest process.py --config_path <SAVED_CONFIG>
```
The plots and scores will be in the `\plots` folder.

