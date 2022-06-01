# CIRCA

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This project contains the code of baselines and simulation data generation for the KDD '22 paper, [Causal Inference-Based Root Cause Analysis for Online Service Systems with Intervention Recognition](https://doi.org/10.1145/3534678.3539041).
Experiment results can be found in [figshare](https://doi.org/10.6084/m9.figshare.19085855), where the code is corresponding to the commit 1522ddd7efd16db55e9f351fd70324501ce9134e.

## Usage

This repository contains a [Dockerfile](Dockerfile) to describe the necessary steps to setup the environment.
To install this project as a package with `pip`, R package [pcalg](build/requirements.R) has to be installed manually.

### Simulation Data Generation

```bash
python -m circa.experiment generate
```

### Simulation Study

```bash
# Explore parameter combinations
python -m circa.experiment --max-workers 16 --model-params params-sim-tune.json tune
# Explore all the datasets with pre-defined parameters
python -m circa.experiment --model-params params-sim-run.json run
# Robustness evaluation
python -m circa.experiment robustness
```

Execute `Rscript img/draw.sim.R` to produce summaries under `img/output`.
- `params-sim-run.json` is created according to `img/output/best-sim-tuning.tex`
- To create parameter template, execute the following command
```bash
python -m circa.experiment params > default.json
```
