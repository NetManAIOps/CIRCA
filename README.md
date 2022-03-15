# CIRCA

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

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
