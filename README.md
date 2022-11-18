# CIRCA

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Actions](https://github.com/NetManAIOps/CIRCA/actions/workflows/actions.yml/badge.svg)](https://github.com/NetManAIOps/CIRCA/actions/workflows/actions.yml)
[![PyPI version](https://badge.fury.io/py/circa-clue.svg)](https://badge.fury.io/py/circa-clue)
[![Downloads](https://pepy.tech/badge/circa-clue)](https://pepy.tech/project/circa-clue)

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

### Toolbox

CIRCA is designed as a toolbox with a set of interfaces.

#### Basic

Each root cause analysis algorithm is separated into two steps, namely *graph construction* and *scoring*.

The graph construction step should implement `circa.graph.GraphFactory`.
`GraphFactory.create` takes data for analysis (an instance of `circa.model.case.CaseData`) and timestamp (`float`) when the algorithm is triggered.
The output is a graph (an instance of `circa.model.graph.Graph`) for the fault under analysis.

The scoring step contains a sequence of scorers (instances of `circa.alg.base.Scorer`).
`Scorer.score` of each scorer needs the following information:

- The graph produced in the graph construction step,
- data for analysis (an instance of `circa.model.case.CaseData`),
- timestamp (`float`) when the algorithm is triggered, and
- (optional) output of the previous scorer.

`Scorer.score` will generate a mapping from a node in the input graph to its score (`circa.alg.base.Score`).
The design of the scorer sequence enables reusing scorers, i.e., two algorithms can share one scorer as a common step.
Note that a scorer may drop some nodes in the input graph, performing as a filter.

`circa.alg.common` provides some common utilizations.
For example, `circa.alg.common.Model` combines a graph factory and a sequence of scorers as a whole with optional names.
`Model.analyze` will forward data and timestamp for them and produce an ordered sequence of scores.
`circa.alg.common.evaluate` will further evaluate a model with a set of cases (instances of `circa.model.case.Case`, each of which combines data and the corresponding answers).

```python
"""
An example showing the basic usage of CIRCA
"""
from collections import defaultdict
from typing import Dict
from typing import Sequence
from typing import Tuple

import networkx as nx
from sklearn.linear_model import LinearRegression

from circa.alg.ci import RHTScorer
from circa.alg.ci.anm import ANMRegressor
from circa.alg.common import Model
from circa.graph.common import StaticGraphFactory
from circa.model.case import CaseData
from circa.model.data_loader import MemoryDataLoader
from circa.model.graph import MemoryGraph
from circa.model.graph import Node


latency = Node("DB", "Latency")
traffic = Node("DB", "Traffic")
saturation = Node("DB", "Saturation")
# circa.model.graph.MemoryGraph is derived from circa.model.graph.Graph
graph = MemoryGraph(
    nx.DiGraph(
        {
            traffic: [latency, saturation],
            saturation: [latency],
        }
    )
)

# 1. Assemble an algorithm
# circa.graph.common.StaticGraphFactory is derived from circa.graph.GraphFactory
graph_factory = StaticGraphFactory(graph)
scorers = [
    # circa.alg.ci.RHTScorer is derived from circa.alg.common.DecomposableScorer,
    # which is further derived from circa.alg.base.Scorer
    RHTScorer(regressor=ANMRegressor(regressor=LinearRegression())),
]
model = Model(graph_factory=graph_factory, scorers=scorers)

# 2. Prepare data
mock_data = {
    latency: (10, 12, 11, 9, 100, 90),
    traffic: (100, 110, 90, 105, 200, 150),
    saturation: (5, 4, 5, 6, 90, 85),
}
mock_data_with_time: Dict[str, Dict[str, Sequence[Tuple[float, float]]]] = defaultdict(
    dict
)
for node, values in mock_data.items():
    mock_data_with_time[node.entity][node.metric] = [
        (index * 60, value) for index, value in enumerate(values)
    ]
data = CaseData(
    # circa.model.data_loader.MemoryDataLoader is derived from
    # circa.model.data_loader.DataLoader, which manages data with configurations
    data_loader=MemoryDataLoader(mock_data_with_time),
    sli=latency,
    detect_time=240,
    lookup_window=4,
    detect_window=2,
)

# 3. Conduct root cause analysis one minute after a fault is detected
print(model.analyze(data=data, current=data.detect_time + 60))
```

#### Advanced

`circa.experiment` supports comparison among models and parameter exploration, as mentioned for the simulation study.
To conduct experiments with your own dataset, start from the following code named `example.py`.
Execute `python -m example -s run-new --output-dir output/test --report-dir report/test` and find the report in `report/test/report.csv`.
Find more command line parameters with `python -m example -h`.

```python
"""
An example showing the advanced usage of CIRCA
"""
import argparse
import logging
import os
from typing import List

from circa.experiment import comparison
from circa.experiment.comparison.models import get_models
from circa.experiment.__main__ import get_parser
from circa.graph.structural import StructuralGraph
from circa.model.case import Case
from circa.utils import silence_third_party


BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def run(args: argparse.Namespace):
    """
    Evaluate multiple models
    """
    data_dir: str = args.data_dir
    report_dir: str = args.report_dir
    os.makedirs(report_dir, exist_ok=True)

    logger = logging.getLogger(__package__)

    logger.info("Loading from %s", data_dir)
    # TODO: Prepare your data with answers here
    cases: List[Case] = []

    models, graph_factories = get_models(
        # TODO: Configure your own structural graph here
        # structural_graph_params=dict(
        #     structural_graph=StructuralGraph(filename="tests/alg/sgraph/index.yml"),
        # ),
        params=args.model_params,
        seed=args.seed,
        cuda=args.cuda,
        max_workers=1,
    )

    logger.info("Start running on %s with #models=%d", data_dir, len(models))
    comparison.run(
        models=models,
        cases=cases,
        graph_factories=graph_factories,
        output_dir=args.output_dir,
        report_filename=os.path.join(report_dir, "report.csv"),
        max_workers=1 if args.cuda else args.max_workers,
    )


def wrap_parsers(subparsers: argparse._SubParsersAction):
    """
    Add argparser for your own experiments
    """
    parser_params = dict(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser = argparse.ArgumentParser(add_help=False, **parser_params)
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.join(BASE_DIR, "data"),
        help="Data directory",
    )

    parser_run: argparse.ArgumentParser = subparsers.add_parser(
        "run-new",
        parents=[parser],
        help="Explore all combinations of model parameters",
        **parser_params,
    )
    parser_run.add_argument(
        "--output-dir", type=str, default="output", help="Output directory"
    )
    parser_run.add_argument(
        "--report-dir", type=str, default="report", help="Report directory"
    )
    parser_run.set_defaults(func=run)


def _main():
    parser, subparsers = get_parser()
    wrap_parsers(subparsers)
    parameters = parser.parse_args()

    if parameters.S:
        logging.basicConfig(level=logging.ERROR)
    elif parameters.V:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    if not parameters.V:
        silence_third_party()

    if "func" in parameters:
        parameters.func(parameters)
    else:
        parser.print_usage()


if __name__ == "__main__":
    _main()
```
