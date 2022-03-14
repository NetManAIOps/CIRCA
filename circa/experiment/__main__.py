"""
Command line utilities
"""
import argparse
import dataclasses
from enum import Enum
import json
import logging
import os
from typing import Any
from typing import List
from typing import Tuple

import numpy as np

from . import comparison
from .comparison.utils import ModelParams
from .simulation import SimDataset
from .simulation import generate
from .simulation import robustness
from .simulation.scorer import SimRHTScorer
from ..alg.common import Model
from ..utils import silence_third_party


_GRAPH_SIZES: List[Tuple[int, int]] = [
    (50, 100),
    (100, 500),
    (500, 5000),
]
_NUM_NODES = [n for n, _ in _GRAPH_SIZES]
_NUM_GRAPHS = 10
_DEFAULT_SEED = 519


def _show_params(_: argparse.Namespace):
    def _convert_value(obj):
        if isinstance(obj, Enum):
            return obj.value
        if isinstance(obj, (list, tuple)):
            return list(map(_convert_value, obj))
        return obj

    def _enum2str(data: List[Tuple[str, Any]]) -> dict:
        return {key: _convert_value(value) for key, value in data}

    params = dataclasses.asdict(ModelParams(), dict_factory=_enum2str)
    print(json.dumps(params, indent=2, ensure_ascii=False))


def _generate(args: argparse.Namespace):
    num_nodes = set(args.num_nodes)
    rng = np.random.default_rng(args.seed)
    for num_node, num_edge in _GRAPH_SIZES:
        if num_node not in num_nodes:
            continue
        for i in range(_NUM_GRAPHS):
            dataset = generate(
                num_node=num_node, num_edge=num_edge, num_cases=args.num_cases, rng=rng
            )
            dataset.dump(os.path.join(args.output_dir, str(num_node), str(i)))


def _tune(args: argparse.Namespace):
    from .comparison.models import get_models  # pylint: disable=import-outside-toplevel

    data_dir: str = args.data_dir
    report_dir: str = args.report_dir
    os.makedirs(report_dir, exist_ok=True)

    logger = logging.getLogger(__package__)
    case_data_params = dict(lookup_window=120)  # 2 hours

    logger.info("Loading from %s", data_dir)
    dataset = SimDataset.load(
        folder=data_dir,
        case_data_params=case_data_params,
    )
    models, _ = get_models(
        graph_factories={"GT": dataset.graph_factory},  # GT: Ground Truth
        params=args.model_params,
        seed=args.seed,
        cuda=args.cuda,
    )
    logger.info("Start tuning on %s with #models=%d", data_dir, len(models))
    comparison.run(
        models=models,
        cases=dataset.cases,
        graph_factories=None,
        output_dir=args.output_dir,
        report_filename=os.path.join(report_dir, "report.csv"),
        max_workers=1 if args.cuda else args.max_workers,
    )


def _run(args: argparse.Namespace):
    from .comparison.models import get_models  # pylint: disable=import-outside-toplevel

    model_params: ModelParams = args.model_params
    report_dir: str = args.report_dir
    os.makedirs(report_dir, exist_ok=True)

    logger = logging.getLogger(__package__)
    case_data_params = dict(lookup_window=120)

    for num_node, _ in _GRAPH_SIZES:
        for i in range(_NUM_GRAPHS):
            dataset_dir = os.path.join(args.data_dir, str(num_node), str(i))
            logger.info("Loading from %s", dataset_dir)
            dataset = SimDataset.load(
                folder=dataset_dir,
                case_data_params=case_data_params,
            )
            models, _ = get_models(
                graph_factories={"GT": dataset.graph_factory},  # GT: Ground Truth
                params=model_params,
                seed=args.seed,
                cuda=args.cuda,
                max_workers=args.max_workers,
            )
            if model_params.rht:
                models.append(
                    Model(
                        graph_factory=dataset.graph_factory,
                        scorers=[
                            SimRHTScorer(seed=args.seed, max_workers=args.max_workers)
                        ],
                        names=["GT", "RHT-PG"],
                    ),
                )
            logger.info("Start running on %s", dataset_dir)
            comparison.run(
                models=models,
                cases=dataset.cases,
                graph_factories=None,
                output_dir=os.path.join(args.output_dir, str(num_node), str(i)),
                report_filename=os.path.join(report_dir, f"report-{num_node}-{i}.csv"),
                max_workers=1,
            )


def _run_robustness(args: argparse.Namespace):
    cache_dir: str = args.cache_dir
    data_dir: str = args.data_dir
    report_dir: str = args.report_dir
    os.makedirs(report_dir, exist_ok=True)

    for num_node, _ in _GRAPH_SIZES:
        robustness.evaluate(
            num_graph=_NUM_GRAPHS,
            cache_dir=os.path.join(cache_dir, str(num_node)),
            data_dir=os.path.join(data_dir, str(num_node)),
            report_dir=os.path.join(report_dir, str(num_node)),
        )


def _add_output_argument(parser: argparse.ArgumentParser, default: str):
    parser.add_argument(
        "--output-dir", type=str, default=default, help="Output directory"
    )


def _add_report_argument(parser: argparse.ArgumentParser, default: str):
    parser.add_argument(
        "--report-dir", type=str, default=default, help="Report directory"
    )


def get_parser() -> Tuple[argparse.ArgumentParser, argparse._SubParsersAction]:
    """
    Prepare the command line parser
    """
    parser_params = dict(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # The entrance
    parser = argparse.ArgumentParser(prog=__package__, **parser_params)
    parser.add_argument("-v", dest="V", action="store_true", help="Verbose")
    parser.add_argument("-s", dest="S", action="store_true", help="Silence")
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED, help="Random seed")
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="The number of workers for parallel calculation",
    )
    parser.add_argument("--cuda", action="store_true", help="Use CUDA (GPU)")
    parser.add_argument(
        "--model-params",
        type=ModelParams,
        required=False,
        help="Provide a json file to specify options for model parameters.",
    )
    subparsers = parser.add_subparsers(title="subcommands")

    parser_show_params = subparsers.add_parser(
        "params", help="Show default model parameters", **parser_params
    )
    parser_show_params.set_defaults(func=_show_params)

    # For data generation
    parser_gen = subparsers.add_parser(
        "generate", help="Generate dataset by simulation", **parser_params
    )
    parser_gen.add_argument(
        "--num-cases",
        type=int,
        default=100,
        help="The number of cases for each graph",
    )
    parser_gen.add_argument(
        "-n",
        "--num-node",
        dest="num_nodes",
        type=int,
        choices=_NUM_NODES,
        default=_NUM_NODES,
        nargs="*",
        help="Choose the number of nodes in the generated graph",
    )
    _add_output_argument(parser_gen, default="dataset")
    parser_gen.set_defaults(func=_generate)

    # Experiments over parameter combinations
    parser_tune = subparsers.add_parser(
        "tune",
        help="Explore all combinations of model parameters in the given dataset",
        **parser_params,
    )
    parser_tune.add_argument(
        "--data-dir",
        type=str,
        default=os.path.join("dataset", str(_GRAPH_SIZES[0][0]), "0"),
        help="Data directory",
    )
    _add_output_argument(parser_tune, default=os.path.join("output", "sim-tuning"))
    _add_report_argument(parser_tune, default=os.path.join("report", "sim-tuning"))
    parser_tune.set_defaults(func=_tune)

    # Experiments on a set of parameters
    parser_run = subparsers.add_parser(
        "run",
        help="Explore all the dataset with pre-defined model parameters",
        **parser_params,
    )
    parser_run.add_argument(
        "--data-dir",
        type=str,
        default="dataset",
        help="Data directory",
    )
    _add_output_argument(parser_run, default=os.path.join("output", "sim"))
    _add_report_argument(parser_run, default=os.path.join("report", "sim"))
    parser_run.set_defaults(func=_run)

    parser_robustness = subparsers.add_parser(
        "robustness",
        help="Compare models with faults of different dependency intensities",
        **parser_params,
    )
    parser_robustness.add_argument(
        "--data-dir",
        type=str,
        default=os.path.join("dataset"),
        help="Data directory",
    )
    parser_robustness.add_argument(
        "--cache-dir",
        type=str,
        default=os.path.join("output", "sim"),
        help="Data directory",
    )
    _add_report_argument(
        parser_robustness, default=os.path.join("report", "sim-robustness")
    )
    parser_robustness.set_defaults(func=_run_robustness)

    return parser, subparsers


def _main():
    parser, _ = get_parser()
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
