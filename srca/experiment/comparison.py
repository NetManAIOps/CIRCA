"""
Compare algorithm combinations
"""
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
import csv
from itertools import chain
from itertools import product
import os
from typing import Dict
from typing import Iterable
from typing import List
from typing import Tuple

from ..alg.base import GraphFactory
from ..alg.base import Scorer
from ..alg.common import EmptyGraphFactory
from ..alg.common import Model
from ..alg.common import NSigmaScorer
from ..alg.common import evaluate
from ..alg.correlation import CorrelationScorer
from ..alg.correlation import PartialCorrelationScorer
from ..alg.dfs import DFSScorer
from ..alg.dfs import MicroHECLScorer
from ..alg.graph.pcts import PCTSFactory
from ..alg.graph.r import PCAlgFactory
from ..alg.random_walk import RandomWalkScorer
from ..alg.random_walk import SecondOrderRandomWalkScorer
from ..model.case import Case
from ..utils import Timer
from ..utils import require_logging


_ALPHAS = (0.01, 0.05, 0.1, 0.5)
_MAX_CONDS_DIMS = (2, 3, 5, 10, None)
_TAU_MAXS = (0, 1, 2, 3)

_ZERO_TO_ONE = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
_TRUE_AND_FALSE = (True, False)


def _require_iterable(item) -> Iterable:
    if not isinstance(item, Iterable):
        return [item]
    return item


def _get_graph_factories(seed: int, params: dict = None) -> Dict[str, GraphFactory]:
    if params is None:
        params = {}
    alphas = _require_iterable(params.get("alpha", _ALPHAS))
    max_conds_dims = _require_iterable(params.get("max_conds_dim", _MAX_CONDS_DIMS))
    tau_maxs = _require_iterable(params.get("tau_maxs", _TAU_MAXS))

    graph_factories: Dict[str, GraphFactory] = {}
    for alpha, max_conds_dim in product(alphas, max_conds_dims):
        params = dict(alpha=alpha, seed=seed)
        suffix = f"_a{alpha}"
        if max_conds_dim is not None:
            params["max_conds_dim"] = max_conds_dim
            suffix += f"_m{max_conds_dim}"
        graph_factories["PC_gauss" + suffix] = PCAlgFactory(method="PC-gauss", **params)
        graph_factories["PC_gsq" + suffix] = PCAlgFactory(method="PC-gsq", **params)
        for tau_max in tau_maxs:
            graph_name = "PCTS" + suffix + f"_t{tau_max}"
            graph_factories[graph_name] = PCTSFactory(tau_max=tau_max, **params)
    return graph_factories


def _get_detectors(**scorer_params) -> Dict[str, Tuple[Scorer, float]]:
    """
    Map a detector name to a pair of Scorer and threshold
    """
    return {"NSigma": (NSigmaScorer(**scorer_params), 3)}


def _get_anomaly_detection_models(**scorer_params) -> List[Model]:
    detectors = _get_detectors(**scorer_params)
    graph_factory = EmptyGraphFactory()
    return [
        Model(
            graph_factory=graph_factory,
            scorers=[
                detector,
            ],
            names=("Empty", name),
        )
        for name, (detector, _) in detectors.items()
    ]


def _get_dfs_models(
    graph_factories: Dict[str, GraphFactory], params: dict = None, **scorer_params
) -> List[Model]:
    if params is None:
        params = {}
    stop_thresholds = _require_iterable(params.get("stop_threshold", _ZERO_TO_ONE))

    models: List[Model] = []
    detectors = _get_detectors(**scorer_params)
    # DFS
    for detector_name, (detector, anomaly_threshold) in detectors.items():
        scorer = DFSScorer(anomaly_threshold=anomaly_threshold, **scorer_params)
        scorer_name = f"DFS_a{anomaly_threshold}"
        for graph_name, graph_factory in graph_factories.items():
            models.append(
                Model(
                    graph_factory=graph_factory,
                    scorers=[detector, scorer],
                    names=[graph_name, detector_name, scorer_name],
                ),
            )
            # Microscope in ICSOC'18
            models.append(
                Model(
                    graph_factory=graph_factory,
                    scorers=[detector, scorer, CorrelationScorer(**scorer_params)],
                    names=[graph_name, detector_name, scorer_name, "Pearson"],
                ),
            )
    # MicroHECL
    for detector_name, (detector, anomaly_threshold) in detectors.items():
        scorers = {
            f"MicroHECL_a{anomaly_threshold}_s{stop_threshold}": MicroHECLScorer(
                anomaly_threshold=anomaly_threshold,
                stop_threshold=stop_threshold,
                **scorer_params,
            )
            for stop_threshold in stop_thresholds
        }
        for graph_name, graph_factory in graph_factories.items():
            for scorer_name, scorer in scorers.items():
                models.append(
                    Model(
                        graph_factory=graph_factory,
                        scorers=[detector, scorer],
                        names=[graph_name, detector_name, scorer_name],
                    ),
                )
    return models


def _get_random_walk_models(
    graph_factories: Dict[str, GraphFactory], params: dict = None, **scorer_params
) -> List[Model]:
    if params is None:
        params = {}
    rhos = _require_iterable(params.get("rho", _ZERO_TO_ONE))
    remove_slas = _require_iterable(params.get("remove_sla", _TRUE_AND_FALSE))
    betas = _require_iterable(params.get("beta", _ZERO_TO_ONE))

    models: List[Model] = []
    for rho, remove_sla in product(rhos, remove_slas):
        params = dict(rho=rho, remove_sla=remove_sla, **scorer_params)
        suffix = f"_r{rho}"
        if remove_sla:
            suffix += "_nosla"
        for graph_name, graph_factory in graph_factories.items():
            # MicroCause
            models.append(
                Model(
                    graph_factory=graph_factory,
                    scorers=[
                        PartialCorrelationScorer(**scorer_params),
                        RandomWalkScorer(**params),
                    ],
                    names=[graph_name, "PartialCorrelation", "RW" + suffix],
                ),
            )
            # CloudRanger
            for beta in betas:
                models.append(
                    Model(
                        graph_factory=graph_factory,
                        scorers=[
                            CorrelationScorer(**scorer_params),
                            SecondOrderRandomWalkScorer(beta=beta, **params),
                        ],
                        names=[
                            graph_name,
                            "Pearson",
                            "RW_2" + suffix + f"_b{beta}",
                        ],
                    ),
                )
    return models


def get_models(
    graph_factories: Dict[str, GraphFactory] = None,
    params: Dict[str, dict] = None,
    seed: int = 0,
) -> Tuple[List[Model], Dict[str, GraphFactory]]:
    """
    Prepare algorithm candidates

    Parameters:
        graph_factories: Specify GraphFactory for models
        params: Specify options for model parameters.
                Values that are not interable will be converted into a list.
            graph: Graph parameters
                alpha: Thresholds for p-value. Default: (0.01, 0.05, 0.1, 0.5)
                max_conds_dim: The maximum size of condition set for PC and PCTS.
                    Default: (2, 3, 5, 10, None)
                tau_max: The maximum lag considered by PCTS. Default: (0, 1, 2, 3)
            dfs: DFS parameters
                stop_threshold: Threshold for MicroHECL. Default: (0.0, 0.1, ..., 1.0)
            rw: Random walk parameters
                rho: Back-ward probability. Default: (0.0, 0.1, ..., 1.0)
                remove_sla: Whether to disable forwarding to the SLA.
                    Default: (True, False)
                beta: For second order random walk. Default: (0.0, 0.1, ..., 1.0)
    """
    if params is None:
        params = {}

    scorer_params = dict(seed=seed)
    if graph_factories is None:
        graph_factories = _get_graph_factories(seed=seed, params=params.get("graph"))
    models = list(
        chain(
            _get_anomaly_detection_models(**scorer_params),
            _get_dfs_models(
                graph_factories=graph_factories,
                params=params.get("dfs"),
                **scorer_params,
            ),
            _get_random_walk_models(
                graph_factories=graph_factories,
                params=params.get("rw"),
                **scorer_params,
            ),
        )
    )

    return models, graph_factories


def _create_graphs(
    graph_factories: Dict[str, GraphFactory],
    cases: List[Case],
    delay: int,
    output_dir: str,
):
    for graph_name, graph_factory in graph_factories.items():
        for index, case in enumerate(cases):
            case_output_dir = os.path.join(output_dir, str(index))
            graph_filename = os.path.join(case_output_dir, f"{graph_name}.json")
            if os.path.isfile(graph_filename):
                continue
            os.makedirs(case_output_dir, exist_ok=True)
            with Timer(name=f"{graph_name} for case {index}"):
                graph = graph_factory.create(
                    data=case.data, current=case.data.detect_time + delay
                )
            graph.dump(graph_filename)


def _evaluate(models: List[Model], cases: List[Case], delay: int, output_dir: str):
    scores: List[Tuple[str, float, float, float, float]] = []
    for model in models:
        name = model.name
        with Timer(name=name):
            report = evaluate(model, cases, delay=delay, output_dir=output_dir)
        scores.append(
            (
                name,
                report.accuracy(1),
                report.accuracy(3),
                report.accuracy(5),
                report.average(5),
            )
        )
    return scores


def run(
    models: List[Model],
    cases: List[Case],
    graph_factories: Dict[str, GraphFactory] = None,
    output_dir: str = None,
    report_filename: str = "report.csv",
    delay: int = 300,
    max_workers: int = 1,
):
    # pylint: disable=too-many-arguments
    """
    Compare different models

    Parameters:
        graph_factories: If given, graphs will be created for each case
            before any model starts.
        output_dir: Where the intermediate results will be cached.
        report_filename: The name of a csv file that will store the experiment results.
        delay: A model is assumed to start in this amount of seconds
            after a fault is detected.
        max_workers: The given number of processes will be created
            to accelerate the calculation.
    """
    if output_dir and graph_factories is not None:
        if max_workers >= 2:
            graph_factory_items = list(graph_factories.items())
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                tasks = [
                    executor.submit(
                        require_logging(_create_graphs),
                        graph_factories=dict(graph_factory_items[i::max_workers]),
                        cases=cases,
                        delay=delay,
                        output_dir=output_dir,
                    )
                    for i in range(max_workers)
                ]
                for task in as_completed(tasks):
                    task.result()
        else:
            _create_graphs(graph_factories=graph_factories, cases=cases, delay=delay,
                        output_dir=output_dir,)

    if max_workers >= 2:
        scores: List[Tuple[str, float, float, float, float]] = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            tasks = [
                executor.submit(
                    require_logging(_evaluate),
                    models=models[i::max_workers],
                    cases=cases,
                    delay=delay,
                    output_dir=output_dir,
                )
                for i in range(max_workers)
            ]
            for task in as_completed(tasks):
                scores += task.result()
    else:
        scores = _evaluate(
            models=models, cases=cases, delay=delay, output_dir=output_dir
        )

    with open(report_filename, "w", encoding="UTF-8") as obj:
        writer = csv.writer(obj)
        writer.writerow(["method", "AC@1", "AC@3", "AC@5", "Avg@5"])
        writer.writerows(scores)
