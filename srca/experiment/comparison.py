"""
Compare algorithm combinations
"""
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
import csv
from itertools import chain
from itertools import product
import logging
import os
from typing import Dict
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


_ALPHAS = (0.01, 0.05, 0.1, 0.5)
_MAX_CONDS_DIMS = (2, 3, 5, 10, None)
_TAU_MAXS = (0, 1, 2, 3)

_ZERO_TO_ONE = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
_TRUE_AND_FALSE = (True, False)


def _get_graph_factories(seed: int) -> Dict[str, GraphFactory]:
    graph_factories: Dict[str, GraphFactory] = {}
    for alpha, max_conds_dim in product(_ALPHAS, _MAX_CONDS_DIMS):
        params = dict(alpha=alpha, seed=seed)
        suffix = f"_a{alpha}"
        if max_conds_dim is not None:
            params["max_conds_dim"] = max_conds_dim
            suffix += f"_m{max_conds_dim}"
        graph_factories["PC_gauss" + suffix] = PCAlgFactory(method="PC-gauss", **params)
        graph_factories["PC_gsq" + suffix] = PCAlgFactory(method="PC-gsq", **params)
        for tau_max in _TAU_MAXS:
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
    graph_factories: Dict[str, GraphFactory], **scorer_params
) -> List[Model]:
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
            for stop_threshold in _ZERO_TO_ONE
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
    graph_factories: Dict[str, GraphFactory], **scorer_params
) -> List[Model]:
    models: List[Model] = []
    for rho, remove_sla in product(_ZERO_TO_ONE, _TRUE_AND_FALSE):
        params = dict(rho=rho, remove_sla=remove_sla, **scorer_params)
        suffix = f"_r{rho}"
        if remove_sla:
            suffix += "_nosla"
        for graph_name, graph_factory in graph_factories.items():
            models.append(
                Model(
                    graph_factory=graph_factory,
                    scorers=[
                        CorrelationScorer(**scorer_params),
                        RandomWalkScorer(**params),
                    ],
                    names=[graph_name, "Pearson", "RW" + suffix],
                ),
            )
            for beta in _ZERO_TO_ONE:
                models.append(
                    Model(
                        graph_factory=graph_factory,
                        scorers=[
                            PartialCorrelationScorer(**scorer_params),
                            SecondOrderRandomWalkScorer(beta=beta, **params),
                        ],
                        names=[
                            graph_name,
                            "PartialCorrelation",
                            "RW_2" + suffix + f"_b{beta}",
                        ],
                    ),
                )
    return models


def get_models(seed: int = 519) -> Tuple[List[Model], Dict[str, GraphFactory]]:
    """
    Prepare algorithm candidates
    """
    scorer_params = dict(seed=seed)
    graph_factories = _get_graph_factories(seed=seed)
    models = list(
        chain(
            _get_anomaly_detection_models(**scorer_params),
            _get_dfs_models(graph_factories=graph_factories, **scorer_params),
            _get_random_walk_models(graph_factories=graph_factories, **scorer_params),
        )
    )

    return models, graph_factories


def _create_graphs(
    graph_factories: Dict[str, GraphFactory],
    cases: List[Case],
    delay: int,
    output_dir: str,
):
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(run.__module__)
    for graph_name, graph_factory in graph_factories.items():
        for index, case in enumerate(cases):
            case_output_dir = os.path.join(output_dir, str(index))
            graph_filename = os.path.join(case_output_dir, f"{graph_name}.json")
            if os.path.isfile(graph_filename):
                continue
            logger.info("Create graph %s for case %d", graph_name, index)
            os.makedirs(case_output_dir, exist_ok=True)
            graph = graph_factory.create(
                data=case.data, current=case.data.detect_time + delay
            )
            graph.dump(graph_filename)


def _evaluate(models: List[Model], cases: List[Case], delay: int, output_dir: str):
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(run.__module__)
    scores: List[Tuple[str, float, float, float, float]] = []
    for model in models:
        name = model.name
        logger.info("Evaluate %s", name)
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
    graph_factories: Dict[str, GraphFactory],
    cases: List[Case],
    output_dir: str = "output",
    report_filename: str = "report.csv",
    delay: int = 300,
    max_workers: int = 1,
):
    # pylint: disable=too-many-arguments
    """
    Compare different models
    """
    graph_factory_items = list(graph_factories.items())
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        tasks = [
            executor.submit(
                _create_graphs,
                graph_factories=dict(graph_factory_items[i::max_workers]),
                cases=cases,
                delay=delay,
                output_dir=output_dir,
            )
            for i in range(max_workers)
        ]
        for task in as_completed(tasks):
            task.result()

    scores: List[Tuple[str, float, float, float, float]] = []
    num = max_workers * 4
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        tasks = [
            executor.submit(
                _evaluate,
                models=models[i::num],
                cases=cases,
                delay=delay,
                output_dir=output_dir,
            )
            for i in range(num)
        ]
        for task in as_completed(tasks):
            scores += task.result()

    with open(report_filename, "w", encoding="UTF-8") as obj:
        writer = csv.writer(obj)
        writer.writerow(["method", "AC@1", "AC@3", "AC@5", "Avg@5"])
        writer.writerows(scores)
