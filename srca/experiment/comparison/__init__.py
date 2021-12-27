"""
Compare algorithm combinations
"""
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
import logging
from multiprocessing import Process
import os
from typing import Dict
from typing import List
from typing import Tuple

from ...alg.base import GraphFactory
from ...alg.common import Model
from ...alg.common import evaluate
from ...model.case import Case
from ...utils import Timer
from ...utils import dump_csv
from ...utils import dump_json
from ...utils import require_logging


def _create_graph(
    graph_factory: GraphFactory,
    case: Case,
    timer_name: str,
    graph_filename: str,
    current: float,
):
    with Timer(name=timer_name):
        graph = graph_factory.create(data=case.data, current=current)
    graph.dump(graph_filename)


def _create_graphs(
    graph_factories: Dict[str, GraphFactory],
    cases: List[Case],
    delay: int,
    output_dir: str,
    timeout: int = 3600,
):
    logger = logging.getLogger(f"{run.__module__}.create_graphs")
    for graph_name, graph_factory in graph_factories.items():
        for index, case in enumerate(cases):
            case_output_dir = os.path.join(output_dir, str(index))
            graph_filename = os.path.join(case_output_dir, f"{graph_name}.json")
            if os.path.isfile(graph_filename):
                continue
            os.makedirs(case_output_dir, exist_ok=True)

            task = Process(
                target=require_logging(_create_graph),
                kwargs=dict(
                    graph_factory=graph_factory,
                    case=case,
                    timer_name=f"{graph_name} for case {index}",
                    graph_filename=graph_filename,
                    current=case.data.detect_time + delay,
                ),
            )
            task.start()
            task.join(timeout=timeout)
            if task.is_alive():
                task.terminate()
                logger.warning("%s: Timeout for case %d", graph_name, index)
                dump_json(graph_filename, {"status": "Timeout"})


def _evaluate(models: List[Model], cases: List[Case], **kwargs):
    scores: List[Tuple[str, float, float, float, float, float]] = []
    num_cases = max(len(cases), 1)
    for model in models:
        name = model.name
        with Timer(name=name) as timer:
            report = evaluate(model, cases, **kwargs)
            duration = timer.duration
        scores.append(
            (
                name,
                report.accuracy(1),
                report.accuracy(3),
                report.accuracy(5),
                report.average(5),
                duration.total_seconds() / num_cases,
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
    timeout: int = 3600,
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
    params = dict(delay=delay, output_dir=output_dir, timeout=timeout)
    if output_dir and graph_factories is not None:
        if max_workers >= 2:
            graph_factory_items = list(graph_factories.items())
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                tasks = [
                    executor.submit(
                        require_logging(_create_graphs),
                        graph_factories=dict(graph_factory_items[i::max_workers]),
                        cases=cases,
                        **params,
                    )
                    for i in range(max_workers)
                ]
                for task in as_completed(tasks):
                    task.result()
        else:
            _create_graphs(
                graph_factories=graph_factories,
                cases=cases,
                **params,
            )

    if max_workers >= 2:
        scores: List[Tuple[str, float, float, float, float, float]] = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            tasks = [
                executor.submit(
                    require_logging(_evaluate),
                    models=models[i::max_workers],
                    cases=cases,
                    **params,
                )
                for i in range(max_workers)
            ]
            for task in as_completed(tasks):
                scores += task.result()
    else:
        scores = _evaluate(models=models, cases=cases, **params)

    dump_csv(
        filename=report_filename,
        data=scores,
        headers=["method", "AC@1", "AC@3", "AC@5", "Avg@5", "avg duration"],
    )
