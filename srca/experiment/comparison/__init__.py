"""
Compare algorithm combinations
"""
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
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
from ...utils import require_logging


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
            _create_graphs(
                graph_factories=graph_factories,
                cases=cases,
                delay=delay,
                output_dir=output_dir,
            )

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

    dump_csv(
        filename=report_filename,
        data=scores,
        headers=["method", "AC@1", "AC@3", "AC@5", "Avg@5"],
    )
