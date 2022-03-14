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
from typing import Set
from typing import Tuple

from ...alg.common import Evaluation
from ...alg.common import Model
from ...alg.common import evaluate
from ...graph import GraphFactory
from ...model.case import Case
from ...utils import Timer
from ...utils import dump_csv
from ...utils import dump_json
from ...utils import load_csv
from ...utils import require_logging


IDEAL_METHOD = "GT-Ideal"


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


def report_evaluation(name: str, report: Evaluation, duration: float = None):
    """
    Return a tuple of (name, AC@1, AC@3, AC@5, Avg@5).
    Duration will show at the end of the tuple if provided.
    """
    score = (
        name,
        report.accuracy(1),
        report.accuracy(3),
        report.accuracy(5),
        report.average(5),
    )
    if duration is not None:
        score += (duration,)
    return score


def report_ideal(cases: List[Case], name: str = IDEAL_METHOD):
    """
    Return the score tuple for the ideal algorithm
    """
    report = Evaluation()
    for case in cases:
        report(ranks=list(case.answer), answers=case.answer)
    return report_evaluation(name, report)


def _wrap_ideal_report(scores: List[tuple], cases: List[Case]):
    names = {score[0] for score in scores}
    if IDEAL_METHOD not in names:
        scores = scores + [report_ideal(cases=cases, name=IDEAL_METHOD)]
    return scores


def _evaluate(models: List[Model], cases: List[Case], **kwargs):
    logger = logging.getLogger(f"{run.__module__}.evaluate")
    scores: List[Tuple[str, float, float, float, float, float]] = []
    num_cases = max(len(cases), 1)
    for model in models:
        name = model.name
        with Timer(name=name) as timer:
            report = evaluate(model, cases, **kwargs)
            duration = timer.duration
        score = report_evaluation(name, report, duration.total_seconds() / num_cases)
        logger.info("Finish: %s,%f,%f,%f,%f,%f", *score)
        scores.append(score)
    return scores


def _load_cached_report(report_filename: str, models: List[Model]):
    scores: List[Tuple[str, float, float, float, float, float]] = []
    cached_models: Set[str] = set()
    if os.path.exists(report_filename):
        reader = load_csv(report_filename)
        _ = next(reader)
        for model_name, *values in reader:
            scores.append((model_name, *map(float, values)))
            cached_models.add(model_name)
    models = [model for model in models if model.name not in cached_models]
    return scores, models


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

    scores, models = _load_cached_report(report_filename=report_filename, models=models)
    if max_workers >= 2:
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
        scores += _evaluate(models=models, cases=cases, **params)
    scores = _wrap_ideal_report(scores=scores, cases=cases)

    dump_csv(
        filename=report_filename,
        data=scores,
        headers=["method", "AC@1", "AC@3", "AC@5", "Avg@5", "avg duration"],
    )
