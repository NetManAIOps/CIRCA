"""
Test suites for common utilities
"""
import numpy as np
from scipy.stats import pearsonr

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from circa.alg.common import DecomposableScorer
from circa.alg.common import Evaluation
from circa.alg.common import Model
from circa.alg.common import NSigmaScorer
from circa.alg.common import evaluate
from circa.alg.common import pearson
from circa.experiment.simulation import generate
from circa.graph import GraphFactory
from circa.model.case import Case
from circa.model.case import CaseData
from circa.model.graph import Graph
from circa.model.graph import Node


def test_parallel_decomposable_scorer(graph: Graph, case_data: CaseData):
    """
    DecomposableScorer with multiple workers shall return the same as with a single one
    """
    assert issubclass(NSigmaScorer, DecomposableScorer)
    params = dict(graph=graph, data=case_data, current=case_data.detect_time + 60)
    result_single = NSigmaScorer(max_workers=1).score(**params)
    result_multiple = NSigmaScorer(max_workers=2).score(**params)
    assert result_single == result_multiple


def _decomposable_score(
    data: CaseData, graph_factory: GraphFactory, max_workers: int, delay: int = 300
):
    current = data.detect_time + delay
    graph = graph_factory.create(data=data, current=current)
    scorer = NSigmaScorer(max_workers=max_workers)
    _ = scorer.score(graph=graph, data=data, current=current)


@pytest.mark.parametrize(
    ("max_workers", "num_node", "num_edge"),
    [
        (1, 100, 500),
        (2, 100, 500),
        (1, 500, 5000),
        (2, 500, 5000),
    ],
)
def test_multiple_processing(
    benchmark: BenchmarkFixture, max_workers: int, num_node: int, num_edge: int
):
    """
    Compare DecomposableScorer with different max_workers
    """
    dataset = generate(
        num_node=num_node, num_edge=num_edge, num_cases=1, rng=np.random.default_rng(0)
    )
    benchmark(
        _decomposable_score,
        data=dataset.cases[0].data,
        graph_factory=dataset.graph_factory,
        max_workers=max_workers,
    )


def test_pearson(size: int = 10):
    """
    pearson shall calculate the same Pearson coefficient as scipy
    """
    rng = np.random.default_rng()
    series_a = rng.standard_normal(size)
    series_b = rng.standard_normal(size)
    assert pearson(series_a, series_b) == pytest.approx(pearsonr(series_a, series_b)[0])
    assert pearson(series_a, np.zeros(size)) == 0.0


def test_evaluation():
    """
    Evaluation shall calculate AC@k and Avg@k for ranking results
    """
    ranks = [Node("DB", "Latency"), Node("DB", "Traffic"), Node("DB", "Saturation")]

    k = 3
    evaluation = Evaluation(recommendation_num=k)
    assert evaluation.accuracy(k) is None

    evaluation(ranks, {ranks[2]})
    assert evaluation.accuracy(2) == 0 and evaluation.accuracy(3) == 1
    evaluation(ranks, {ranks[1]})
    assert evaluation.accuracy(2) == 0.5 and evaluation.accuracy(3) == 1
    assert evaluation.average(k) == 0.5


def test_evaluate(graph_factory: GraphFactory, case_data: CaseData, tempdir: str):
    """
    evaluate shall reuse cached results
    """
    cases = [
        Case(data=case_data, answer={Node("DB", "Latency")}),
        Case(data=case_data, answer={Node("DB", "Saturation")}),
    ]
    names = ("graph", "scorer")
    model = Model(
        graph_factory=graph_factory,
        scorers=[
            NSigmaScorer(),
        ],
        names=names,
    )
    report = evaluate(model, cases, delay=60, output_dir=tempdir)
    # Model.analyze shall not be called, using cache instead
    cached_report = evaluate(
        Model(None, [None], names), cases, delay=60, output_dir=tempdir
    )
    for k in range(1, 4):
        assert report.accuracy(k) == cached_report.accuracy(k)
