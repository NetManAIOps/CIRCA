"""
Test suites for common utilities
"""
import numpy as np
from scipy.stats import pearsonr

import pytest

from srca.alg.base import GraphFactory
from srca.alg.base import Score
from srca.alg.common import Evaluation
from srca.alg.common import Model
from srca.alg.common import NSigmaScorer
from srca.alg.common import ScoreRanker
from srca.alg.common import evaluate
from srca.alg.common import pearson
from srca.model.case import Case
from srca.model.case import CaseData
from srca.model.graph import Node


def test_pearson(size: int = 10):
    """
    pearson shall calculate the same Pearson coefficient as scipy
    """
    rng = np.random.default_rng()
    series_a = rng.standard_normal(size)
    series_b = rng.standard_normal(size)
    assert pearson(series_a, series_b) == pytest.approx(
        pearsonr(series_a, series_b)[0]
    )
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


def test_score_ranker():
    """
    ScoreRanker shall rank the node with the highest score as the first
    """
    latency = Node("DB", "Latency")
    traffic = Node("DB", "Traffic")
    saturation = Node("DB", "Saturation")
    scores = {
        latency: Score(0.8),
        traffic: Score(0.9),
        saturation: Score(0.5),
    }
    ranks = ScoreRanker().rank(None, None, scores, 0)
    assert [node for node, _ in ranks] == [traffic, latency, saturation]


def test_evaluate(graph_factory: GraphFactory, case_data: CaseData, tempdir: str):
    """
    evaluate shall reuse cached results
    """
    cases = [
        Case(data=case_data, answer={Node("DB", "Latency")}),
        Case(data=case_data, answer={Node("DB", "Saturation")}),
    ]
    names = ("graph", "scorer", "ranker")
    model = Model(
        graph_factory=graph_factory,
        scorer=NSigmaScorer(),
        ranker=ScoreRanker(),
        names=names,
    )
    report = evaluate(model, cases, delay=60, output_dir=tempdir)
    # Model.analyze shall not be called, using cache instead
    cached_report = evaluate(
        Model(None, None, None, names), cases, delay=60, output_dir=tempdir
    )
    for k in range(1, 4):
        assert report.accuracy(k) == cached_report.accuracy(k)
