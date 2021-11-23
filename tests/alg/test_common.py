"""
Test suites for common utilities
"""
import os

from srca.alg.base import Score
from srca.alg.common import Evaluation
from srca.alg.common import NSigmaScorer
from srca.alg.common import ScoreRanker
from srca.alg.common import evaluate
from srca.model.case import Case
from srca.model.case import CaseData
from srca.model.graph import Node


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
    ranks = ScoreRanker().rank(None, scores, 0)
    assert [node for node, _ in ranks] == [traffic, latency, saturation]


def test_evaluate(case_data: CaseData, tempdir: str):
    """
    evaluate shall reuse cached results
    """
    output_filename = os.path.join(tempdir, "report.json")
    cases = [
        Case(data=case_data, answer={Node("DB", "Latency")}),
        Case(data=case_data, answer={Node("DB", "Saturation")}),
    ]
    scorer = NSigmaScorer()
    ranker = ScoreRanker()
    report = evaluate(scorer, ranker, cases, delay=60, output_filename=output_filename)
    # scorer and ranker shall not be called, using cache instead
    cached_report = evaluate(
        None, None, cases, delay=60, output_filename=output_filename
    )
    for k in range(1, 4):
        assert report.accuracy(k) == cached_report.accuracy(k)
