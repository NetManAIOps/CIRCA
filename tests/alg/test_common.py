"""
Test suites for common utilities
"""
from srca.alg.base import Score
from srca.alg.common import Evaluation
from srca.alg.common import ScoreRanker
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
