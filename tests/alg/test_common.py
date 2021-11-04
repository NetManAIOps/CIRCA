"""
Test suites for common utilities
"""
from srca.alg.base import Score
from srca.alg.common import ScoreRanker
from srca.model.graph import Node


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
