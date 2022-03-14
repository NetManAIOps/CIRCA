"""
Anomaly Detection in Streams with Extreme Value Theory

See SPOT in KDD'17
https://doi.org/10.1145/3097983.3098144
"""
from typing import Dict
from typing import List
from typing import Sequence

from ads_evt import biSPOT
import numpy as np

from .base import Score
from .common import DecomposableScorer
from ..model.case import CaseData
from ..model.graph import Graph
from ..model.graph import Node


def spot(train_y: np.ndarray, test_y: np.ndarray, proba: float = 1e-4) -> np.ndarray:
    """
    Estimate to what extend each value in test_y violates
    the normal distribution defined by train_y
    """
    model = biSPOT(q=proba)
    model.fit(init_data=train_y, data=test_y)
    model.initialize()
    results = model.run(with_alarm=False)
    scores: List[float] = []
    for index, (upper, lower) in enumerate(
        zip(results["upper_thresholds"], results["lower_thresholds"])
    ):
        width: float = upper - lower
        if width <= 0:
            width = 1
        if test_y[index] > upper:
            scores.append((test_y[index] - upper) / width)
        elif test_y[index] < lower:
            scores.append((lower - test_y[index]) / width)
        else:
            scores.append(0)

    return np.array(scores)


class SPOTScorer(DecomposableScorer):
    """
    Score nodes by SPOT
    """

    def __init__(self, proba: float = 1e-4, **kwargs):
        super().__init__(**kwargs)
        self._porba = proba

    def score_node(
        self,
        graph: Graph,
        series: Dict[Node, Sequence[float]],
        node: Node,
        data: CaseData,
    ) -> Score:
        series_y = np.array(series[node])
        train_y: np.ndarray = series_y[: data.train_window]
        test_y: np.ndarray = series_y[-data.test_window :]
        scores = spot(train_y, test_y, proba=self._porba)
        spot_score = self._aggregator(abs(scores))
        if spot_score == 0:
            return None
        score = Score(spot_score)
        score["spot-score"] = spot_score
        return score
