"""
Common utilities
"""
import datetime
from typing import Callable
from typing import Dict
from typing import List
from typing import Sequence

import numpy as np
from sklearn.preprocessing import StandardScaler

from .base import Ranker
from .base import Scorer
from ..model.case import CaseData
from ..model.graph import Node


class NSigmaScorer(Scorer):
    """
    Score nodes by n-sigma
    """

    def __init__(
        self,
        interval: datetime.timedelta = datetime.timedelta(minutes=1),
        lookup_window: int = 120,
        detect_window: int = 10,
        aggregator: Callable[[Sequence[float]], float] = max,
    ):
        self._interval = interval

        self._train_window = lookup_window - detect_window + 1
        self._test_window = detect_window
        self._lookup_window = lookup_window * interval.total_seconds()
        self._aggregator = aggregator

    def score_node(
        self, series: Dict[Node, Sequence[float]], node: Node, data: CaseData
    ) -> float:
        # pylint: disable=unused-argument
        """
        Estimate how suspicious a node is
        """
        series_y = np.array(series[node])
        train_y: np.ndarray = series_y[: self._train_window]
        test_y: np.ndarray = series_y[-self._test_window :]
        scaler = StandardScaler().fit(train_y.reshape(-1, 1))
        z_score = scaler.transform(test_y.reshape(-1, 1))[:, 0]
        return self._aggregator(abs(z_score))

    def score(self, data: CaseData, current: float) -> Dict[Node, float]:
        current = max(current, data.detect_time)

        start = data.detect_time - self._lookup_window
        series: Dict[Node, Sequence[float]] = {}
        for node in data.graph.nodes:
            node_data = data.data_loader.load(
                entity=node.entity,
                metric=node.metric,
                start=start,
                end=current,
                interval=self._interval,
            )
            if node_data:
                series[node] = node_data

        return {node: self.score_node(series, node, data) for node in series}


class ScoreRanker(Ranker):
    """
    Rank nodes by scores directly
    """

    def rank(
        self, data: CaseData, scores: Dict[Node, float], current: float
    ) -> List[Node]:
        return sorted(scores.keys(), key=lambda node: -scores[node])


def analyze(
    scorer: Scorer, ranker: Ranker, data: CaseData, current: float
) -> List[Node]:
    """
    Conduct root cause analysis
    """
    scores = scorer.score(data=data, current=current)
    return ranker.rank(data=data, scores=scores, current=current)
