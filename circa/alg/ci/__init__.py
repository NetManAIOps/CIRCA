"""
Causal Inference-based Root Cause Analysis (CIRCA)
"""
from typing import Callable
from typing import Dict
from typing import Sequence
from typing import Tuple

import numpy as np

from .anm import ANMRegressor
from .base import Regressor
from ..base import Score
from ..base import Scorer
from ..common import DecomposableScorer
from ..common import zscore_conf
from ...model.case import CaseData
from ...model.graph import Graph
from ...model.graph import Node


class RHTScorer(DecomposableScorer):
    """
    Scorer with regression-based hypothesis testing
    """

    def __init__(
        self,
        tau_max: int = 0,
        regressor: Regressor = None,
        use_confidence: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._tau_max = max(tau_max, 0)
        self._regressor = regressor if regressor else ANMRegressor()
        self._use_confidence = use_confidence

    @staticmethod
    def _split_train_test(
        series_x: np.ndarray,
        series_y: np.ndarray,
        train_window: int,
        test_window: int,
    ):
        train_x: np.ndarray = series_x[:train_window, :]
        train_y: np.ndarray = series_y[:train_window]
        test_x: np.ndarray = series_x[-test_window:, :]
        test_y: np.ndarray = series_y[-test_window:]
        return train_x, test_x, train_y, test_y

    def split_data(
        self,
        data: Dict[Node, Sequence[float]],
        node: Node,
        parents: Sequence[Node],
        case_data: CaseData,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data for training and testing

        Return (train_x, test_x, train_y, test_y)
        """
        series = np.array([data[parent] for parent in parents if parent in data]).T
        if len(series) == 0:
            series = np.zeros((0, 0))
        series_x = np.hstack([np.roll(series, i, 0) for i in range(self._tau_max + 1)])
        series_x: np.ndarray = series_x[self._tau_max :, :]
        series_y = np.array(data[node][self._tau_max :])

        return self._split_train_test(
            series_x=series_x,
            series_y=series_y,
            train_window=case_data.train_window - self._tau_max,
            test_window=case_data.test_window,
        )

    def score_node(
        self,
        graph: Graph,
        series: Dict[Node, Sequence[float]],
        node: Node,
        data: CaseData,
    ) -> Score:
        parents = list(graph.parents(node))

        train_x, test_x, train_y, test_y = self.split_data(series, node, parents, data)
        z_scores = self._regressor.score(
            train_x=train_x, test_x=test_x, train_y=train_y, test_y=test_y
        )
        z_score = self._aggregator(abs(z_scores))
        confidence = zscore_conf(z_score)
        if self._use_confidence:
            score = Score(confidence)
            score.key = (score.score, z_score)
        else:
            score = Score(z_score)
        score["z-score"] = z_score
        score["Confidence"] = confidence

        return score


class DAScorer(Scorer):
    """
    Scorer with descendant adjustment
    """

    def __init__(
        self,
        threshold: float = 0,
        aggregator: Callable[[Sequence[float]], float] = max,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._threshold = max(threshold, 0.0)
        self._aggregator = aggregator

    def score(
        self,
        graph: Graph,
        data: CaseData,
        current: float,
        scores: Dict[Node, Score] = None,
    ) -> Dict[Node, Score]:
        sorted_nodes = [
            {node for node in nodes if node in scores}
            for nodes in graph.topological_sort
        ]
        # 0. Set topological rank
        for index, nodes in enumerate(sorted_nodes):
            for node in nodes:
                score = scores[node]
                score["index"] = index

        # 1. Gather child scores
        child_scores: Dict[Node, Dict[Node, float]] = {}
        for nodes in reversed(sorted_nodes):
            for node in nodes:
                child_score: Dict[Node, float] = {}
                for child in graph.children(node):
                    if child in scores:
                        child_score[child] = scores[child].score
                        if scores[child].score < self._threshold:
                            child_score.update(child_scores.get(child, {}))
                child_scores[node] = child_score

        # 2. Set child_score
        for node, score in scores.items():
            if score.score >= self._threshold:
                child_score = child_scores[node]
                if child_score:
                    child_score = self._aggregator(child_score.values())
                    score.score += child_score
                    score["child_score"] = child_score

        # 3. Set key
        for score in scores.values():
            score.key = (score.score, -score["index"], score.get("z-score", 0))
        return scores
