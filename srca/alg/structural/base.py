"""
Abstract interfaces to regress one node on its parents
"""
from abc import ABC
import logging
from typing import Dict
from typing import Sequence

import numpy as np

from ..base import Score
from ..common import DecomposableScorer
from ..common import zscore
from ...model.case import CaseData
from ...model.graph import Graph
from ...model.graph import Node


def dzscore(
    train_y: np.ndarray, test_y: np.ndarray, minimum_std: float = 1
) -> np.ndarray:
    """
    Estimate to what extend each value in test_y violates
    the normal distribution defined by train_y

    Set minimum for the standard deviation
    """
    mean: float = train_y.mean()
    std: float = train_y.std()
    std = max(std, minimum_std, train_y.max() * 0.1)
    _zscore = (test_y.reshape(-1) - mean) / std
    return _zscore


class DiscreteNSigmaScorer(DecomposableScorer):
    """
    Score nodes by dzscore
    """

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
        z_scores = dzscore(train_y, test_y)
        z_score = self._aggregator(abs(z_scores))
        score = Score(z_score)
        score["z-score"] = z_score
        return score


class Regressor(ABC):
    """
    Regress one node on its parents, assuming x ~ P(x | pa(X))
    """

    def __init__(
        self,
        use_discrete: bool = False,
    ):
        self._use_discrete = use_discrete

        klass = self.__class__
        self._logger = logging.getLogger(f"{klass.__module__}.{klass.__name__}")

    def _zscore(
        self, train_y: np.ndarray, test_y: np.ndarray, ref_y: np.ndarray = None
    ) -> np.ndarray:
        if self._use_discrete:
            if ref_y is None:
                return dzscore(train_y=train_y, test_y=test_y)
            return dzscore(
                train_y=train_y, test_y=test_y, minimum_std=max(1, ref_y.max() * 0.1)
            )
        return zscore(train_y=train_y, test_y=test_y)

    def _score(
        self,
        train_x: np.ndarray,
        test_x: np.ndarray,
        train_y: np.ndarray,
        test_y: np.ndarray,
    ) -> np.ndarray:
        raise NotImplementedError

    def score(
        self,
        train_x: np.ndarray,
        test_x: np.ndarray,
        train_y: np.ndarray,
        test_y: np.ndarray,
    ) -> np.ndarray:
        """
        Estimate to what extend each value in test_y violates regression
        """
        if len(train_x) == 0:
            return self._zscore(train_y=train_y, test_y=test_y)
        try:
            return self._score(
                train_x=train_x, test_x=test_x, train_y=train_y, test_y=test_y
            )
        except ValueError as err:
            self._logger.warning(err, exc_info=True)
            return self._zscore(train_y=train_y, test_y=test_y)
