"""
Abstract interfaces to regress one node on its parents
"""
from abc import ABC
import logging

import numpy as np

from ..common import zscore


class Regressor(ABC):
    """
    Regress one node on its parents, assuming x ~ P(x | pa(X))
    """

    def __init__(self):
        klass = self.__class__
        self._logger = logging.getLogger(f"{klass.__module__}.{klass.__name__}")

    @staticmethod
    def _zscore(train_y: np.ndarray, test_y: np.ndarray) -> np.ndarray:
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
