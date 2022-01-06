"""
Regress one node on its parents with the random forest
"""
import numpy as np

from .base import Regressor
from .prob_rf import ProbRF


class ForestRegressor(Regressor):
    """
    Regress one node on its parents with the random forest
    """

    def __init__(self, regressor: ProbRF = None, seed: int = None, **kwargs):
        super().__init__(**kwargs)
        self._regressor = regressor if regressor else ProbRF(seed=seed)

    def _score(
        self,
        train_x: np.ndarray,
        test_x: np.ndarray,
        train_y: np.ndarray,
        test_y: np.ndarray,
    ) -> np.ndarray:
        self._regressor.fit(train_x, train_y)
        return np.array(
            [
                self._zscore(np.array(samples), target)[0]
                for samples, target in zip(self._regressor.predict(test_x), test_y)
            ]
        )
