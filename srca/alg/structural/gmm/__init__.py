"""
Regress one node on its parents with the Gaussian Mixture Model
"""
import numpy as np

from .base import GMMPredictor
from .prob_rf import ProbRF
from ..base import Regressor


class GMMRegressor(Regressor):
    """
    Regress one node on its parents with the Gaussian Mixture Model
    """

    def __init__(
        self,
        regressor: GMMPredictor = None,
        sample_size: int = 100,
        seed: int = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._regressor = regressor if regressor else ProbRF(seed=seed)
        self._sample_size = sample_size

    def _score(
        self,
        train_x: np.ndarray,
        test_x: np.ndarray,
        train_y: np.ndarray,
        test_y: np.ndarray,
    ) -> np.ndarray:
        self._regressor.train(train_x, train_y)
        return np.array(
            [
                self._zscore(gmm.sample(self._sample_size), target)[0]
                for gmm, target in zip(self._regressor.predict(test_x), test_y)
            ]
        )
