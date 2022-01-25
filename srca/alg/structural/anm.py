"""
Regress one node on its parents with an Additive Noise Model
"""
import numpy as np
from sklearn.linear_model._base import LinearModel
from sklearn.linear_model import LinearRegression

from .base import Regressor


class ANMRegressor(Regressor):
    """
    Regress one node on its parents with an Additive Noise Model
    assuming x = f(pa(X)) + e and e follows a normal distribution
    """

    def __init__(self, regressor: LinearModel = None, **kwargs):
        super().__init__(**kwargs)
        self._regressor = regressor if regressor else LinearRegression()

    def _score(
        self,
        train_x: np.ndarray,
        test_x: np.ndarray,
        train_y: np.ndarray,
        test_y: np.ndarray,
    ) -> np.ndarray:
        self._regressor.fit(train_x, train_y)
        train_err: np.ndarray = train_y - self._regressor.predict(train_x)
        test_err: np.ndarray = test_y - self._regressor.predict(test_x)
        return self._zscore(train_y=train_err, test_y=test_err)
