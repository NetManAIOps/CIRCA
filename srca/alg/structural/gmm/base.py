"""
Predict one node on its parents with the Gaussian Mixture Model
"""
from abc import ABC
from typing import List
from typing import Sequence

import numpy as np
from sklearn.mixture import GaussianMixture


class GMM(ABC):
    """
    Gaussian Mixture Model
    """

    def sample(self, n_samples: int = 1) -> np.ndarray:
        """
        Sample from the model and return 1-D array
        """
        raise NotImplementedError


class SKLearnGMM(GMM):
    """
    GMM with sklearn
    """

    def __init__(self, samples: Sequence[float], gmm_k: int = 2, seed: int = None):
        self._model = GaussianMixture(
            n_components=min(len(samples), gmm_k),
            covariance_type="full",
            random_state=seed,
        )
        self._model.fit(samples)

    def sample(self, n_samples: int = 1) -> np.ndarray:
        samples = self._model.sample(n_samples=n_samples)[0]
        return samples.reshape(-1)


class GMMPredictor(ABC):
    """
    Predict one node on its parents with the Gaussian Mixture Model
    """

    def __init__(self, gmm_k: int = 2, seed: int = None):
        self._gmm_k = gmm_k
        self._seed = seed

    def train(self, train_x: np.ndarray, train_y: np.ndarray):
        """
        Train the model
        """
        raise NotImplementedError

    def predict(self, test_x: np.ndarray) -> List[GMM]:
        """
        Generate GMM
        """
        raise NotImplementedError
