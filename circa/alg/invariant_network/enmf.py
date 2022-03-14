"""
Ranking Causal Anomalies via Temporal and Dynamical Analysis on Vanishing Correlations

See KDD'16
https://doi.org/10.1145/2939672.2939765
"""
from itertools import combinations
import logging
from typing import Dict
from typing import Tuple

import numpy as np
from scipy.special import softmax


_ZERO = 1e-6


class ARX:
    # pylint: disable=invalid-name, too-many-instance-attributes
    """
    AutoRegressive eXogenous (ARX) model

    y(t) = \\sum_{i=1}^{n} a_{i} y(t - i) + \\sum_{i=0}^{m} b_{i} x(t - k - i) + d

    where 0 <= n, m, l <= k is a popular choice
    """

    def __init__(
        self,
        train_x: np.ndarray,
        train_y: np.ndarray,
        moment_2: float = None,
        n: int = 3,
        m: int = 2,
        k: int = 0,
    ):
        # pylint: disable=too-many-arguments
        """
        Fit the ARX model with the training data
        """
        self._n = n
        self._m = m
        self._k = k
        self._start = max(self._n, self._m + self._k)

        x, y = self._prepare_data(train_x=train_x, train_y=train_y)
        theta: np.ndarray = np.linalg.lstsq(x, y, rcond=None)[0]
        residuals: np.ndarray = y - x @ theta

        self._theta = theta
        self._epsilon_max: float = abs(residuals).max()

        if moment_2 is None:
            moment_2 = y.shape[0] * y.var()
        self._fitness: float = 1 - np.sqrt((residuals**2).sum() / moment_2)

    @property
    def epsilon_max(self) -> float:
        """
        Maximum training error
        """
        return self._epsilon_max

    @property
    def fitness(self) -> float:
        """
        Fitness on the training data
        """
        return self._fitness

    def _prepare_data(self, train_x: np.ndarray, train_y: np.ndarray):
        num_samples = train_y.size
        y: np.ndarray = train_y[self._start :]
        x = np.array(
            [train_y[self._start - i : -i] for i in range(1, self._n + 1)]
            + [
                train_x[self._start - self._k - i : num_samples - self._k - i]
                for i in range(self._m + 1)
            ]
            + [np.ones(train_y.size - self._start)]
        ).T
        return x, y

    def residuals(self, test_x: np.ndarray, test_y: np.ndarray) -> np.ndarray:
        """
        Residuals on the testing data
        """
        x, y = self._prepare_data(train_x=test_x, train_y=test_y)
        return y - x @ self._theta


class InvariantNetwork:
    """
    Learn invariant network based on the AutoRegressive eXogenous (ARX) model
    """

    LOWER_THRESHOLD = 0.5
    UPPER_THRESHOLD = 0.7

    BROKEN_THRESHOLD = 1.1

    def __init__(self, n: int = 3, m: int = 2, k: int = 0):
        # pylint: disable=invalid-name
        """
        (n, m) is the order of the ARX model
        """
        self._arx_params = dict(n=n, m=m, k=k)
        self._start = max(n, m + k)
        self._edges: Dict[Tuple[int, int], ARX] = None
        self._num_nodes = 0

    @property
    def start(self) -> int:
        """
        The number of samples that will not be predicted by the ARX model,
            which is max(n, m + k)
        """
        return self._start

    def fit(self, data: np.ndarray):
        """
        Search invariant in a brute-force way

        The judgement of invariant edge comes from the code[2] of Reference[1] in KDD'16

        References:
        [1] Ranking Causal Anomalies via Temporal and Dynamical Analysis on Vanishing
            Correlations. https://doi.org/10.1145/2939672.2939765
        [2] gRank-mRank.zip:gRank-mRank/code/pruning/Scratch.m in
            https://github.com/chengw07/CausalRanking
        """
        edges = {}
        self._num_nodes = data.shape[1]

        valid_data: np.ndarray = data[self._start :, :]
        moment_2: np.ndarray = valid_data.shape[0] * valid_data.var(axis=0)
        for i, j in combinations(np.where(moment_2)[0], 2):
            arx_ij = ARX(
                train_x=data[:, j],
                train_y=data[:, i],
                moment_2=moment_2[i],
                **self._arx_params,
            )
            arx_ji = ARX(
                train_x=data[:, i],
                train_y=data[:, j],
                moment_2=moment_2[j],
                **self._arx_params,
            )
            if (
                min(arx_ij.fitness, arx_ji.fitness) >= self.LOWER_THRESHOLD
                and max(arx_ij.fitness, arx_ji.fitness) >= self.UPPER_THRESHOLD
            ):
                if arx_ji.fitness >= arx_ij.fitness:
                    edges[(j, i)] = arx_ji
                else:
                    edges[(i, j)] = arx_ij

        self._edges = edges

    def invariant_matrix(self, discrete: bool = True) -> np.ndarray:
        """
        The symmetric matrix of the invariant network
        """
        matrix = np.zeros((self._num_nodes, self._num_nodes), dtype=float)
        if discrete:
            for i, j in self._edges:
                matrix[i, j] = matrix[j, i] = 1
        else:
            for (i, j), arx in self._edges.items():
                matrix[i, j] = matrix[j, i] = arx.fitness
        return matrix

    def broken_matrix(self, data: np.ndarray, discrete: bool = True) -> np.ndarray:
        """
        The symmetric matrix of the broken network

        Notice:
            The code[3] of Reference[1] in KDD'16 uses 1.1 times of
            the 99.5 percentiles of training residuals as the threshold

        References:
        [1] Ranking Causal Anomalies via Temporal and Dynamical Analysis on Vanishing
            Correlations. https://doi.org/10.1145/2939672.2939765
        [3] gRank-mRank.zip:gRank-mRank/code/ranking(realdata)/InvTracking.m in
            https://github.com/chengw07/CausalRanking
        """
        assert data.shape[1] == self._num_nodes
        matrix = np.zeros((self._num_nodes, self._num_nodes), dtype=float)
        for (i, j), arx in self._edges.items():
            residuals = arx.residuals(test_x=data[:, j], test_y=data[:, i])
            normalized_residual = np.nanmax(abs(residuals) / arx.epsilon_max)
            if normalized_residual > self.BROKEN_THRESHOLD:
                matrix[i, j] = matrix[j, i] = 1 if discrete else normalized_residual
        return matrix


class ENMF:
    """
    ENMF is edited based on the code[4] of Reference[1] in KDD'16

    References:
    [1] Ranking Causal Anomalies via Temporal and Dynamical Analysis on Vanishing
        Correlations. https://doi.org/10.1145/2939672.2939765
    [4] rankingCausal.zip:rankingCausal/optENMF.m in
        https://github.com/chengw07/CausalRanking
    """

    def __init__(
        self,
        gamma: float = 0.5,
        tau: float = 1.0,
        epoches: int = 500,
        invariant_network: InvariantNetwork = None,
    ):
        """
        Constructor

        Parameters:
            gamma: A parameter to balance
                the award for neighboring nodes to have similar status scores, and
                the penalty of large bias from the initial seeds
            tau: A larger tau typically results in more zeros in e
        """
        self._gamma = gamma
        self._tau = tau

        self._epoches = epoches

        if invariant_network is None:
            self._invariant_network = InvariantNetwork()
        else:
            self._invariant_network = invariant_network

        self._logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )

    @staticmethod
    def degree_normalize(matrix: np.ndarray):
        """
        Calculate D^{-0.5} A D^{-0.5},
        where D is a diagonal matrix with D_{xx} = \\sum_{y} A_{xy}
        """
        d_root = np.diag(
            [0 if item == 0 else 1 / item for item in np.sqrt(matrix.sum(axis=1))]
        )
        return d_root * matrix * d_root

    @property
    def start(self) -> int:
        """
        The number of samples that will not be predicted by the ARX model,
            which is max(n, m + k)
        """
        return self._invariant_network.start

    def _loss(
        self,
        broken: np.ndarray,
        causes: np.ndarray,
        status_scores: np.ndarray,
        mask: np.ndarray,
    ) -> Tuple[float, float]:
        # Square of Frobenius norm
        reconstruct_loss: float = (
            np.linalg.norm(
                mask * (status_scores @ status_scores.T) - broken,
                ord="fro",
            )
            ** 2
        )
        loss: float = reconstruct_loss + self._tau * abs(causes).sum()
        return loss, reconstruct_loss

    def enmf(
        self,
        invariant: np.ndarray,
        broken: np.ndarray,
        causes: np.ndarray,
        weight: np.ndarray = None,
    ):
        # pylint: disable=too-many-locals
        """
        Causal Anomaly Ranking with an iterative updating algorithm
        """
        # A: invariant network
        # B: broken network
        # e: causes
        # H: propagation with r = H e, where r is the anomaly score vector
        # H = (1 - c) (I_n - c \tilde{A})^{-1}
        propagation: np.ndarray = (1 - self._gamma) * np.linalg.inv(
            np.eye(invariant.shape[0]) - self._gamma * self.degree_normalize(invariant)
        )
        mask = invariant.astype(bool).astype(float)

        def _status_scores(causes: np.ndarray) -> np.ndarray:
            if weight is None:
                return propagation @ causes
            return propagation @ (weight * causes)

        for epoch in range(self._epoches):
            causes_old = causes
            status_scores = _status_scores(causes)
            numerator: np.ndarray = (
                4 * (mask * (propagation.T @ broken)) @ status_scores
            )
            denominator: np.ndarray = (
                4
                * (mask * (propagation.T @ status_scores @ status_scores.T))
                @ status_scores
                + self._tau
            )
            causes = causes * ((numerator / denominator) ** 0.25)
            err: float = np.linalg.norm(causes - causes_old, ord="fro")

            loss, reconstruct_loss = self._loss(
                broken=broken,
                causes=causes,
                status_scores=_status_scores(causes),
                mask=mask,
            )

            self._logger.debug(
                "Epoch of ENMF: %04d, loss: %06f, reconstruct_loss: %06f, change: %06f",
                epoch,
                loss,
                reconstruct_loss,
                err,
            )
            if err < _ZERO:
                break
        return causes

    def train(self, data: np.ndarray):
        """
        Learn the invariant network
        """
        self._invariant_network.fit(data)

    def _rank(self, invariant: np.ndarray, broken: np.ndarray) -> np.ndarray:
        causes = np.ones((invariant.shape[0], 1))
        causes = self.enmf(invariant=invariant, broken=broken, causes=causes)
        return causes[:, 0]

    def rank(self, data: np.ndarray, discrete: bool = True) -> np.ndarray:
        """
        Detect the broken network and rank nodes

        Returns:
            An array of causal anomaly scores for each node
        """
        invariant = self._invariant_network.invariant_matrix(discrete=discrete)
        if np.all(invariant == 0):
            return np.zeros(invariant.shape[0])
        broken = self._invariant_network.broken_matrix(data=data, discrete=discrete)

        return self._rank(invariant=invariant, broken=broken)


class ENMFSoft(ENMF):
    """
    ENMF with softmax normalization

    ENMFSoft is edited based on the code[5] of Reference[1] in KDD'16

    References:
    [1] Ranking Causal Anomalies via Temporal and Dynamical Analysis on Vanishing
        Correlations. https://doi.org/10.1145/2939672.2939765
    [5] rankingCausal.zip:rankingCausal/optENMFSoft.m in
        https://github.com/chengw07/CausalRanking
    """

    def _rank(self, invariant: np.ndarray, broken: np.ndarray) -> np.ndarray:
        # pylint: disable=too-many-locals
        # A: invariant network
        # B: broken network
        # e: causes
        # H: propagation with r = H e, where r is the anomaly score vector
        # H = (1 - c) (I_n - c \tilde{A})^{-1}
        propagation: np.ndarray = (1 - self._gamma) * np.linalg.inv(
            np.eye(invariant.shape[0]) - self._gamma * self.degree_normalize(invariant)
        )
        mask = invariant.astype(bool).astype(float)

        causes = np.ones((invariant.shape[0], 1))
        status_scores = softmax(propagation @ causes)
        loss, _ = self._loss(
            broken=broken, causes=causes, status_scores=status_scores, mask=mask
        )

        for epoch in range(self._epoches):
            loss_old = loss
            status_scores: np.ndarray = softmax(propagation @ causes)
            phi = np.diag(status_scores) - status_scores @ status_scores.T

            numerator: np.ndarray = (
                4 * (mask * (propagation.T @ phi @ broken)) @ status_scores
            )
            numerator[numerator < 0] = 0
            denominator: np.ndarray = (
                4
                * (mask * (propagation.T @ phi @ status_scores @ status_scores.T))
                @ status_scores
                + self._tau
            )
            causes = causes * ((numerator / denominator) ** 0.25)
            loss, reconstruct_loss = self._loss(
                broken=broken, causes=causes, status_scores=status_scores, mask=mask
            )
            err = abs(loss - loss_old)

            self._logger.debug(
                "Epoch of ENMF: %04d, loss: %06f, reconstruct_loss: %06f, change: %06f",
                epoch,
                loss,
                reconstruct_loss,
                err,
            )
            if err < _ZERO:
                break
        return causes[:, 0]
