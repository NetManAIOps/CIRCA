"""
Cluster Ranking based fault Diagnosis

See ICDM'17
https://doi.org/10.1109/ICDM.2017.129
"""
from itertools import combinations
import logging
from typing import Dict
from typing import Tuple

import numpy as np
import torch
from torch.optim import SGD


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
        self._fitness: float = 1 - np.sqrt((residuals ** 2).sum() / moment_2)

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


class CRD:
    # pylint: disable=invalid-name, too-many-instance-attributes
    """
    Cluster Ranking based fault Diagnosis
    """

    def __init__(
        self,
        num_cluster: int = 2,
        alpha: float = 1.2,
        beta: float = 0.1,
        lambda1: float = 0.1,
        gamma: float = 0.5,
        tau: float = 1.0,
        epoches: int = 3000,
        learning_rate: float = 0.1,
        invariant_network: InvariantNetwork = None,
        cuda: bool = False,
        seed: int = None,
    ):
        # pylint: disable=too-many-arguments
        """
        Constructor

        Parameters:
            num_cluster: The number of clusters in an invariant network
            alpha: A parameter in the Dirichlet distribution with alpha >= 1
            beta: A parameter to balance the object functions for network clustering
                and broken score
            lambda1: l1 penalty parameter
            gamma: A parameter to balance
                the award for neighboring nodes to have similar status scores, and
                the penalty of large bias from the initial seeds
            tau: A larger tau typically results in more zeros in E
        """
        if seed is not None:
            torch.manual_seed(seed)
            if cuda:
                torch.cuda.manual_seed(seed)

        # Parameters for broken cluster identification
        self._num_cluster = num_cluster
        self._alpha = alpha
        self._beta = beta
        self._lambda1 = lambda1

        # Parameters for causal anomaly ranking
        self._gamma = gamma
        self._tau = tau

        self._epoches = epoches
        self._learning_rate = learning_rate
        self._cuda = cuda

        if invariant_network is None:
            self._invariant_network = InvariantNetwork()
        else:
            self._invariant_network = invariant_network

        self._logger = logging.getLogger(
            f"{self.__class__.__module__}.{self.__class__.__name__}"
        )

    @staticmethod
    def degree_normalize(A: torch.FloatTensor):
        """
        Calculate D^{-0.5} A D^{-0.5},
        where D is a diagonal matrix with D_{xx} = \\sum_{y} A_{xy}
        """
        diag = torch.FloatTensor(
            [0 if item == 0 else 1 / item for item in A.sum(dim=1).sqrt().cpu()],
            device=A.device,
        )
        d_root = torch.diag(diag)
        return d_root * A * d_root

    @property
    def start(self) -> int:
        """
        The number of samples that will not be predicted by the ARX model,
            which is max(n, m + k)
        """
        return self._invariant_network.start

    def _cluster_broken(self, A: torch.FloatTensor, B: torch.FloatTensor):
        """
        Broken Cluster Identification

        Parameters:
            A: invariant_matrix
            B: broken_matrix
        Returns:
            U: Cluster membership matrix with the shape of (num_nodes, num_cluster)
            s: Broken score vector with the shape of (num_nodes, )
        """
        # TODO: Get rid of torch
        # As the authors state that, they "take an alternating minimization framework
        # that alternately solves U and s until a stationary point is achieved" to
        # identify broken clusters.

        # The indicator matrix, with W_{xy} = 1 iff (x, y) is invariant but not broken
        W = (A > 0).float() - (B > 0).float()
        U_origin = torch.randn(A.size()[0], self._num_cluster, requires_grad=True)
        s_origin = torch.rand(self._num_cluster, requires_grad=True)
        if self._cuda:
            U_origin = U_origin.cuda()
            s_origin = s_origin.cuda()
        optimizer = SGD([U_origin, s_origin], lr=self._learning_rate, momentum=0.1)

        for epoch in range(self._epoches):
            optimizer.zero_grad()
            U, s = torch.softmax(U_origin, dim=1), torch.sigmoid(s_origin)

            # Eq.(4)
            A_hat = (U / torch.sum(U, dim=0, keepdim=True)) @ U.transpose(0, 1)
            # Eq.(6)
            J_A = -torch.sum(A * torch.log(A_hat)) - (self._alpha - 1) * torch.sum(
                torch.log(U)
            )
            P = (U * s.view(1, -1)) @ U.transpose(0, 1)  # Eq.(7)
            # Eq.(10)
            J_B = -torch.sum(B * torch.log(P)) - torch.sum(W * torch.log(1 - P))

            J_CR: torch.FloatTensor = (
                J_A + self._beta * J_B + self._lambda1 * U_origin.norm(p=1)
            )
            J_CR.backward()
            optimizer.step()

            self._logger.debug(
                "Epoch of Broken Cluster Identification: %04d, "
                "J_A: %06f, J_B: %06f, J_CR: %06f",
                epoch,
                J_A.item(),
                J_B.item(),
                J_CR.item(),
            )
        with torch.no_grad():
            return torch.softmax(U_origin, dim=1), torch.sigmoid(s_origin)

    def _rank_anomaly_sgd(
        self,
        A: torch.FloatTensor,
        B: torch.FloatTensor,
        U: torch.FloatTensor,
        H: torch.FloatTensor,
    ):
        """
        Causal Anomaly Ranking with Stochastic Gradient Descent
        """
        C = (A > 0).float()

        E_origin = torch.randn(A.size()[0], self._num_cluster, requires_grad=True)
        if self._cuda:
            E_origin = E_origin.cuda()
        optimizer = SGD([E_origin], lr=self._learning_rate)

        for epoch in range(self._epoches):
            optimizer.zero_grad()
            E = torch.sigmoid(E_origin)

            status_scores = H @ (U * E)
            # Square of Frobenius norm
            reconstruct_loss: torch.FloatTensor = (
                C * (status_scores @ status_scores.transpose(0, 1)) - B
            ).norm(p="fro") ** 2
            J_H: torch.FloatTensor = reconstruct_loss + self._tau * E.norm(p=1)
            J_H.backward()
            optimizer.step()

            self._logger.debug(
                "Epoch of Causal Anomaly Ranking: %04d, "
                "J_H: %06f, reconstruct_loss: %06f",
                epoch,
                J_H.item(),
                reconstruct_loss.item(),
            )
        with torch.no_grad():
            return torch.sigmoid(E_origin)

    def _rank_anomaly_enmf(
        self,
        A: torch.FloatTensor,
        B: torch.FloatTensor,
        U: torch.FloatTensor,
        H: torch.FloatTensor,
    ):
        """
        Causal Anomaly Ranking with an iterative updating algorithm

        This method is edited based on the code[4] of Reference[1] in KDD'16

        References:
        [1] Ranking Causal Anomalies via Temporal and Dynamical Analysis on Vanishing
            Correlations. https://doi.org/10.1145/2939672.2939765
        [4] rankingCausal.zip:rankingCausal/optENMF.m in
            https://github.com/chengw07/CausalRanking
        """
        C = (A > 0).float()
        e = torch.ones(A.size()[0], self._num_cluster)
        for epoch in range(self._epoches):
            e_old = e
            status_scores = H @ (U * e)
            numerator = 4 * (C * (H.T @ B)) @ status_scores
            denominator = (
                4 * (C * (H.T @ status_scores @ status_scores.T)) @ status_scores
                + self._tau
            )
            e = e * ((numerator / denominator) ** 0.25)
            err: torch.FloatTensor = (e - e_old).norm(p="fro")

            status_scores = H @ (U * e)
            # Square of Frobenius norm
            reconstruct_loss: torch.FloatTensor = (
                C * (status_scores @ status_scores.transpose(0, 1)) - B
            ).norm(p="fro") ** 2
            J_H: torch.FloatTensor = reconstruct_loss + self._tau * e.norm(p=1)

            self._logger.debug(
                "Epoch of Causal Anomaly Ranking: %04d, "
                "J_H: %06f, reconstruct_loss: %06f, change: %06f",
                epoch,
                J_H.item(),
                reconstruct_loss.item(),
                err.item(),
            )
            if err.item() < _ZERO:
                break
        return e

    def _rank_anomaly(
        self,
        A: torch.FloatTensor,
        B: torch.FloatTensor,
        U: torch.FloatTensor,
        use_sgd: bool = True,
    ):
        """
        Causal Anomaly Ranking
        """
        In = torch.eye(A.size()[0], device=A.device)
        H = (1 - self._gamma) * torch.inverse(
            In - self._gamma * self.degree_normalize(A)
        )
        if use_sgd:
            return self._rank_anomaly_sgd(A=A, B=B, U=U, H=H)
        return self._rank_anomaly_enmf(A=A, B=B, U=U, H=H)

    def train(self, data: np.ndarray):
        """
        Learn the invariant network
        """
        self._invariant_network.fit(data)

    def rank(
        self, data: np.ndarray, discrete: bool = True, use_sgd: bool = True
    ) -> np.ndarray:
        """
        Detect the broken network and rank nodes

        Returns:
            An array of causal anomaly scores for each node
        """
        A = self._invariant_network.invariant_matrix(discrete=discrete)
        if np.all(A == 0):
            return np.zeros(A.shape[0])
        B = self._invariant_network.broken_matrix(data=data, discrete=discrete)
        A, B = torch.FloatTensor(A), torch.FloatTensor(B)
        if self._cuda:
            A, B = A.cuda(), B.cuda()

        U, s = self._cluster_broken(A=A, B=B)
        E = self._rank_anomaly(A=A, B=B, U=U, use_sgd=use_sgd)
        scores = (U * E) @ s

        return scores.cpu().numpy()
