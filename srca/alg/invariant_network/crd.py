"""
Cluster Ranking based fault Diagnosis

See ICDM'17
https://doi.org/10.1109/ICDM.2017.129
"""
import numpy as np
import torch
from torch.optim import SGD

from .enmf import ENMF
from .enmf import InvariantNetwork


class CRD(ENMF):
    # pylint: disable=invalid-name
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
        use_sgd: bool = True,
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
        super().__init__(
            gamma=gamma, tau=tau, epoches=epoches, invariant_network=invariant_network
        )
        if seed is not None:
            torch.manual_seed(seed)
            if cuda:
                torch.cuda.manual_seed(seed)

        # Parameters for broken cluster identification
        self._num_cluster = num_cluster
        self._alpha = alpha
        self._beta = beta
        self._lambda1 = lambda1

        self._learning_rate = learning_rate
        self._use_sgd = use_sgd
        self._cuda = cuda

    @property
    def _device(self):
        return "cuda" if self._cuda else "cpu"

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
        U_origin = torch.randn(
            A.size()[0], self._num_cluster, requires_grad=True, device=self._device
        )
        s_origin = torch.rand(
            self._num_cluster, requires_grad=True, device=self._device
        )
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
    ):
        """
        Causal Anomaly Ranking with Stochastic Gradient Descent
        """
        H = torch.FloatTensor(self.degree_normalize(A.cpu().numpy()))
        if self._cuda:
            H = H.cuda()
        C = (A > 0).float()

        E_origin = torch.randn(
            A.size()[0], self._num_cluster, requires_grad=True, device=self._device
        )
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

    def _rank_anomaly(
        self,
        A: torch.FloatTensor,
        B: torch.FloatTensor,
        U: torch.FloatTensor,
    ):
        """
        Causal Anomaly Ranking
        """
        if self._use_sgd:
            return self._rank_anomaly_sgd(A=A, B=B, U=U)
        E = super().enmf(
            invariant=A.numpy(),
            broken=B.numpy(),
            causes=np.ones((A.shape[0], self._num_cluster)),
            weight=U.cpu().numpy(),
        )
        return torch.FloatTensor(E, device=U.device)

    def rank(self, data: np.ndarray, discrete: bool = True) -> np.ndarray:
        A = self._invariant_network.invariant_matrix(discrete=discrete)
        if np.all(A == 0):
            return np.zeros(A.shape[0])
        B = self._invariant_network.broken_matrix(data=data, discrete=discrete)
        A, B = torch.FloatTensor(A), torch.FloatTensor(B)
        if self._cuda:
            A, B = A.cuda(), B.cuda()

        U, s = self._cluster_broken(A=A, B=B)
        E = self._rank_anomaly(A=A, B=B, U=U)
        scores = (U * E) @ s

        return scores.cpu().numpy()
