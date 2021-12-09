"""
Wrapper for PCMCI
"""
from typing import List
from typing import Tuple

import numpy as np
from tigramite import data_processing
from tigramite.independence_tests import ParCorr
from tigramite.independence_tests.independence_tests_base import CondIndTest
from tigramite.pcmci import PCMCI

from .base import DynamicGraphFactory
from ...model.graph import Node


class _ParCorr(ParCorr):
    # pylint: disable=invalid-name
    """
    Wrap ParCorr to handle constant
    """

    def _get_single_residuals(
        self,
        array: np.ndarray,
        target_var: int,
        standardize: bool = True,
        return_means: bool = False,
    ) -> np.ndarray:

        dim, _ = array.shape
        dim_z = dim - 2

        # Standardize
        if standardize:
            array -= array.mean(axis=1).reshape(dim, 1)
            # Skip constant variables
            std: np.ndarray = array.std(axis=1)
            for var in np.where(std)[0]:
                array[var, :] /= std[var]
            if np.isnan(array).sum() != 0:
                raise ValueError(
                    "nans after standardizing, " "possibly constant array!"
                )

        y = array[target_var, :]

        if dim_z > 0:
            z = np.fastCopyAndTranspose(array[2:, :])
            beta_hat = np.linalg.lstsq(z, y, rcond=None)[0]
            mean = np.dot(z, beta_hat)
            resid = y - mean
        else:
            resid = y
            mean = None

        if return_means:
            return (resid, mean)
        return resid


class PCTSFactory(DynamicGraphFactory):
    """
    Mine the graph with path condition time series (PCTS [1]) based on PCMCI [2]

    References
    [1] MicroCause. IWQoS'20. https://doi.org/10.1109/IWQoS49365.2020.9213058
    [2] PCMCI. Science Advances'19. https://doi.org/10.1126/sciadv.aau4996
    """

    def __init__(
        self,
        alpha: float = 0.05,
        cond_ind_test: CondIndTest = None,
        tau_max: int = 3,
        max_conds_dim: int = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if cond_ind_test is None:
            cond_ind_test = _ParCorr(significance="analytic")
        self._alpha = alpha
        self._cond_ind_test = cond_ind_test
        self._tau_max = tau_max
        self._max_conds_dim = max_conds_dim

    @staticmethod
    def _gather_tau(p_matrix: np.ndarray) -> np.ndarray:
        # reason, result, tau
        num, result_num, _ = p_matrix.shape
        assert num == result_num

        link_matrix = []
        for reason in range(num):
            link_matrix.append([min(p_matrix[reason][result]) for result in range(num)])
        return np.array(link_matrix)

    def _create(
        self, data: np.ndarray, nodes: List[Node]
    ) -> Tuple[np.ndarray, List[Node]]:
        dataframe = data_processing.DataFrame(data)
        pcmci = PCMCI(
            dataframe=dataframe, cond_ind_test=self._cond_ind_test, verbosity=0
        )
        report = pcmci.run_pcmci(
            tau_max=self._tau_max,
            pc_alpha=self._alpha,
            max_conds_dim=self._max_conds_dim,
        )
        matrix = self._gather_tau(report["p_matrix"])
        return matrix, nodes
