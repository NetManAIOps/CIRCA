"""
Wrapper for R tools
"""
from enum import Enum
import os
from typing import List
from typing import Tuple

import numpy as np
from rpy2 import robjects
from rpy2.robjects import numpy2ri
from rpy2.robjects import default_converter
from rpy2.robjects.conversion import localconverter

from .utils import bcp
from ..common import DynamicGraphFactory
from ...model.graph import Node


robjects.r(
    f"source(\"{os.path.dirname(__file__).replace(os.path.sep, '/')}/causal.R\")"
)


class PCAlgFactory(DynamicGraphFactory):
    """
    Mine the graph with the R package pcalg
    """

    class _Method(Enum):

        PC_GAUSS = "PC-gauss"
        PC_GSQ = "PC-gsq"

    def __init__(
        self,
        alpha: float = 0.05,
        method: str = _Method.PC_GAUSS.value,
        tau_max: int = 3,
        max_conds_dim: int = np.inf,
        num_cores: int = 1,
        **kwargs,
    ):
        # pylint: disable=too-many-arguments
        """
        alpha: desired significance level in (0, 1)
        method: name of the method to be used
            - "PC-gauss": PC algorithm with gaussCItest
            - "PC-gsq": PC algorithm with disCItest
        max_conds_dim: Maximum size of a condition set, unrestricted by default.
        """
        super().__init__(**kwargs)
        self._alpha = alpha
        assert method in {_method.value for _method in self._Method}
        self._method = self._Method(method)
        self._tau_max = tau_max
        self._max_conds_dim = max_conds_dim
        self._num_cores = num_cores

    def _create(
        self, data: np.ndarray, nodes: List[Node]
    ) -> Tuple[np.ndarray, List[Node]]:
        if self._method == self._Method.PC_GSQ:
            series = {
                node: bcp(data[:, index], tau_max=self._tau_max)
                for index, node in enumerate(nodes)
            }
            nodes = [
                node
                for node, change_points in series.items()
                if len(np.unique(change_points)) > 1
            ]
            if len(nodes) < 2:
                return np.zeros((0, 0)), nodes
            params = {
                "d": np.array([series[node] for node in nodes]).T,
                "CItest": "gsq",
                "alpha": self._alpha,
                "m.max": self._max_conds_dim,
                "numCores": self._num_cores,
            }
            fun = robjects.r["runPC"]
        else:  # self._method == self._Method.PC_GAUSS
            params = {
                "d": data,
                "CItest": "gauss",
                "alpha": self._alpha,
                "m.max": self._max_conds_dim,
                "numCores": self._num_cores,
            }
            fun = robjects.r["runPC"]

        with localconverter(default_converter + numpy2ri.converter):
            matrix: np.ndarray = fun(**params)
        return matrix, nodes
