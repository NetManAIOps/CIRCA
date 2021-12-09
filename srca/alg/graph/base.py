"""
Base GraphFactory based on data mining
"""
from typing import List
from typing import Tuple

import networkx as nx
import numpy as np

from ..base import GraphFactory
from ...model.case import CaseData
from ...model.graph import Graph
from ...model.graph import Node
from ...model.graph import MemoryGraph


class DynamicGraphFactory(GraphFactory):
    """
    Create graph mined from data
    """

    def _create(
        self, data: np.ndarray, nodes: List[Node]
    ) -> Tuple[np.ndarray, List[Node]]:
        """
        Mine graph from data

        Return:
            matrix: if matrix[i, j] is True, i may be one of the causes of j
            nodes: mapping from the matrix indexes to Nodes
        """
        raise NotImplementedError

    def create(self, data: CaseData, current: float) -> Graph:
        series = data.load_data(current=current)
        nodes = list(series.keys())
        series = np.array([series[node] for node in nodes]).T

        matrix, nodes = self._create(data=series, nodes=nodes)

        graph = nx.DiGraph()
        graph.add_nodes_from(nodes)
        graph.add_node(data.sla)
        graph.add_edges_from(
            (nodes[cause], nodes[effect]) for cause, effect in zip(*np.where(matrix))
        )
        return MemoryGraph(graph)
