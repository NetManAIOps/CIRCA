"""
Depth-first searching alone anomalies
"""
from typing import Dict
from typing import Sequence
from typing import Set

import numpy as np

from .base import Score
from .base import Scorer
from .common import pearson
from ..model.case import CaseData
from ..model.graph import Graph
from ..model.graph import Node


class DFSScorer(Scorer):
    """
    Filter in anomalous nodes that do not have anomalous parents in the graph,
    through searching from the SLI.
    """

    def __init__(self, anomaly_threshold: float, **kwargs):
        super().__init__(**kwargs)
        self._anomaly_threshold = anomaly_threshold

    def _extended(
        self, node: Node, parent: Node, anomalies: Set[Node], data: CaseData
    ) -> bool:
        # pylint: disable=unused-argument, no-self-use
        return parent in anomalies

    def score(
        self,
        graph: Graph,
        data: CaseData,
        current: float,
        scores: Dict[Node, Score] = None,
    ) -> Dict[Node, Score]:
        anomalies = {
            node
            for node, score in scores.items()
            if score.score >= self._anomaly_threshold
        }
        layer = {data.sli}
        visited = {data.sli}
        roots: Set[Node] = set()
        while layer:
            next_layer: Set[Node] = set()
            for node in layer:
                is_root = True
                for parent in graph.parents(node):
                    if parent in visited:
                        is_root = False
                    elif self._extended(node, parent, anomalies, data):
                        is_root = False
                        visited.add(parent)
                        next_layer.add(parent)
                if is_root:
                    roots.add(node)
            layer = next_layer

        return {node: scores[node] for node in roots if node in scores}


class MicroHECLScorer(DFSScorer):
    """
    Extend DFSScorer, stopping if the correlation between two linked nodes
    is lower than the given threshold.

    See MicroHECL in ICSE-SEIP'21
    https://doi.org/10.1109/ICSE-SEIP52600.2021.00043

    ATTENTION: This implementation is not thread-safe
    """

    def __init__(self, stop_threshold: float = 0.7, **kwargs):
        super().__init__(**kwargs)
        self._stop_threshold = stop_threshold
        self._data: Dict[Node, Sequence[float]] = None

    def _extended(
        self, node: Node, parent: Node, anomalies: Set[Node], data: CaseData
    ) -> bool:
        if parent not in anomalies:
            return False
        series_a = np.array(self._data[node][-data.test_window :])
        series_b = np.array(self._data[parent][-data.test_window :])
        correlation = pearson(series_a, series_b)
        return correlation >= self._stop_threshold

    def score(
        self,
        graph: Graph,
        data: CaseData,
        current: float,
        scores: Dict[Node, Score] = None,
    ) -> Dict[Node, Score]:
        self._data = data.load_data(graph, current)
        scores = super().score(graph=graph, data=data, current=current, scores=scores)
        self._data = None
        return scores
