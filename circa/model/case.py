"""
Define the data structure for algorithms as context
"""
import datetime
from typing import Dict
from typing import Sequence
from typing import Set

import numpy as np

from .data_loader import DataLoader
from .graph import Graph
from .graph import Node


class CaseData:
    # pylint: disable=too-many-instance-attributes
    """
    Case data that algorithms can access
    """

    def __init__(
        self,
        data_loader: DataLoader,
        sli: Node,
        detect_time: float,
        interval: datetime.timedelta = datetime.timedelta(minutes=1),
        lookup_window: int = 120,
        detect_window: int = 10,
        prune: bool = True,
    ):
        # pylint: disable=too-many-arguments
        self._data_loader = data_loader
        self._sli = sli
        self._detect_time = detect_time

        # Parameters for the algorithm
        self._interval = interval
        self._train_window = lookup_window - detect_window + 1
        self._test_window = detect_window
        self._lookup_window = lookup_window * interval.total_seconds()
        self._prune = prune

    @property
    def data_loader(self) -> DataLoader:
        """
        Interface to access raw data
        """
        return self._data_loader

    @property
    def sli(self) -> Node:
        """
        The service level indicator (SLI) that is violated
        """
        return self._sli

    @property
    def detect_time(self) -> float:
        """
        Unix timestamp when the service level indicator (SLI) is violated
        """
        return self._detect_time

    @property
    def train_window(self) -> int:
        """
        Number of data points for learning the normal pattern
        """
        return self._train_window

    @property
    def test_window(self) -> int:
        """
        Number of data points for analyzing the fault
        """
        return self._test_window

    def load_data(
        self, graph: Graph = None, current: float = None
    ) -> Dict[Node, Sequence[float]]:
        """
        Parepare data
        """
        if current is None:
            current = self._detect_time
        else:
            current = max(current, self._detect_time)
        nodes = self._data_loader.nodes if graph is None else graph.nodes

        start = self._detect_time - self._lookup_window
        length = int((current - start) / self._interval.total_seconds()) + 1
        series: Dict[Node, Sequence[float]] = {}
        for node in nodes:
            node_data = self._data_loader.load(
                entity=node.entity,
                metric=node.metric,
                start=start,
                end=current,
                interval=self._interval,
            )
            if self._prune:
                if node_data and len(set(node_data)) > 1:
                    series[node] = node_data[:length]
            else:
                if not node_data:
                    node_data = np.zeros(length)
                series[node] = node_data[:length]
        return series


class Case:
    """
    Case data for evaluation
    """

    def __init__(self, data: CaseData, answer: Set[Node]):
        self._data = data
        self._answer = answer

    @property
    def data(self) -> CaseData:
        """
        Case data
        """
        return self._data

    @property
    def answer(self) -> Set[Node]:
        """
        Ground truth for this case
        """
        return self._answer
