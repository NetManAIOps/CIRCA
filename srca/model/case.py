"""
Define the data structure for algorithms as context
"""
import datetime
from typing import Dict
from typing import Sequence
from typing import Set

from .data_loader import DataLoader
from .graph import Graph
from .graph import Node


class CaseData:
    """
    Case data that algorithms can access
    """

    def __init__(
        self,
        data_loader: DataLoader,
        sla: Node,
        detect_time: float,
        interval: datetime.timedelta = datetime.timedelta(minutes=1),
        lookup_window: int = 120,
        detect_window: int = 10,
    ):
        # pylint: disable=too-many-arguments
        self._data_loader = data_loader
        self._sla = sla
        self._detect_time = detect_time

        # Parameters for the algorithm
        self._interval = interval
        self._train_window = lookup_window - detect_window + 1
        self._test_window = detect_window
        self._lookup_window = lookup_window * interval.total_seconds()

    @property
    def data_loader(self) -> DataLoader:
        """
        Interface to access raw data
        """
        return self._data_loader

    @property
    def sla(self) -> Node:
        """
        The service level agreement (SLA) that is violated
        """
        return self._sla

    @property
    def detect_time(self) -> float:
        """
        Unix timestamp when the service level agreement (SLA) is violated
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

    def load_data(self, graph: Graph, current: float) -> Dict[Node, Sequence[float]]:
        """
        Parepare data
        """
        current = max(current, self._detect_time)

        start = self._detect_time - self._lookup_window
        series: Dict[Node, Sequence[float]] = {}
        for node in graph.nodes:
            node_data = self._data_loader.load(
                entity=node.entity,
                metric=node.metric,
                start=start,
                end=current,
                interval=self._interval,
            )
            if node_data:
                series[node] = node_data
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