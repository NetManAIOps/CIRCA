"""
Define the data structure for algorithms as context
"""
from typing import Set

from .data_loader import DataLoader
from .graph import Node


class CaseData:
    """
    Case data that algorithms can access
    """

    def __init__(self, data_loader: DataLoader, sla: Set[Node], detect_time: float):
        self._data_loader = data_loader
        self._sla = sla
        self._detect_time = detect_time

    @property
    def data_loader(self) -> DataLoader:
        """
        Interface to access raw data
        """
        return self._data_loader

    @property
    def sla(self) -> Set[Node]:
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
