"""
Abstract interfaces for graph generation
"""
from abc import ABC
from typing import Union

from ..model.case import CaseData
from ..model.graph import Graph
from ..model.graph import MemoryGraph


class GraphFactory(ABC):
    """
    The abstract interface to create Graph
    """

    def __init__(self, seed: int = 0):
        self._seed = seed

    @staticmethod
    def load(filename: str) -> Union[Graph, None]:
        """
        Load a graph from the given file

        Returns:
        - A graph, if available
        - None, if dump/load is not supported
        - Raise LoadingInvalidGraphException if the file cannot be parsed
        """
        return MemoryGraph.load(filename)

    def create(self, data: CaseData, current: float) -> Graph:
        """
        Create the Graph
        """
        raise NotImplementedError
