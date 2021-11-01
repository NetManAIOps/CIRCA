"""
Define the interface for algorithms to access relations
"""
from abc import ABC
from typing import Dict
from typing import Sequence


class Node:
    """
    The element of a graph
    """

    def __init__(self, entity: str, metric: str):
        self._entity = entity
        self._metric = metric

    @property
    def entity(self) -> str:
        """
        Entity getter
        """
        return self._entity

    @property
    def metric(self) -> str:
        """
        Metric getter
        """
        return self._metric

    def __eq__(self, obj: object) -> bool:
        if isinstance(obj, Node):
            return self.entity == obj.entity and self.metric == obj.metric
        return False

    def __hash__(self) -> int:
        return hash((self.entity, self.metric))

    def __repr__(self) -> str:
        return f"Node{(self.entity, self.metric)}"


class Graph(ABC):
    """
    The abstract interface to access relations
    """

    def parents(self, node: Node, **kwargs) -> Sequence[Node]:
        """
        Get the parents of the given node in the graph
        """
        raise NotImplementedError


class MemoryGraph(Graph):
    """
    Implement Graph with data in memory
    """

    def __init__(self, parents: Dict[Node, Sequence[Node]]):
        """
        parents: parents[Node(entity, metric)] is a sequence of nodes
            which are the parents for the given metric of the given entity
        """
        self._parents = parents

    def parents(self, node: Node, **kwargs) -> Sequence[Node]:
        return self._parents.get(node, [])
