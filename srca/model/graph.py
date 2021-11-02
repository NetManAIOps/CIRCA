"""
Define the interface for algorithms to access relations
"""
from abc import ABC
from typing import Dict
from typing import Set


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

    def __init__(self):
        self._node: Set[Node] = set()

    @property
    def nodes(self) -> Set[Node]:
        """
        Get the set of nodes in the graph
        """
        return self._node

    def parents(self, node: Node, **kwargs) -> Set[Node]:
        """
        Get the parents of the given node in the graph
        """
        raise NotImplementedError


class MemoryGraph(Graph):
    """
    Implement Graph with data in memory
    """

    def __init__(self, parents: Dict[Node, Set[Node]]):
        """
        parents: parents[Node(entity, metric)] is a set of nodes
            which are the parents for the given metric of the given entity
        """
        super().__init__()
        for child, parent in parents.items():
            parents[child] = set(parent)
            self._node.add(child)
            self._node.update(parent)
        self._parents = parents

    def parents(self, node: Node, **kwargs) -> Set[Node]:
        return self._parents.get(node, set())
