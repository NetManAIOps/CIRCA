"""
Define the interface for algorithms to access relations
"""
from abc import ABC
from typing import Dict
from typing import List
from typing import Set

import networkx as nx


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

    def asdict(self) -> Dict[str, float]:
        """
        Serialized as a dict
        """
        return {"entity": self._entity, "metric": self._metric}

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
        self._nodes: Set[Node] = set()
        self._sorted_nodes: List[Set[Node]] = None

    @property
    def nodes(self) -> Set[Node]:
        """
        Get the set of nodes in the graph
        """
        return self._nodes

    @property
    def topological_sort(self) -> List[Set[Node]]:
        """
        Sort nodes with parents first

        The graph specifies the parents of each node.
        """
        raise NotImplementedError

    def children(self, node: Node, **kwargs) -> Set[Node]:
        """
        Get the children of the given node in the graph
        """
        raise NotImplementedError

    def parents(self, node: Node, **kwargs) -> Set[Node]:
        """
        Get the parents of the given node in the graph
        """
        raise NotImplementedError


class MemoryGraph(Graph):
    """
    Implement Graph with data in memory
    """

    def __init__(self, graph: nx.DiGraph):
        """
        graph: The whole graph
        """
        super().__init__()
        self._graph = graph
        self._nodes.update(self._graph.nodes)

    @property
    def topological_sort(self) -> List[Set[Node]]:
        if self._sorted_nodes is None:
            self._sorted_nodes = [
                set(nodes) for nodes in nx.topological_generations(self._graph)
            ]
        return self._sorted_nodes

    def children(self, node: Node, **kwargs) -> Set[Node]:
        if not self._graph.has_node(node):
            return set()
        return set(self._graph.successors(node))

    def parents(self, node: Node, **kwargs) -> Set[Node]:
        if not self._graph.has_node(node):
            return set()
        return set(self._graph.predecessors(node))
