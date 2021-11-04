"""
Define the interface for algorithms to access relations
"""
from abc import ABC
from typing import Dict
from typing import List
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
        if self._sorted_nodes:
            return self._sorted_nodes

        degrees = {node: len(self.parents(node)) for node in self.nodes}

        nodes: List[Set[Node]] = []
        while degrees:
            minimum = min(degrees.values())
            node_set = {node for node, degree in degrees.items() if degree == minimum}
            nodes.append(node_set)
            for node in node_set:
                degrees.pop(node)
                for child in self.children(node):
                    if child in degrees:
                        degrees[child] -= 1

        self._sorted_nodes = nodes
        return nodes

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

    def __init__(self, graph: Dict[Node, Set[Node]]):
        """
        parents: parents[Node(entity, metric)] is a set of nodes
            which are the parents for the given metric of the given entity
        """
        super().__init__()
        children: Dict[Node, Set[Node]] = {}
        for child, parents in graph.items():
            graph[child] = set(parents)
            self._nodes.add(child)
            self._nodes.update(parents)

            for parent in parents:
                if parent not in children:
                    children[parent] = set()
                children[parent].add(child)

        self._children = children
        self._parents = graph

    def children(self, node: Node, **kwargs) -> Set[Node]:
        return self._children.get(node, set())

    def parents(self, node: Node, **kwargs) -> Set[Node]:
        return self._parents.get(node, set())
