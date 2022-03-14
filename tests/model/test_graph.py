"""
Test suites for Graph
"""
import networkx as nx

from circa.model.graph import MemoryGraph
from circa.model.graph import Node


class TestNode:
    """
    Test cases for Node
    """

    @staticmethod
    def test_compare():
        """
        One node can be identified by both entity and metric
        """
        first = Node(entity="entity1", metric="metric1")
        assert first == Node(entity="entity1", metric="metric1")
        assert first != Node(entity="entity2", metric="metric1")
        assert first != Node(entity="entity1", metric="metric2")
        assert first != dict(entity="entity1", metric="metric1")
        assert first != ("entity1", "metric1")

    @staticmethod
    def test_hash():
        """
        Node can be used as the element of a set or the key of a dict
        """
        first = Node(entity="entity1", metric="metric1")
        second = ("entity1", "metric1")
        assert len({first, Node(entity="entity1", metric="metric1")}) == 1
        assert len({first, Node(entity="entity1", metric="metric2")}) == 2
        assert len({first, second}) == 2
        assert len({first: 1, Node(entity="entity1", metric="metric1"): 2}) == 1
        assert len({first: 1, Node(entity="entity1", metric="metric2"): 2}) == 2
        assert len({first: 1, second: 2}) == 2

    @staticmethod
    def test_asdict():
        """
        Node.asdict shall provide parameters to create a new Node
        """
        node = Node(entity="entity1", metric="metric1")
        assert Node(**node.asdict()) == node


def test_memory_graph():
    """
    Test case for MemoryGraph
    """
    latency = Node("DB", "Latency")
    traffic = Node("DB", "Traffic")
    saturation = Node("DB", "Saturation")
    graph = MemoryGraph(
        nx.DiGraph(
            {
                traffic: [latency, saturation],
                saturation: [latency],
            }
        )
    )
    assert graph.nodes == {latency, traffic, saturation}
    assert not graph.parents(traffic)
    assert graph.parents(latency) == {traffic, saturation}
    assert not graph.children(latency)
    assert graph.children(traffic) == {latency, saturation}
    assert graph.topological_sort == [{traffic}, {saturation}, {latency}]
