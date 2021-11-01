"""
Test suites for Graph
"""
from srca.model.graph import MemoryGraph
from srca.model.graph import Node


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


def test_memory_graph():
    """
    Test case for MemoryGraph
    """
    latency = Node("DB", "Latency")
    traffic = Node("DB", "Traffic")
    saturation = Node("DB", "Saturation")
    parents = {
        latency: [traffic, saturation],
        saturation: {traffic},
    }
    graph = MemoryGraph(parents)
    assert not graph.parents(traffic)
    assert set(graph.parents(latency)) == {traffic, saturation}
