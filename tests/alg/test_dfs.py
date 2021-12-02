"""
Test suites for DFSRanker
"""
import networkx as nx

from srca.alg.base import Score
from srca.alg.dfs import DFSRanker
from srca.model.case import CaseData
from srca.model.graph import MemoryGraph
from srca.model.graph import Graph
from srca.model.graph import Node


def test_dfs_ranker(graph: Graph, case_data: CaseData):
    """
    DFSRanker shall filter in anomalous nodes with no anomalous parents
    """
    latency = Node("DB", "Latency")
    traffic = Node("DB", "Traffic")
    saturation = Node("DB", "Saturation")
    scores = {
        latency: Score(2),
        traffic: Score(1),
        saturation: Score(3),
    }
    params = dict(data=case_data, scores=scores, current=case_data.detect_time + 60)

    ranker = DFSRanker(anomaly_threshold=0)
    # With an empty graph
    empty_graph = nx.DiGraph()
    empty_graph.add_nodes_from([latency, traffic, saturation])
    ranks = ranker.rank(graph=MemoryGraph(empty_graph), **params)
    assert {node for node, _ in ranks} == {latency}
    # Search all nodes
    ranks = ranker.rank(graph=graph, **params)
    assert {node for node, _ in ranks} == {traffic}
    # Filter out nodes
    ranker = DFSRanker(anomaly_threshold=2)
    ranks = ranker.rank(graph=graph, **params)
    assert {node for node, _ in ranks} == {saturation}
