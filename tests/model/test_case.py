"""
Test suites for case models
"""
import networkx as nx

from circa.model.case import CaseData
from circa.model.data_loader import DataLoader
from circa.model.graph import MemoryGraph
from circa.model.graph import Node


def test_load_data(data_loader: DataLoader, case_data_params: dict):
    """
    CaseData.load_data shall load data without constant
        or filling missing metrics on demand.
    """
    current = case_data_params["detect_time"] + 60

    case_data = CaseData(data_loader=data_loader, **case_data_params, prune=True)
    data_origin = case_data.load_data(current=current)
    nodes_origin = set(data_origin.keys())

    graph = nx.DiGraph()
    graph.add_nodes_from(nodes_origin)
    node = Node("DB", "Errors")
    assert node not in nodes_origin
    graph.add_node(node)

    # With existing metrics only
    data = case_data.load_data(graph=MemoryGraph(graph), current=current)
    assert set(data.keys()) == nodes_origin

    # Filling missing metrics
    case_data = CaseData(data_loader=data_loader, **case_data_params, prune=False)
    data = case_data.load_data(graph=MemoryGraph(graph), current=current)
    assert set(data.keys()) == (nodes_origin | {node})
