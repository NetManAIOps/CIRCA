# pylint: disable=redefined-outer-name
"""
Fixtures for test cases
"""
import tempfile
from typing import Dict
from typing import Sequence
from typing import Tuple

import networkx as nx

import pytest

from srca.alg.base import GraphFactory
from srca.alg.common import StaticGraphFactory
from srca.model.case import CaseData
from srca.model.data_loader import DataLoader
from srca.model.data_loader import MemoryDataLoader
from srca.model.graph import Graph
from srca.model.graph import MemoryGraph
from srca.model.graph import Node


@pytest.fixture
def tempdir() -> str:
    """
    Create temporary directory
    """
    with tempfile.TemporaryDirectory() as folder:
        yield folder


@pytest.fixture
def case_data(data_loader: DataLoader) -> CaseData:
    """
    Create a CaseData for test
    """
    return CaseData(
        data_loader=data_loader,
        sla=Node("DB", "Latency"),
        detect_time=240,
        lookup_window=4,
        detect_window=2,
    )


@pytest.fixture
def data_loader(mock_data: Dict[Node, Sequence[float]]) -> DataLoader:
    """
    Create a MemoryDataLoader for test
    """
    data: Dict[str, Dict[str, Sequence[Tuple[float, float]]]] = {}
    for node, values in mock_data.items():
        if node.entity not in data:
            data[node.entity] = {}
        data[node.entity][node.metric] = [
            (index * 60, value) for index, value in enumerate(values)
        ]
    return MemoryDataLoader(data)


@pytest.fixture
def graph_factory(graph: Graph) -> GraphFactory:
    """
    Create a GraphFactory for test
    """
    return StaticGraphFactory(graph)


@pytest.fixture
def graph() -> Graph:
    """
    Create a Graph for test
    """
    latency = Node("DB", "Latency")
    traffic = Node("DB", "Traffic")
    saturation = Node("DB", "Saturation")
    return MemoryGraph(
        nx.DiGraph(
            {
                traffic: [latency, saturation],
                saturation: [latency],
            }
        )
    )


@pytest.fixture
def mock_data() -> Dict[Node, Sequence[float]]:
    """
    Mock data for test
    """
    return {
        Node("DB", "Latency"): (10, 12, 11, 9, 100, 90),
        Node("DB", "Traffic"): (100, 110, 90, 105, 200, 150),
        Node("DB", "Saturation"): (5, 4, 5, 6, 90, 85),
    }