"""
Test suites for graph factories
"""
import os

import numpy as np

import pytest

from circa.graph import GraphFactory
from circa.graph.pcts import PCTSFactory
from circa.graph.r import PCAlgFactory
from circa.graph.structural import Component
from circa.graph.structural import StructuralGraph
from circa.model.case import CaseData
from circa.model.data_loader import MemoryDataLoader
from circa.model.graph import Graph
from circa.model.graph import Node


@pytest.mark.parametrize(
    "factory",
    [PCAlgFactory(method="PC-gauss"), PCAlgFactory(method="PC-gsq"), PCTSFactory()],
)
def test_smoke(factory: GraphFactory):
    """
    GraphFactory shall not throw exceptions
    """
    size = 120
    timestamps = np.array(range(size + 1)) * 60
    data_loader = MemoryDataLoader(
        data={
            "DB": {
                str(metric): list(zip(timestamps, np.random.rand(size + 11)))
                for metric in range(10)
            }
        }
    )
    case_data = CaseData(
        data_loader=data_loader,
        sli=Node(entity="DB", metric="0"),
        detect_time=timestamps[-2],
        lookup_window=size,
    )
    graph = factory.create(data=case_data, current=case_data.detect_time + 60)
    assert isinstance(graph, Graph)


class TestStructuralGraph:
    """
    Test cases for structural graph
    """

    _GRAPH_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "sgraph")
    _GRAPH = {
        "query_per_second": {"response_time", "error_rate", "transaction_per_second"},
        "response_time": {"error_rate"},
        "error_rate": set(),
        "transaction_per_second": {"db_time", "table_space", "error_rate"},
        "db_time": {"response_time", "response_time", "error_rate"},
        "table_space": {"db_time", "error_rate"},
    }
    _METRICS = set(_GRAPH.keys())

    def test_component(self):
        """
        Component shall load metrics and sub-components
        """
        component = Component(self._GRAPH_DIR, "mock")
        assert len(component.parallel) == 1
        sub_component = component.parallel[0]
        assert sub_component.name == "DB"
        assert set(component.list_metrics()) == self._METRICS

    def test_structural_graph(self):
        """
        StructuralGraph shall create graph among metrics
        """
        sgraph = StructuralGraph(self._GRAPH_DIR, "mock")
        graph = sgraph.visit()
        assert set(graph.nodes) == self._METRICS
        for metric, children in self._GRAPH.items():
            assert set(graph.successors(metric)) == children
