"""
Test suites for graph factories
"""
import os

import numpy as np

import pytest

from circa.graph import GraphFactory
from circa.graph.pcts import PCTSFactory
from circa.graph.r import PCAlgFactory
from circa.graph.structural import Config
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

    _GRAPH_FILENAME = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "sgraph", "index.yml"
    )
    _GRAPH = {
        Node("WEB", "query_per_second"): {
            Node("WEB", "response_time"),
            Node("WEB", "error_rate"),
            Node("DB", "transaction_per_second"),
        },
        Node("WEB", "response_time"): {Node("WEB", "error_rate")},
        Node("WEB", "error_rate"): set(),
        Node("DB", "transaction_per_second"): {
            Node("DB", "db_time"),
            Node("DB", "table_space"),
            Node("WEB", "error_rate"),
        },
        Node("DB", "db_time"): {
            Node("WEB", "response_time"),
            Node("WEB", "error_rate"),
        },
        Node("DB", "table_space"): {Node("DB", "db_time"), Node("WEB", "error_rate")},
    }
    _METRICS = set(_GRAPH.keys())

    def test_config(self):
        """
        Config shall load components
        """
        config = Config(self._GRAPH_FILENAME)
        assert len(config.components) == 2

    def test_structural_graph(self):
        """
        StructuralGraph shall create graph among metrics
        """
        sgraph = StructuralGraph(filename=self._GRAPH_FILENAME)
        graph = sgraph.visit()
        assert set(graph.nodes) == self._METRICS
        for metric, children in self._GRAPH.items():
            assert set(graph.successors(metric)) == children
