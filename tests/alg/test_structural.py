"""
Test suites for Structural Root Cause Analysis
"""
import os
from typing import Dict
from typing import Sequence
from unittest.mock import patch

import numpy as np

from srca.alg.structural import StructuralScorer
from srca.alg.structural.graph import Component
from srca.alg.structural.graph import StructuralGraph
from srca.model.case import CaseData
from srca.model.graph import Graph
from srca.model.graph import Node


class TestStructuralScorer:
    """
    Test cases for StructuralScorer
    """

    _node = Node("DB", "Latency")
    # Use ordered list for this test suite
    _parents = [Node("DB", "Traffic"), Node("DB", "Saturation"), Node("DB", "error")]

    def test_split_data(
        self, mock_data: Dict[Node, Sequence[float]], case_data: CaseData
    ):
        """
        Test StructuralScorer.split_data
        """
        scorer = StructuralScorer(tau_max=0)
        train_x, test_x, train_y, test_y = scorer.split_data(
            mock_data, self._node, self._parents, case_data
        )
        assert np.allclose(train_x, np.array([[100, 110, 90], [5, 4, 5]]).T)
        assert np.allclose(train_y, np.array([10, 12, 11]))
        assert np.allclose(test_x, np.array([[200, 150], [90, 85]]).T)
        assert np.allclose(test_y, np.array([100, 90]))

        scorer = StructuralScorer(tau_max=1)
        train_x, test_x, train_y, test_y = scorer.split_data(
            mock_data, self._node, self._parents, case_data
        )
        assert np.allclose(train_x, np.array([[110, 90], [4, 5], [100, 110], [5, 4]]).T)
        assert np.allclose(train_y, np.array([12, 11]))
        assert np.allclose(
            test_x, np.array([[200, 150], [90, 85], [105, 200], [6, 90]]).T
        )
        assert np.allclose(test_y, np.array([100, 90]))

        train_x, test_x, train_y, test_y = scorer.split_data(
            mock_data, self._node, [], case_data
        )
        assert len(train_x) == 0
        assert np.allclose(train_y, np.array([12, 11]))
        assert len(test_x) == 0
        assert np.allclose(test_y, np.array([100, 90]))

    def test_score_node(
        self, graph: Graph, case_data: CaseData, mock_data: Dict[Node, Sequence[float]]
    ):
        """
        Test StructuralScorer.score_node
        """
        scorer = StructuralScorer()
        score_parents = scorer.score_node(graph, mock_data, self._node, case_data)
        with patch.object(graph, "parents") as mock_parents:
            mock_parents.return_value = []
            score_alone = scorer.score_node(graph, mock_data, self._node, case_data)
            mock_parents.assert_called_once_with(self._node)
        assert score_parents != score_alone


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
