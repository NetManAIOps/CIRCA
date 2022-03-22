"""
Test suites for Regression-based Hypothesis Testing
"""
from typing import Dict
from typing import Sequence
from unittest.mock import patch

import numpy as np

from circa.alg.ci import RHTScorer
from circa.model.case import CaseData
from circa.model.graph import Graph
from circa.model.graph import Node


class TestRHTScorer:
    """
    Test cases for RHTScorer
    """

    _node = Node("DB", "Latency")
    # Use ordered list for this test suite
    _parents = [Node("DB", "Traffic"), Node("DB", "Saturation"), Node("DB", "error")]

    def test_split_data(
        self, mock_data: Dict[Node, Sequence[float]], case_data: CaseData
    ):
        """
        Test RHTScorer.split_data
        """
        scorer = RHTScorer(tau_max=0)
        train_x, test_x, train_y, test_y = scorer.split_data(
            mock_data, self._node, self._parents, case_data
        )
        assert np.allclose(train_x, np.array([[100, 110, 90], [5, 4, 5]]).T)
        assert np.allclose(train_y, np.array([10, 12, 11]))
        assert np.allclose(test_x, np.array([[200, 150], [90, 85]]).T)
        assert np.allclose(test_y, np.array([100, 90]))

        scorer = RHTScorer(tau_max=1)
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
        Test RHTScorer.score_node
        """
        scorer = RHTScorer()
        score_parents = scorer.score_node(graph, mock_data, self._node, case_data)
        with patch.object(graph, "parents") as mock_parents:
            mock_parents.return_value = []
            score_alone = scorer.score_node(graph, mock_data, self._node, case_data)
            mock_parents.assert_called_once_with(self._node)
        assert score_parents != score_alone
