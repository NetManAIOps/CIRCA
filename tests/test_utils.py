"""
Test suites for utilities
"""
import time

import networkx as nx

import pytest

from circa.utils import _HAS_SIGALRM
from circa.utils import Timeout
from circa.utils import topological_sort


@pytest.mark.skipif(not _HAS_SIGALRM, reason="signal.SIGALRM is unavailable")
def test_timeout():
    """
    Timeout shall terminate the inside task after the given time
    """
    origin_value = 0
    new_value = origin_value + 1
    value = origin_value

    with Timeout(seconds=1):
        with pytest.raises(TimeoutError):
            time.sleep(2)
            value = new_value
    assert value == origin_value

    with Timeout(seconds=1):
        time.sleep(0.5)
        value = new_value
    assert value == new_value


def test_topological_sort():
    """
    topological_sort shall handle loops
    """
    graph = nx.DiGraph()
    graph.add_edges_from(
        [
            (1, 2),
            (1, 3),
            (2, 1),
            (2, 3),
            (3, 1),
            (3, 2),
            (3, 4),
        ]
    )
    assert topological_sort(
        nodes=set(graph.nodes),
        predecessors=graph.predecessors,
        successors=graph.successors,
    ) == [{1, 2, 3}, {4}]
