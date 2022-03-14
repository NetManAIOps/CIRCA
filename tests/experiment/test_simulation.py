"""
Test suites for the simulation experiment
"""
import networkx as nx
import numpy as np

import pytest

from circa.experiment.simulation import generate
from circa.experiment.simulation import generate_sedag


@pytest.mark.parametrize(
    ["num_node", "num_edge"],
    [
        (10, 11),
        (10, 30),
    ],
)
def test_generate_sedag(num_node: int, num_edge: int, rounds: int = 10):
    """
    generate_sedag shall generate DAG with only one end
    """
    pre_matrix = np.zeros((num_node, num_node))
    for _ in range(rounds):
        matrix = generate_sedag(num_node=num_node, num_edge=num_edge)
        assert (matrix != pre_matrix).sum() > 0, "Generate the same matrix"
        assert matrix.astype(bool).sum() == num_edge
        assert nx.is_directed_acyclic_graph(nx.DiGraph(zip(*np.where(matrix))))
        no_results: np.ndarray = matrix.astype(bool).sum(axis=0) == 0
        assert no_results[0], "The first node is not an end"
        assert no_results.sum() == 1, "More than one nodes do not have results"
        power = np.eye(num_node)
        for _ in range(num_node):
            power = power @ matrix
        assert (power == 0).all()
        pre_matrix = matrix


def test_generate():
    """
    generate shall create a dataset with a single-end DAG and several cases
    """
    num_node, num_edge, num_cases = 10, 30, 2
    dataset = generate(num_node=num_node, num_edge=num_edge, num_cases=num_cases)
    assert len(dataset.cases) == num_cases
    case = dataset.cases[0]
    graph = dataset.graph_factory.create(data=case.data, current=case.data.detect_time)
    assert len(graph.children(case.data.sli)) == 0
