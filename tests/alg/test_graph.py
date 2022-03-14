"""
Test suites for graph factories
"""
import numpy as np

import pytest

from circa.alg.base import GraphFactory
from circa.alg.graph.pcts import PCTSFactory
from circa.alg.graph.r import PCAlgFactory
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
