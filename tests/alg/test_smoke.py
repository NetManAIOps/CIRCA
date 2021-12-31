"""
Smoke tests for algorithms
"""
from typing import Sequence

import pytest

from srca.alg.base import GraphFactory
from srca.alg.base import Scorer
from srca.alg.common import Model
from srca.alg.common import NSigmaScorer
from srca.alg.correlation import CorrelationScorer
from srca.alg.correlation import PartialCorrelationScorer
from srca.alg.invariant_network import CRDScorer
from srca.alg.invariant_network import ENMFScorer
from srca.alg.invariant_network.crd import CRD
from srca.alg.invariant_network.enmf import ENMF
from srca.alg.invariant_network.enmf import InvariantNetwork
from srca.alg.dfs import DFSScorer
from srca.alg.dfs import MicroHECLScorer
from srca.alg.evt import SPOTScorer
from srca.alg.random_walk import RandomWalkScorer
from srca.alg.random_walk import SecondOrderRandomWalkScorer
from srca.model.case import CaseData


@pytest.mark.parametrize(
    ("scorers",),
    [
        ((NSigmaScorer(),),),
        ((NSigmaScorer(), MicroHECLScorer(anomaly_threshold=3, stop_threshold=0.7)),),
        ((NSigmaScorer(), DFSScorer(anomaly_threshold=3)),),
        ((SPOTScorer(proba=0.1),),),
        (
            (
                CRDScorer(
                    model=CRD(
                        epoches=10, invariant_network=InvariantNetwork(n=1, m=1)
                    )
                ),
            ),
        ),
        ((ENMFScorer(model=ENMF(invariant_network=InvariantNetwork(n=1, m=1))),),),
        ((PartialCorrelationScorer(), RandomWalkScorer()),),
        ((CorrelationScorer(), SecondOrderRandomWalkScorer()),),
    ],
)
def test_smoke(
    graph_factory: GraphFactory, scorers: Sequence[Scorer], case_data: CaseData
):
    """
    Smoke tests
    """
    model = Model(graph_factory=graph_factory, scorers=scorers)
    assert model.analyze(data=case_data, current=case_data.detect_time + 60)
