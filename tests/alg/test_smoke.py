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
from srca.alg.invariant_network.enmf import InvariantNetwork
from srca.alg.dfs import DFSScorer
from srca.alg.dfs import MicroHECLScorer
from srca.alg.evt import SPOTScorer
from srca.alg.random_walk import RandomWalkScorer
from srca.alg.random_walk import SecondOrderRandomWalkScorer
from srca.alg.structural import StructuralRanker
from srca.alg.structural import StructuralScorer
from srca.alg.structural.gmm import GMMRegressor
from srca.alg.structural.gmm.mdn import MDNPredictor
from srca.alg.structural.gmm.prob_rf import ProbRF
from srca.alg.structural.linear import LinearRegressor
from srca.model.case import CaseData


_in_params = dict(epoches=10, invariant_network=InvariantNetwork(n=1, m=1))


@pytest.mark.parametrize(
    ("scorers",),
    [
        ((NSigmaScorer(),),),
        ((NSigmaScorer(), MicroHECLScorer(anomaly_threshold=3, stop_threshold=0.7)),),
        ((NSigmaScorer(), DFSScorer(anomaly_threshold=3)),),
        ((SPOTScorer(proba=0.1),),),
        ((CRDScorer(model_params=_in_params),),),
        ((ENMFScorer(model_params=_in_params, use_softmax=False),),),
        ((ENMFScorer(model_params=_in_params, use_softmax=True),),),
        ((PartialCorrelationScorer(), RandomWalkScorer()),),
        ((CorrelationScorer(), SecondOrderRandomWalkScorer()),),
        (
            (
                StructuralScorer(regressor=LinearRegressor(use_discrete=True)),
                StructuralRanker(threshold=3),
            ),
        ),
        ((StructuralScorer(regressor=GMMRegressor(regressor=ProbRF())),),),
        ((StructuralScorer(regressor=GMMRegressor(regressor=MDNPredictor())),),),
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
