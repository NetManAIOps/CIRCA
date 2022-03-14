"""
Smoke tests for algorithms
"""
from typing import Sequence

import pytest

from circa.alg.base import Scorer
from circa.alg.common import Model
from circa.alg.common import NSigmaScorer
from circa.alg.correlation import CorrelationScorer
from circa.alg.correlation import PartialCorrelationScorer
from circa.alg.invariant_network import CRDScorer
from circa.alg.invariant_network import ENMFScorer
from circa.alg.invariant_network.enmf import InvariantNetwork
from circa.alg.dfs import DFSScorer
from circa.alg.dfs import MicroHECLScorer
from circa.alg.evt import SPOTScorer
from circa.alg.random_walk import RandomWalkScorer
from circa.alg.random_walk import SecondOrderRandomWalkScorer
from circa.alg.ci import DAScorer
from circa.alg.ci import RHTScorer
from circa.alg.ci.anm import ANMRegressor
from circa.alg.ci.gmm import GMMRegressor
from circa.alg.ci.gmm.mdn import MDNPredictor
from circa.alg.ci.gmm.prob_rf import ProbRF
from circa.graph import GraphFactory
from circa.model.case import CaseData


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
                RHTScorer(regressor=ANMRegressor()),
                DAScorer(threshold=3),
            ),
        ),
        ((RHTScorer(regressor=GMMRegressor(regressor=ProbRF())),),),
        ((RHTScorer(regressor=GMMRegressor(regressor=MDNPredictor())),),),
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
