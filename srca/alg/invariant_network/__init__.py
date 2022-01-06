"""
Ranking based on the broken edges in the invariant network model
"""
from typing import Dict

import numpy as np

from .crd import CRD
from .enmf import ENMF
from .enmf import ENMFSoft
from ..base import Score
from ..base import Scorer
from ...model.case import CaseData
from ...model.graph import Graph
from ...model.graph import Node


class ENMFScorer(Scorer):
    """
    Score based on the ENMF model
    """

    def __init__(
        self,
        model_params: dict = None,
        use_softmax: bool = False,
        discrete: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        if model_params is None:
            model_params = {}
        if use_softmax:
            self._model = ENMFSoft(**model_params)
        else:
            self._model = ENMF(**model_params)
        self._discrete = discrete

    def score(
        self,
        graph: Graph,
        data: CaseData,
        current: float,
        scores: Dict[Node, Score] = None,
    ) -> Dict[Node, Score]:
        series = data.load_data(graph, current)
        nodes = list(series.keys())
        time_series = np.array([series[node] for node in nodes]).T
        self._model.train(time_series[: data.train_window, :])
        broken_scores = self._model.rank(
            data=time_series[-data.test_window - self._model.start :, :],
            discrete=self._discrete,
        )
        results = {
            node: Score(score=broken_score)
            for broken_score, node in zip(broken_scores, nodes)
        }
        if scores is None:
            return results
        return {node: score.update(results[node]) for node, score in scores.items()}


class CRDScorer(ENMFScorer):
    """
    Score based on the CRD model
    """

    def __init__(self, model_params: dict = None, discrete: bool = True, **kwargs):
        super().__init__(discrete=discrete, **kwargs)
        if model_params is None:
            model_params = {}
        model_params = dict(cuda=self._cuda, seed=self._seed, **model_params)
        self._model = CRD(**model_params)
