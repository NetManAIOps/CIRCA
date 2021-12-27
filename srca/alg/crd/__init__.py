"""
Cluster Ranking based fault Diagnosis
"""
from typing import Dict

import numpy as np

from .crd import CRD
from ..base import Score
from ..base import Scorer
from ...model.case import CaseData
from ...model.graph import Graph
from ...model.graph import Node


class CRDScorer(Scorer):
    """
    Score based on the CRD model
    """

    def __init__(
        self,
        crd_model: CRD = None,
        discrete: bool = True,
        use_sgd: bool = False,
        **kwargs
    ):
        super().__init__(**kwargs)
        self._model = CRD(seed=self._seed) if crd_model is None else crd_model
        self._discrete = discrete
        self._use_sgd = use_sgd

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
        crd_scores = self._model.rank(
            data=time_series[-data.test_window - self._model.start :, :],
            discrete=self._discrete,
            use_sgd=self._use_sgd,
        )
        results = {
            node: Score(score=crd_score) for crd_score, node in zip(crd_scores, nodes)
        }
        if scores is None:
            return results
        return {node: score.update(results[node]) for node, score in scores.items()}
