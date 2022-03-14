"""
Random walk
"""
from collections import defaultdict
from typing import Callable
from typing import Dict
from typing import List
from typing import Union

import numpy as np
import pandas as pd

from .base import Score
from .base import Scorer
from ..model.case import CaseData
from ..model.graph import Graph
from ..model.graph import Node


def _times(num: int, multiplier: int = 10) -> int:
    return num * multiplier


class RandomWalkScorer(Scorer):
    """
    Scorer based on random walk
    """

    def __init__(
        self,
        rho: float = 0.5,
        remove_sli: bool = False,
        num_loop: Union[int, Callable[[int], int]] = None,
        **kwargs,
    ):
        # pylint: disable=too-many-arguments
        super().__init__(**kwargs)
        self._rho = rho
        self._remove_sli = remove_sli
        self._num_loop = num_loop if num_loop is not None else _times
        self._rng = np.random.default_rng(self._seed)

    def generate_transition_matrix(
        self, graph: Graph, data: CaseData, scores: Dict[Node, Score]
    ) -> pd.DataFrame:
        """
        Generate the transition matrix
        """
        nodes = list(scores.keys())
        size = len(nodes)
        matrix = pd.DataFrame(np.zeros([size, size]), index=nodes, columns=nodes)
        for node in scores:
            for child in graph.children(node):
                if child in scores:
                    matrix[node][child] = self._rho * abs(scores[child].score)

            parents = graph.parents(node)
            if self._remove_sli:
                parents -= {data.sli}
            for parent in parents:
                if parent in scores:
                    matrix[node][parent] = abs(scores[parent].score)

            matrix[node][node] = max(abs(scores[node].score) - matrix[node].max(), 0)

            total_weight = matrix[node].sum()
            if total_weight > 0:
                matrix[node] /= total_weight
            else:
                matrix[node] = 1 / size
        return matrix

    def _walk(
        self, start: Node, num_loop: int, matrix: pd.DataFrame
    ) -> Dict[Node, int]:
        node: Node = start
        counter = defaultdict(int)
        for _ in range(num_loop):
            node = self._rng.choice(matrix.index, p=matrix[node])
            counter[node] += 1
        return counter

    def score(
        self,
        graph: Graph,
        data: CaseData,
        current: float,
        scores: Dict[Node, Score] = None,
    ) -> Dict[Node, Score]:
        if not scores:
            return scores

        matrix = self.generate_transition_matrix(graph=graph, data=data, scores=scores)
        if isinstance(self._num_loop, int):
            num_loop = self._num_loop
        else:
            num_loop = self._num_loop(len(scores))
        counter = self._walk(start=data.sli, num_loop=num_loop, matrix=matrix)

        for node, score in scores.items():
            score["pagerank"] = score.score = counter[node] / num_loop
        return scores


class SecondOrderRandomWalkScorer(RandomWalkScorer):
    """
    Scorer based on second-order random walk
    """

    def __init__(self, beta: float = 0.5, **kwargs):
        super().__init__(**kwargs)
        self._beta = beta

    def _walk(
        self, start: Node, num_loop: int, matrix: pd.DataFrame
    ) -> Dict[Node, int]:
        node: Node = start
        node_pre: Node = start
        counter = defaultdict(int)
        for _ in range(num_loop):
            prob_pre = matrix[node_pre][node]
            node_pre = node

            candidates: List[Node] = []
            weights: List[float] = []
            for key, value in matrix[node].iteritems():
                if value > 0:
                    candidates.append(key)
                    weights.append((1 - self._beta) * prob_pre + self._beta * value)
            total_weight = sum(weights)
            if total_weight == 0:
                node = self._rng.choice(candidates)
            else:
                node = self._rng.choice(
                    candidates, p=[weight / total_weight for weight in weights]
                )
            counter[node] += 1
        return counter
