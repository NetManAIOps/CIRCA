"""
Common utilities
"""
import json
import logging
import os
from typing import Dict
from typing import List
from typing import Sequence
from typing import Set
from typing import Tuple

import networkx as nx
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler

from .base import GraphFactory
from .base import Score
from .base import Scorer
from ..model.case import Case
from ..model.case import CaseData
from ..model.graph import Graph
from ..model.graph import MemoryGraph
from ..model.graph import Node


def pearson(series_a: np.ndarray, series_b: np.ndarray) -> float:
    """
    Pearson coefficient, checking constant
    """
    std_a: float = series_a.std()
    std_b: float = series_b.std()
    if std_a == 0 or std_b == 0:
        return 0
    prod: np.ndarray = (series_a - series_a.mean()) * (series_b - series_b.mean())
    return prod.sum() / (std_a * std_b * len(series_a))


def zscore(train_y: np.ndarray, test_y: np.ndarray) -> np.ndarray:
    """
    Estimate to what extend each value in test_y violates
    the normal distribution defined by train_y
    """
    scaler = StandardScaler().fit(train_y.reshape(-1, 1))
    return scaler.transform(test_y.reshape(-1, 1))[:, 0]


def zscore_conf(score: float) -> float:
    """
    Convert z-score into confidence about the hypothesis the score is abnormal
    """
    return 1 - 2 * norm.cdf(-abs(score))


class EmptyGraphFactory(GraphFactory):
    """
    Create a graph with nodes only
    """

    def create(self, data: CaseData, current: float) -> Graph:
        graph = nx.DiGraph()
        graph.add_nodes_from(data.data_loader.nodes)
        return MemoryGraph(graph)


class StaticGraphFactory(GraphFactory):
    """
    Create the same graph
    """

    def __init__(self, graph: Graph):
        self._graph = graph

    def create(self, data: CaseData, current: float) -> Graph:
        return self._graph


class DecomposableScorer(Scorer):
    """
    Score each node separately
    """

    def score_node(
        self,
        graph: Graph,
        series: Dict[Node, Sequence[float]],
        node: Node,
        data: CaseData,
    ) -> Score:
        """
        Estimate how suspicious a node is
        """
        raise NotImplementedError

    def score(
        self,
        graph: Graph,
        data: CaseData,
        current: float,
        scores: Dict[Node, Score] = None,
    ) -> Dict[Node, Score]:
        series = data.load_data(graph, current)
        return {node: self.score_node(graph, series, node, data) for node in series}


class NSigmaScorer(DecomposableScorer):
    """
    Score nodes by n-sigma
    """

    def score_node(
        self,
        graph: Graph,
        series: Dict[Node, Sequence[float]],
        node: Node,
        data: CaseData,
    ) -> Score:
        series_y = np.array(series[node])
        train_y: np.ndarray = series_y[: data.train_window]
        test_y: np.ndarray = series_y[-data.test_window :]
        z_scores = zscore(train_y, test_y)
        z_score = self._aggregator(abs(z_scores))
        score = Score(z_score)
        score["z-score"] = z_score
        return score


class Model:
    """
    A combination of the algorithms
    """

    def __init__(
        self,
        graph_factory: GraphFactory,
        scorers: Sequence[Scorer],
        names: Tuple[str, ...] = None,
    ):
        if not scorers:
            raise ValueError("Please provide at least one scorer")
        self._graph_factory = graph_factory
        self._scorers = scorers
        num_scorers = len(scorers)
        if names is None:
            names = []
        names = list(names) + [None] * (1 + num_scorers - len(names))
        self._name_graph = names[0] or graph_factory.__class__.__name__
        self._names_scorer = [
            obj.__class__.__name__ if name is None else name
            for name, obj in zip(names[1:][:num_scorers], scorers)
        ]
        self._name = "-".join([self._name_graph] + self._names_scorer)

    @property
    def name(self) -> str:
        """
        Model name
        """
        return self._name

    def analyze(self, data: CaseData, current: float) -> List[Tuple[Node, Score]]:
        """
        Conduct root cause analysis
        """
        # TODO: Cache each step
        graph = self._graph_factory.create(data=data, current=current)
        scores: Dict[Node, Score] = None
        for scorer in self._scorers:
            scores = scorer.score(
                graph=graph, data=data, current=current, scores=scores
            )
        return sorted(scores.items(), key=lambda item: item[1].key, reverse=True)


class Evaluation:
    """
    Evalution results
    """

    def __init__(self, recommendation_num: int = 5):
        self._recommendation_num = recommendation_num
        self._accuracy = {k: 0.0 for k in range(1, recommendation_num + 1)}
        self._ranks: List[List[Node]] = []

    def __call__(self, ranks: Sequence[Node], answers: Set[Node]):
        self._ranks.append(ranks[: self._recommendation_num])
        answer_num = len(answers)
        for k in range(1, self._recommendation_num + 1):
            self._accuracy[k] += len(answers.intersection(ranks[:k])) / min(
                k, answer_num
            )

    def dump(self, filename: str) -> None:
        """
        Dump ranks into the given file
        """
        with open(filename, "w", encoding="UTF-8") as obj:
            json.dump(
                [[node.asdict() for node in ranks] for ranks in self._ranks],
                obj,
                ensure_ascii=False,
                indent=2,
            )

    def load(self, filename: str, answers: Sequence[Set[Node]]) -> None:
        """
        Load ranks from the given file
        """
        self._ranks = []
        with open(filename, "r", encoding="UTF-8") as obj:
            report: List[List[dict]] = json.load(obj)
        for ranks, answer in zip(report, answers):
            self(
                [Node(entity=node["entity"], metric=node["metric"]) for node in ranks],
                answer,
            )

    def accuracy(self, k: int) -> float:
        """
        AC@k is the average of accuracy@k among cases

        For each case, accuracy@k = |ranks[:k] \\cap answers| / min(k, |answers|)
        """
        if k not in self._accuracy or not self._ranks:
            return None
        return self._accuracy[k] / len(self._ranks)

    def average(self, k: int) -> float:
        """
        Avg@k = \\sum_{j=1}^{k} AC@j / k
        """
        if k not in self._accuracy or not self._ranks:
            return None
        return sum(self.accuracy(i) for i in range(1, k + 1)) / k


def evaluate(
    model: Model,
    cases: Sequence[Case],
    delay: int = 300,
    output_dir: str = None,
) -> Evaluation:
    """
    Evaluate the composition of Scorer and Ranker

    delay: the expected interval in seconds to conduct root cause analysis
        after the case is detected
    """
    logger = logging.getLogger(f"{evaluate.__module__}.{evaluate.__name__}")
    report = Evaluation()
    if output_dir is not None:
        output_filename = os.path.join(output_dir, f"{model.name}.json")
        if os.path.exists(output_filename):
            report.load(output_filename, [case.answer for case in cases])
            return report

    for index, case in enumerate(cases):
        logger.debug("Analyze case %d", index)
        ranks = model.analyze(data=case.data, current=case.data.detect_time + delay)
        report(ranks=[node for node, _ in ranks], answers=case.answer)
    if output_dir is not None:
        report.dump(output_filename)
    return report
