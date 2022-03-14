"""
Common utilities
"""
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import as_completed
import logging
from multiprocessing import Process
from multiprocessing import Queue
import os
import queue
from typing import Dict
from typing import List
from typing import Sequence
from typing import Set
from typing import Tuple

import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler

from .base import Score
from .base import Scorer
from ..graph import GraphFactory
from ..model.case import Case
from ..model.case import CaseData
from ..model.graph import Graph
from ..model.graph import LoadingInvalidGraphException
from ..model.graph import Node
from ..utils import dump_json
from ..utils import load_json
from ..utils import require_logging


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

    def _score(
        self,
        candidates: Sequence[Node],
        series: Dict[Node, Sequence[float]],
        graph: Graph,
        data: CaseData,
    ):
        results: Dict[Node, Score] = {}
        for node in candidates:
            score = self.score_node(graph, series, node, data)
            if score is not None:
                results[node] = score
        return results

    def score(
        self,
        graph: Graph,
        data: CaseData,
        current: float,
        scores: Dict[Node, Score] = None,
    ) -> Dict[Node, Score]:
        series = data.load_data(graph, current)
        candidates = list(series.keys()) if scores is None else list(scores.keys())

        if self._max_workers >= 2:
            results: Dict[Node, Score] = {}
            with ProcessPoolExecutor(max_workers=self._max_workers) as executor:
                tasks = [
                    executor.submit(
                        require_logging(self._score),
                        candidates=candidates[i :: self._max_workers],
                        series=series,
                        graph=graph,
                        data=data,
                    )
                    for i in range(self._max_workers)
                ]
                for task in as_completed(tasks):
                    results.update(task.result())
        else:
            results = self._score(
                candidates=candidates, series=series, graph=graph, data=data
            )

        if scores is None:
            return results
        return {node: scores[node].update(score) for node, score in results.items()}


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

    @staticmethod
    def dump(scores: Dict[Node, Score], filename: str):
        """
        Dump scores into the given file
        """
        data = [
            dict(node=node.asdict(), score=score.asdict())
            for node, score in scores.items()
        ]
        dump_json(filename=filename, data=data)

    @staticmethod
    def load(filename: str) -> Dict[Node, Score]:
        """
        Load scores from the given file
        """
        data: List[Dict[str, dict]] = load_json(filename)
        return {Node(**item["node"]): Score(**item["score"]) for item in data}

    @property
    def name(self) -> str:
        """
        Model name
        """
        return self._name

    def analyze(
        self, data: CaseData, current: float, output_dir: str = None
    ) -> List[Tuple[Node, Score]]:
        """
        Conduct root cause analysis
        """
        # 1. Create a graph
        if output_dir is not None:
            graph_filename = os.path.join(output_dir, f"{self._name_graph}.json")
            graph = None
            if os.path.isfile(graph_filename):
                graph = self._graph_factory.load(graph_filename)
            if graph is None:
                graph = self._graph_factory.create(data=data, current=current)
        else:
            graph = self._graph_factory.create(data=data, current=current)

        # 2. Score nodes
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
            self._accuracy[k] += len(answers.intersection(ranks[:k])) / answer_num

    @property
    def num(self) -> int:
        """
        Number of cases
        """
        return len(self._ranks)

    def dump(self, filename: str) -> None:
        """
        Dump ranks into the given file
        """
        dump_json(
            filename=filename,
            data=[[node.asdict() for node in ranks] for ranks in self._ranks],
        )

    def load(self, filename: str, answers: Sequence[Set[Node]]) -> None:
        """
        Load ranks from the given file
        """
        self._ranks = []
        report: List[List[dict]] = load_json(filename)
        for ranks, answer in zip(report, answers):
            self(
                [Node(entity=node["entity"], metric=node["metric"]) for node in ranks],
                answer,
            )

    def accuracy(self, k: int) -> float:
        """
        AC@k is the average of accuracy@k among cases

        For each case, accuracy@k = |ranks[:k] \\cap answers| / |answers|
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


def _analyze(
    consumer: Queue, model: Model, case: Case, current: float, output_dir: str
):
    try:
        ranks = model.analyze(
            data=case.data,
            current=current,
            output_dir=output_dir,
        )
    except LoadingInvalidGraphException:
        ranks = []
    consumer.put(ranks)


def evaluate(
    model: Model,
    cases: Sequence[Case],
    delay: int = 300,
    output_dir: str = None,
    timeout: int = 3600,
) -> Evaluation:
    """
    Evaluate the composition of Scorers

    delay: the expected interval in seconds to conduct root cause analysis
        after the case is detected
    """
    logger = logging.getLogger(
        ".".join([evaluate.__module__, evaluate.__name__, model.name])
    )
    report = Evaluation()
    if output_dir is not None:
        output_filename = os.path.join(output_dir, f"{model.name}.json")
        if os.path.exists(output_filename):
            report.load(output_filename, [case.answer for case in cases])
            return report
        os.makedirs(output_dir, exist_ok=True)

    for index, case in enumerate(cases):
        logger.debug("Analyze case %d", index)
        case_output_dir = None
        if output_dir is not None:
            case_output_dir = os.path.join(output_dir, str(index))

        consumer = Queue()
        task = Process(
            target=require_logging(_analyze),
            kwargs=dict(
                consumer=consumer,
                model=model,
                case=case,
                current=case.data.detect_time + delay,
                output_dir=case_output_dir,
            ),
        )
        task.start()
        task.join(timeout=timeout)
        if task.is_alive():
            task.terminate()
            logger.warning("Timeout for case %d", index)
            ranks = []
        else:
            try:
                ranks = consumer.get(timeout=10)
            except queue.Empty:
                logger.warning("Unexpected termination for case %d", index)
                ranks = []
        report(ranks=[node for node, _ in ranks], answers=case.answer)
    if output_dir is not None:
        report.dump(output_filename)
    return report
