"""
Abstract interfaces for root cause analysis
"""
from abc import ABC
from typing import Callable
from typing import Dict
from typing import List
from typing import Sequence
from typing import Tuple

from ..model.case import CaseData
from ..model.graph import Graph
from ..model.graph import Node


class Score:
    """
    The more suspicious a node, the higher the score.
    """

    def __init__(self, score: float):
        self._score = score
        self._info = {}

    def __eq__(self, obj) -> bool:
        if isinstance(obj, Score):
            return self.score == obj.score
        return False

    def __getitem__(self, key: str):
        return self._info[key]

    def __setitem__(self, key: str, value):
        self._info[key] = value

    def get(self, key: str, default=None):
        """
        Return the value for key if key has been set, else default
        """
        return self._info.get(key, default)

    @property
    def score(self) -> float:
        """
        The overall score
        """
        return self._score

    @score.setter
    def score(self, value: float):
        """
        Update score
        """
        self._score = value

    def asdict(self) -> Dict[str, float]:
        """
        Serialized as a dict
        """
        return {"score": self.score, "info": {**self._info}}

    def __repr__(self) -> str:
        return str(self.asdict())


class GraphFactory(ABC):
    """
    The abstract interface to create Graph
    """

    def create(self, data: CaseData, current: float) -> Graph:
        """
        Create the Graph
        """
        raise NotImplementedError


class Scorer(ABC):
    """
    The abstract interface to score nodes
    """

    def __init__(self, aggregator: Callable[[Sequence[float]], float] = max):
        self._aggregator = aggregator

    def score(self, graph: Graph, data: CaseData, current: float) -> Dict[Node, Score]:
        """
        Estimate suspicious nodes
        """
        raise NotImplementedError


class Ranker(ABC):
    """
    The abstract interface to rank nodes
    """

    def rank(
        self, graph: Graph, data: CaseData, scores: Dict[Node, Score], current: float
    ) -> List[Tuple[Node, Score]]:
        """
        Rank suspicious nodes

        The most suspicious shall be ranked as the first.
        """
        raise NotImplementedError
