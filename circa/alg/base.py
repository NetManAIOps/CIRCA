"""
Abstract interfaces for root cause analysis
"""
from abc import ABC
from typing import Callable
from typing import Dict
from typing import Sequence
from typing import Union

from ..model.case import CaseData
from ..model.graph import Graph
from ..model.graph import Node


class Score:
    """
    The more suspicious a node, the higher the score.
    """

    def __init__(self, score: float, info: dict = None, key: tuple = None):
        self._score = score
        self._key = key
        self._info = {} if info is None else info

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

    @property
    def key(self) -> float:
        """
        key for sorting
        """
        return self.score if self._key is None else self._key

    @key.setter
    def key(self, value: tuple):
        """
        Update key
        """
        self._key = value

    @property
    def info(self) -> dict:
        """
        Extra information
        """
        return self._info

    def update(self, score: "Score") -> "Score":
        """
        Update score and info
        """
        self._info.update(score.info)
        self.score = score.score
        self.key = score.key
        return self

    def asdict(self) -> Dict[str, Union[float, dict, tuple]]:
        """
        Serialized as a dict
        """
        data = {"score": self._score, "info": {**self._info}}
        if self._key is not None:
            data["key"] = self._key
        return data

    def __repr__(self) -> str:
        return str(self.asdict())


class Scorer(ABC):
    """
    The abstract interface to score nodes
    """

    def __init__(
        self,
        aggregator: Callable[[Sequence[float]], float] = max,
        max_workers: int = 1,
        seed: int = 0,
        cuda: bool = False,
    ):
        self._aggregator = aggregator
        self._max_workers = max_workers
        self._seed = seed
        self._cuda = cuda

    def score(
        self,
        graph: Graph,
        data: CaseData,
        current: float,
        scores: Dict[Node, Score] = None,
    ) -> Dict[Node, Score]:
        """
        Estimate suspicious nodes
        """
        raise NotImplementedError
