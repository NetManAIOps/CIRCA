"""
Abstract interfaces for root cause analysis
"""
from abc import ABC
from typing import Dict
from typing import List

from ..model.case import CaseData
from ..model.graph import Node


class Score:
    """
    Node score which can be extended to more than a single float
    """

    def __init__(self, score: float):
        self._score = score

    def __eq__(self, obj) -> bool:
        if isinstance(obj, Score):
            return self.score == obj.score
        return False

    @property
    def score(self) -> float:
        """
        The overall score
        """
        return self._score


class Scorer(ABC):
    """
    The abstract interface to score nodes
    """

    def score(self, data: CaseData, current: float) -> Dict[Node, Score]:
        """
        Estimate suspicious nodes

        The more suspicious a node, the higher the score.
        """
        raise NotImplementedError


class Ranker(ABC):
    """
    The abstract interface to rank nodes
    """

    def rank(
        self, data: CaseData, scores: Dict[Node, Score], current: float
    ) -> List[Node]:
        """
        Rank suspicious nodes

        The most suspicious shall be ranked as the first.
        """
        raise NotImplementedError
