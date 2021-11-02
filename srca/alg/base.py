"""
Abstract interfaces for root cause analysis
"""
from abc import ABC
from typing import Dict
from typing import List

from ..model.case import CaseData
from ..model.graph import Node


class Scorer(ABC):
    """
    The abstract interface to score nodes
    """

    def score(self, data: CaseData, current: float) -> Dict[Node, float]:
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
        self, data: CaseData, scores: Dict[Node, float], current: float
    ) -> List[Node]:
        """
        Rank suspicious nodes

        The most suspicious shall be ranked as the first.
        """
        raise NotImplementedError
