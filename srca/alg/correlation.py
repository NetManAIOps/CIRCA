"""
Correlation
"""
from typing import Dict
from typing import Sequence
from typing import Set

import pandas as pd
import pingouin as pg
from scipy.stats import pearsonr

from .base import Score
from .common import DecomposableScorer
from ..model.case import CaseData
from ..model.graph import Graph
from ..model.graph import Node


def partial_correlation(
    node: Node,
    cause: Node,
    graph: Graph,
    series: Dict[Node, Sequence[float]],
    corr_type: str = "pearson",
) -> float:
    """
    Partial correlation coefficient

    corr_type: For the "method" parameter of pingouin.partial_corr
    """
    if node == cause:
        return 1

    confounders = graph.parents(node) | graph.parents(cause)
    confounders -= {node, cause}
    data_frame = pd.DataFrame(series)

    nodes: Set[Node] = set(data_frame.index)
    confounders = {
        confounder
        for confounder in confounders
        if confounder in nodes and len(data_frame[confounder].unique()) > 1
    }

    return pg.partial_corr(
        data=data_frame, x=node, y=cause, covar=confounders, method=corr_type
    )["r"].values[0]


class CorrelationScorer(DecomposableScorer):
    """
    Score nodes by correlation
    """

    def score_node(
        self,
        graph: Graph,
        series: Dict[Node, Sequence[float]],
        node: Node,
        data: CaseData,
    ) -> Score:
        series_node = series[node]
        correlation, p_value = pearsonr(series_node, series[data.sli])

        score = Score(abs(correlation))
        score["pearson"] = correlation
        score["p-value"] = p_value
        return score


class PartialCorrelationScorer(DecomposableScorer):
    """
    Score nodes by partial correlation coefficient
    """

    def score_node(
        self,
        graph: Graph,
        series: Dict[Node, Sequence[float]],
        node: Node,
        data: CaseData,
    ) -> Score:
        data_frame = pd.DataFrame(series)
        correlation = partial_correlation(data.sli, node, graph, data_frame)
        score = Score(abs(correlation))
        score["partial-correlation"] = correlation
        return score
