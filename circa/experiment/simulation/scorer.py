"""
Scorers with the ground truth graph in the simulation dataset
"""
from typing import Dict
from typing import Sequence
from typing import Tuple

import numpy as np

from ...alg.ci import RHTScorer
from ...model.case import CaseData
from ...model.graph import Node


class SimRHTScorer(RHTScorer):
    """
    Update RHTScorer with the ground truth parents in the simulation dataset
        Pa(V_{i}^{(t)}) = Pa^{(t)}(V_{i}^{t}) \\cup {V_{i}^{(t - 1)}}
    """

    def split_data(
        self,
        data: Dict[Node, Sequence[float]],
        node: Node,
        parents: Sequence[Node],
        case_data: CaseData,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data for training and testing

        Return (train_x, test_x, train_y, test_y)
        """
        series = [data[parent] for parent in parents if parent in data]
        # from [x1, x2, ..., xn] to [xn, x1, x2, ...]
        series.append(np.roll(data[node], 1))
        series = np.array(series).T
        series_x: np.ndarray = series[1:, :]
        series_y = np.array(data[node][1:])

        return self._split_train_test(
            series_x=series_x,
            series_y=series_y,
            train_window=case_data.train_window - 1,
            test_window=case_data.test_window,
        )
