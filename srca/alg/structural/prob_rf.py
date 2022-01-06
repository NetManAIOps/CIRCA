"""
ProbRF is proposed by Silvery Fu, et.al. in NSDI 2021

This file is edited from `oracle/learn/prob_rf.py` in
    [their open-source code](https://github.com/perfd/perfd).

Reference:
[1] Silvery Fu, Saurabh Gupta, Radhika Mittal, and Sylvia Ratnasamy. On the Use of
    ML for Blackbox System Performance Prediction. NSDI 2021
"""
from collections import defaultdict
from itertools import chain
from typing import Sequence

import numpy as np
from sklearn.ensemble import RandomForestRegressor


class ProbRF:
    """
    Regress with a distribution
    """

    def __init__(self, seed=None):
        self._forest = RandomForestRegressor(random_state=seed)
        self._leaf2samples = defaultdict(list)

    def fit(self, train_x: np.ndarray, train_y: np.ndarray):
        """
        Cluster train_y based on the forest
        """
        self._forest.fit(train_x, train_y)
        self._leaf2samples = defaultdict(list)
        indexes = self._forest.apply(train_x)
        for sample, leaves in zip(train_y, indexes):
            for tree_id, leaf in enumerate(leaves):
                self._leaf2samples[(tree_id, leaf)].append(sample.reshape(1))

    def predict(self, test_x: np.ndarray) -> Sequence[Sequence[np.ndarray]]:
        """
        Walk through the leave nodes and return the raw samples
        """
        indexes = self._forest.apply(test_x)
        return [
            list(
                chain(
                    *(
                        self._leaf2samples[(tree_id, leaf)]
                        for tree_id, leaf in enumerate(leaves)
                    )
                )
            )
            for leaves in indexes
        ]
