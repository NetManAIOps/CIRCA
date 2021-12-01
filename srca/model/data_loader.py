"""
Define the interface for algorithms to access data
"""
from abc import ABC
from datetime import timedelta
from typing import Dict
from typing import Sequence
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd

from .graph import Node


class DataLoader(ABC):
    """
    The abstract interface to access data
    """

    @property
    def entities(self) -> Sequence[str]:
        """
        Fetch names of available entities
        """
        raise NotImplementedError

    @property
    def metrics(self) -> Dict[str, Sequence[str]]:
        """
        Fetch available metrics as a mapping from entity names to metric names
        """
        raise NotImplementedError

    @property
    def nodes(self) -> Sequence[Node]:
        """
        Pairs of entity and metric
        """
        return [
            Node(entity=entity, metric=metric)
            for entity, metrics in self.metrics.items()
            for metric in metrics
        ]

    def load(
        self,
        entity: str,
        metric: str,
        start: float,
        end: float,
        interval: timedelta,
        **kwargs,
    ) -> Union[None, Sequence[float]]:
        # pylint: disable=too-many-arguments
        """
        Load the time series for the given metric of the given entity

        start: expected start time of the time series,
            which is a unix timestamp in seconds
        end: expected end time of the time series,
            which is a unix timestamp in seconds
        interval: interval between two data points
        """
        raise NotImplementedError

    @staticmethod
    def preprocess(
        time_series: Sequence[Tuple[float, float]],
        start: float,
        end: float,
        interval: timedelta,
        **kwargs,
    ) -> Sequence[float]:
        """
        Truncate the time series and fill missing data points

        This method is edited from k-Shape, which is released under the MIT license.
        See https://github.com/sieve-microservices/kshape/blob/master/README.md

        time_series: 2-tuples (timestamp, value),
            where timestamp is a unix timestamp in seconds
        start: expected start time of the time series,
            which is a unix timestamp in seconds
        end: expected end time of the time series,
            which is a unix timestamp in seconds
        interval: interval between two data points
        """
        if not time_series:
            return None
        data: np.ndarray = np.array(time_series)
        # 1. Truncate the time series and make sure that [start, end] is the boundry
        data = data[(data[:, 0] >= start) & (data[:, 0] <= end), :]
        if len(data) == 0:
            return None
        data = np.vstack(
            [
                data,
                np.array([(start, np.nan), (end, np.nan)]),
            ]
        )

        # 2. Fill missing data points with fixed frequency
        data_frame = pd.DataFrame(data)
        data_frame[0] = pd.to_datetime(
            data_frame[0], unit=kwargs.get("unit", "s"), utc=True
        )
        data_frame: pd.DataFrame = (
            data_frame.set_index(0).resample(interval, origin="start").mean()
        )
        data_frame.interpolate(method="time", limit_direction="both", inplace=True)
        data_frame.fillna(method="bfill", inplace=True)

        # 3. Return values only
        data_frame.sort_index(inplace=True)
        return tuple(data_frame[1])


class MemoryDataLoader(DataLoader):
    """
    Implement DataLoader with data in memory
    """

    def __init__(self, data: Dict[str, Dict[str, Sequence[Tuple[float, float]]]]):
        """
        data: data[entity][metric] is the 2-tuples (timestamp, value)
            for the given metric of the given entity
        """
        self._data = data

    @property
    def entities(self) -> Sequence[str]:
        return tuple(self._data.keys())

    @property
    def metrics(self) -> Dict[str, Sequence[str]]:
        return {entity: tuple(metrics.keys()) for entity, metrics in self._data.items()}

    def load(
        self,
        entity: str,
        metric: str,
        start: float,
        end: float,
        interval: timedelta,
        **kwargs,
    ) -> Union[None, Sequence[float]]:
        # pylint: disable=too-many-arguments
        if entity not in self._data or metric not in self._data[entity]:
            return None
        return self.preprocess(
            time_series=self._data[entity][metric],
            start=start,
            end=end,
            interval=interval,
            **kwargs,
        )
