"""
Test suites for DataLoader
"""
from datetime import timedelta

import numpy as np

import pytest

from circa.model.data_loader import DataLoader
from circa.model.data_loader import MemoryDataLoader


@pytest.mark.parametrize(
    ("start", "end", "interval"),
    [
        (100, 280, timedelta(seconds=30)),
        (100, 280, timedelta(seconds=60)),
        (60, 200, timedelta(seconds=60)),
        (60, 280, timedelta(seconds=60)),
        (60, 360, timedelta(seconds=60)),
        (120, 360, timedelta(seconds=60)),
        (0, 90, timedelta(seconds=60)),
        (300, 600, timedelta(seconds=60)),
    ],
)
def test_preprocess(start: float, end: float, interval: timedelta):
    """
    Test case for DataLoader.preprocess
    """
    time_series = [
        (100, 1.5),
        (159, 2),
        (221, 3),
        (280, 1),
    ]
    params = dict(start=start, end=end, interval=interval)
    assert DataLoader.preprocess(time_series=[], **params) is None
    data = DataLoader.preprocess(time_series=time_series, **params)
    if start > max(timestamp for timestamp, _ in time_series) or end < min(
        timestamp for timestamp, _ in time_series
    ):
        assert data is None
    else:
        assert len(data) == int((end - start) / interval.total_seconds()) + 1
        assert all(
            isinstance(item, (float, int)) and not np.isnan(item) for item in data
        )


def test_memory_data_loader():
    """
    Test case for MemoryDataLoader
    """
    data = {
        "db": {
            "transaction per second": [
                (100, 1000),
                (159, 1200),
                (221, 1100),
            ],
            "average active sessions": [
                (100, 10),
                (159, 12.5),
                (221, 10.3),
            ],
        },
        "storage": {
            "iops": [
                (99, 2000),
                (159, 5000),
                (219, 4000),
            ],
        },
    }
    data_loader = MemoryDataLoader(data)
    assert set(data_loader.entities) == {"db", "storage"}
    metrics = {entity: set(metrics) for entity, metrics in data_loader.metrics.items()}
    assert metrics == {
        "db": {"transaction per second", "average active sessions"},
        "storage": {
            "iops",
        },
    }
    params = dict(start=60, end=300, interval=timedelta(seconds=60))
    assert len(data_loader.load("db", "transaction per second", **params)) == 5

    assert data_loader.load("no-entity", "iops", **params) is None
    assert data_loader.load("storage", "no-metric", **params) is None
    assert data_loader.load("storage", "iops", **params) is not None
