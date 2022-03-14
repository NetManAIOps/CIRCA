"""
Test suites for utilities
"""
import time

import pytest

from circa.utils import Timeout


def test_timeout():
    """
    Timeout shall terminate the inside task after the given time
    """
    origin_value = 0
    new_value = origin_value + 1
    value = origin_value

    with Timeout(seconds=1):
        with pytest.raises(TimeoutError):
            time.sleep(2)
            value = new_value
    assert value == origin_value

    with Timeout(seconds=1):
        time.sleep(0.5)
        value = new_value
    assert value == new_value
