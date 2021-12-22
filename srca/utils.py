"""
Utilities
"""
import datetime
import logging
from typing import Callable
from typing import TypeVar


_Template = TypeVar("_Template")


def get_logging_level() -> int:
    """
    Detect current logging level
    """
    return logging.getLogger().getEffectiveLevel()


class _LoggingWrapper:
    def __init__(self, logging_level: int, fun: Callable[..., _Template]):
        self._logging_level = logging_level
        self._fun = fun

    def __call__(self, *args, **kwargs):
        logging.basicConfig(level=self._logging_level)
        return self._fun(*args, **kwargs)


def require_logging(fun: Callable[..., _Template]) -> Callable[..., _Template]:
    """
    Set logging level in a new process
    """
    logging_level = get_logging_level()
    return _LoggingWrapper(logging_level=logging_level, fun=fun)


class Timer:
    """
    Record duration as a context manager
    """

    def __init__(self, name: str):
        self._name = name
        class_name = f"{self.__class__.__module__}.{self.__class__.__name__}"
        self._logger = logging.getLogger(class_name)

        self._start: datetime.datetime = None

    def __enter__(self) -> datetime.datetime:
        self._start = datetime.datetime.now()
        self._logger.info("%s starts at %s", self._name, self._start)
        return self._start

    def __exit__(self, *_):
        duration = datetime.datetime.now() - self._start
        self._logger.info("Duration of %s: %.6f", self._name, duration.total_seconds())
