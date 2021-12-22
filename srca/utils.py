"""
Utilities
"""
import csv
import datetime
import json
import logging
from typing import Callable
from typing import Sequence
from typing import TypeVar


_Template = TypeVar("_Template")

ENCODING = "UTF-8"


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


def dump_csv(filename: str, data: Sequence[Sequence], headers: Sequence[str] = None):
    """
    Dump data into a csv file
    """
    with open(filename, "w", encoding=ENCODING) as obj:
        writer = csv.writer(obj)
        if headers is not None:
            writer.writerow(headers)
        writer.writerows(data)


def load_csv(filename: str):
    """
    Load data from a csv file
    """
    with open(filename, encoding=ENCODING) as obj:
        reader = csv.reader(obj)
        for row in reader:
            yield row


def dump_json(filename: str, data):
    """
    Dump data into a json file
    """
    with open(filename, "w", encoding=ENCODING) as obj:
        json.dump(data, obj, ensure_ascii=False, indent=2)


def load_json(filename: str):
    """
    Load data from a json file
    """
    with open(filename, encoding=ENCODING) as obj:
        return json.load(obj)
