"""
Utilities
"""
import csv
import datetime
import json
import logging
import signal
from typing import Callable
from typing import Sequence
from typing import TypeVar
import warnings


try:
    _ = signal.SIGALRM
    _HAS_SIGALRM = True
except AttributeError:
    _HAS_SIGALRM = False
_Template = TypeVar("_Template")

ENCODING = "UTF-8"


def get_logging_level() -> int:
    """
    Detect current logging level
    """
    return logging.getLogger().getEffectiveLevel()


def silence_third_party():
    """
    Silence logging of third party packages
    """
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


class _LoggingWrapper:
    def __init__(self, logging_level: int, fun: Callable[..., _Template]):
        self._logging_level = logging_level
        self._fun = fun

    def __call__(self, *args, **kwargs):
        logging.basicConfig(level=self._logging_level)
        if self._logging_level > logging.DEBUG:
            silence_third_party()
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

    @property
    def start(self) -> datetime.datetime:
        """
        Start time
        """
        return self._start

    @property
    def duration(self) -> datetime.timedelta:
        """
        Elapsed time since start
        """
        return datetime.datetime.now() - self._start

    def __enter__(self):
        self._start = datetime.datetime.now()
        self._logger.info("%s starts at %s", self._name, self._start)
        return self

    def __exit__(self, *_):
        self._logger.info(
            "Duration of %s: %.6f", self._name, self.duration.total_seconds()
        )


class Timeout:
    """
    Control execution duration as a context manager

    Stolen from the following answer, which is under the "CC BY-SA 4.0" license
    https://stackoverflow.com/a/39773925

    Exemples:
    >>> import time
    >>> with Timeout(seconds=3):
    ...     time.sleep(4)
    Traceback (most recent call last):
        ...
    TimeoutError
    """

    def __init__(self, seconds: int = 0):
        if not _HAS_SIGALRM:
            warnings.warn(
                "signal.SIGALRM is not supported and "
                f"{Timeout.__module__}.{Timeout.__name__} cannot work"
            )
        self._seconds = int(seconds)

    def _handle_timeout(self, *_):
        # pylint: disable=no-self-use
        raise TimeoutError

    def __enter__(self):
        if _HAS_SIGALRM:
            signal.signal(signal.SIGALRM, self._handle_timeout)
            signal.alarm(self._seconds)

    def __exit__(self, *_):
        if _HAS_SIGALRM:
            signal.alarm(0)


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
