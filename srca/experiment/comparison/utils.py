"""
Utilities
"""
from abc import ABC
import dataclasses
from enum import Enum
from textwrap import indent
from typing import Dict
from typing import Iterable
from typing import Tuple
from typing import Type
from typing import Union

from ...utils import load_json


_ALPHAS = (0.01, 0.05, 0.1, 0.5)
_MAX_CONDS_DIMS = (2, 3, 5, 10, None)
_TAU_MAXS = (0, 1, 2, 3)
_RISKS = (1e-2, 1e-3, 1e-4)

_ZERO_TO_ONE = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
_TRUE_AND_FALSE = (True, False)


class GraphMethod(Enum):
    """Supported graph factories"""

    PC_GAUSS = "PC-gauss"
    PC_GSQ = "PC-gsq"
    PCTS = "PCTS"


class ADMethod(Enum):
    """Supported anomaly detection scorers"""

    NSIGMA = "NSigma"
    SPOT = "SPOT"


class DFSMethod(Enum):
    """Supported DFS-based scorers"""

    DFS = "DFS"
    MICRO_SCOPE = "Microscope"
    MICRO_HECL = "MicroHECL"


class RandomWalkMethod(Enum):
    """Supported random walk-based scorers"""

    MICRO_CAUSE = "MicroCause"
    CLOUD_RANGER = "CloudRanger"


class OtherMethod(Enum):
    """Supported other scorers"""

    CRD = "CRD"


GRAPH_METHODS, AD_METHODS, DFS_METHODS, RANDOM_WALK_METHODS, OTHER_METHODS = (
    tuple(method.value for method in methods)
    for methods in (
        GraphMethod,
        ADMethod,
        DFSMethod,
        RandomWalkMethod,
        OtherMethod,
    )
)


@dataclasses.dataclass
class Params(ABC):
    """Parameter template"""

    METHOD = Enum

    method: Tuple[Enum] = dataclasses.field(default=())

    def __post_init__(self):
        if self.__class__ is Params:
            raise self._prohibit_instantiate()
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if field.metadata.get("iterable", True) and (
                isinstance(value, str) or not isinstance(value, Iterable)
            ):
                value = [value]
            if field.name == "method":
                value = [self.METHOD(item) for item in value]
            setattr(self, field.name, value)

    @classmethod
    def _prohibit_instantiate(cls):
        return TypeError(f"Can't instantiate abstract class {cls.__name__}")

    @classmethod
    def field_doc(cls, field: dataclasses.Field) -> str:
        """
        Generate doc string for Field

        >>> from dataclasses import field
        >>> age: int = field(default=1, metadata={"help": "example"})
        >>> age.name = "age"
        >>> Params.field_doc(age)
        'age: example. [Default: 1]'
        """
        if isinstance(field.type, type) and issubclass(field.type, cls):
            return "\n".join(
                [
                    f"{field.name}: {field.type.__doc__}",
                    indent(cls.dataclass_doc(field.type), " " * 2),
                ]
            )
        help_message = field.metadata.get("help")
        if help_message:
            return f"{field.name}: {help_message}. [Default: {field.default}]"
        return f"{field.name}: [Default: {field.default}]"

    @classmethod
    def dataclass_doc(cls, obj: Type[object]) -> str:
        """
        Generate doc string for a class wrapped by dataclass

        >>> Params.dataclass_doc(Params)
        '- method: [Default: ()]'
        """
        return "\n".join(
            [f"- {cls.field_doc(field)}" for field in dataclasses.fields(obj)]
        )


@dataclasses.dataclass
class GraphParams(Params):
    """Parameters for graph factories"""

    METHOD = GraphMethod

    method: Tuple[GraphMethod] = dataclasses.field(default=GRAPH_METHODS)
    alpha: Tuple[float, ...] = dataclasses.field(
        default=_ALPHAS, metadata={"help": "Thresholds for p-value"}
    )
    max_conds_dim: Tuple[float, ...] = dataclasses.field(
        default=_MAX_CONDS_DIMS,
        metadata={"help": "The maximum size of condition set for PC and PCTS"},
    )
    tau_max: Tuple[int, ...] = dataclasses.field(
        default=_TAU_MAXS, metadata={"help": "The maximum lag considered by PCTS"}
    )
    num_cores: int = dataclasses.field(
        default=1,
        metadata={
            "iterable": False,
            "help": "The number of cores to be used for parallel estimation of skeleton"
            " for PC num_cores shall be an integer",
        },
    )


@dataclasses.dataclass
class ScorerParams(Params):
    """Parameters for scorer with an extra graph fields"""

    graph: GraphParams = dataclasses.field(
        default_factory=GraphParams, metadata={"iterable": False}
    )

    def __post_init__(self):
        if self.__class__ is ScorerParams:
            raise self._prohibit_instantiate()
        if isinstance(self.graph, dict):
            self.graph = GraphParams(**self.graph)
        super().__post_init__()


@dataclasses.dataclass
class ADParams(Params):
    """Parameters for anomaly detection scorers"""

    METHOD = ADMethod

    method: Tuple[ADMethod] = dataclasses.field(default=AD_METHODS)
    risk: Tuple[float, ...] = dataclasses.field(
        default=_RISKS, metadata={"help": "The probability of risk for SPOT"}
    )


@dataclasses.dataclass
class DFSParams(ScorerParams):
    """Parameters for DFS-based scorers"""

    METHOD = DFSMethod

    method: Tuple[DFSMethod] = dataclasses.field(default=DFS_METHODS)
    detector: ADParams = dataclasses.field(
        default_factory=ADParams, metadata={"iterable": False}
    )
    stop_threshold: Tuple[float, ...] = dataclasses.field(
        default=_ZERO_TO_ONE, metadata={"help": "Threshold for MicroHECL"}
    )

    def __post_init__(self):
        if isinstance(self.detector, dict):
            self.detector = ADParams(**self.detector)
        super().__post_init__()


@dataclasses.dataclass
class RandomWalkParams(ScorerParams):
    """Parameters for random walk-based scorers"""

    METHOD = RandomWalkMethod

    method: Tuple[RandomWalkMethod] = dataclasses.field(default=RANDOM_WALK_METHODS)
    rho: Tuple[float, ...] = dataclasses.field(
        default=_ZERO_TO_ONE, metadata={"help": "Back-ward probability"}
    )
    remove_sla: Tuple[bool, ...] = dataclasses.field(
        default=_TRUE_AND_FALSE,
        metadata={"help": "Whether to disable forwarding to the SLA"},
    )
    beta: Tuple[float, ...] = dataclasses.field(
        default=_ZERO_TO_ONE, metadata={"help": "For second order random walk"}
    )


@dataclasses.dataclass
class OtherParams(ScorerParams):
    """Parameters for other scorers"""

    METHOD = OtherMethod

    method: Tuple[OtherMethod] = dataclasses.field(default=OTHER_METHODS)


@dataclasses.dataclass
class ModelParams:
    """
    Specify model parameters
    """

    anomaly_detection: ADParams = dataclasses.field()
    dfs: DFSParams = dataclasses.field()
    random_walk: RandomWalkParams = dataclasses.field()
    other: OtherParams = dataclasses.field()

    def __init__(self, params: Union[Dict[str, dict], str] = None):
        """
        params shall be dict. The name of a json file is also accepted.
        """
        if params is None:
            params = {}
        elif isinstance(params, str):
            params = load_json(params)
        if not isinstance(params, dict):
            raise ValueError(
                f"ModelParams requires dict or json dict, not {type(params)}"
            )
        for key, value in params.items():
            if not isinstance(value, dict):
                raise ValueError(
                    f"ModelParams requires dict for {key}, not {type(value)}"
                )

        for field in dataclasses.fields(self):
            setattr(self, field.name, field.type(**params.get(field.name, {})))


ModelParams.__init__.__doc__ += indent(
    "\n".join(["\nParameters:\n", Params.dataclass_doc(ModelParams)]), " " * 8
)
