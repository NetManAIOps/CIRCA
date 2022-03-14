"""
Utilities
"""
from abc import ABC
import dataclasses
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

_ZERO_TO_ONE = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
_TRUE_AND_FALSE = (True, False)


@dataclasses.dataclass
class Params(ABC):
    """Parameter template"""

    HELP_MESSAGE = "help"
    IS_ITERABLE = "iterable"
    ABBREVIATION = "abbreviation"
    TARGET = "target"

    def __post_init__(self):
        if self.__class__ is Params:
            raise self._prohibit_instantiate()
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if (
                isinstance(field.type, type)
                and issubclass(field.type, Params)
                and isinstance(value, dict)
            ):
                value = field.type(**value)
            elif field.metadata.get(self.IS_ITERABLE, True) and (
                isinstance(value, str) or not isinstance(value, Iterable)
            ):
                value = [value]
            setattr(self, field.name, value)

    @classmethod
    def _prohibit_instantiate(cls):
        return TypeError(f"Can't instantiate abstract class {cls.__name__}")

    @classmethod
    def field_doc(cls, field: dataclasses.Field) -> str:
        """
        Generate doc string for Field

        >>> age: int = dataclasses.field(default=1, metadata={"help": "example"})
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
        help_message = field.metadata.get(cls.HELP_MESSAGE)
        if help_message:
            return f"{field.name}: {help_message}. [Default: {field.default}]"
        return f"{field.name}: [Default: {field.default}]"

    @classmethod
    def dataclass_doc(cls, obj: Type[object]) -> str:
        """
        Generate doc string for a class wrapped by dataclass

        >>> @dataclasses.dataclass
        ... class A:
        ...     method: tuple = dataclasses.field(default=())
        >>> Params.dataclass_doc(A)
        '- method: [Default: ()]'
        """
        return "\n".join(
            [f"- {cls.field_doc(field)}" for field in dataclasses.fields(obj)]
        )


@dataclasses.dataclass
class OptionalParams(Params):
    """
    Optional parameters

    Enable all options by default.
    With any arguments provided, options that are not specified will be disabled.
    """

    def __init__(self, **kwargs):
        # pylint: disable=super-init-not-called
        if kwargs:
            for field in dataclasses.fields(self):
                if field.name in kwargs:
                    setattr(self, field.name, field.type(**kwargs[field.name]))
                else:
                    setattr(self, field.name, None)
        else:
            for field in dataclasses.fields(self):
                setattr(self, field.name, field.type())


@dataclasses.dataclass
class GraphFactoryParams(Params):
    """Parameters for graph factories"""


@dataclasses.dataclass
class PCParams(GraphFactoryParams):
    """Parameters for PC algorithm"""

    alpha: Tuple[float, ...] = dataclasses.field(
        default=_ALPHAS,
        metadata={
            Params.HELP_MESSAGE: "Thresholds for p-value",
            Params.ABBREVIATION: "a",
        },
    )
    max_conds_dim: Tuple[float, ...] = dataclasses.field(
        default=_MAX_CONDS_DIMS,
        metadata={
            Params.HELP_MESSAGE: "The maximum size of condition set",
            Params.ABBREVIATION: "m",
        },
    )
    num_cores: int = dataclasses.field(
        default=1,
        metadata={
            Params.IS_ITERABLE: False,
            Params.HELP_MESSAGE: "The number of cores to be used for "
            "parallel estimation of skeleton num_cores shall be an integer",
        },
    )


@dataclasses.dataclass
class PCGaussParams(PCParams):
    """Parameters for PC algorithm with gauss CI test"""


@dataclasses.dataclass
class PCGSquareParams(PCParams):
    """Parameters for PC algorithm with G-square CI test"""


@dataclasses.dataclass
class PCTSParams(GraphFactoryParams):
    """Parameters for PCTS"""

    alpha: Tuple[float, ...] = dataclasses.field(
        default=_ALPHAS,
        metadata={
            Params.HELP_MESSAGE: "Thresholds for p-value",
            Params.ABBREVIATION: "a",
        },
    )
    max_conds_dim: Tuple[float, ...] = dataclasses.field(
        default=_MAX_CONDS_DIMS,
        metadata={
            Params.HELP_MESSAGE: "The maximum size of condition set",
            Params.ABBREVIATION: "m",
        },
    )
    tau_max: Tuple[int, ...] = dataclasses.field(
        default=_TAU_MAXS,
        metadata={
            Params.HELP_MESSAGE: "The maximum lag considered",
            Params.ABBREVIATION: "t",
        },
    )


@dataclasses.dataclass
class StructuralGraphParams(GraphFactoryParams):
    """Parameters for structural graph construction"""


@dataclasses.dataclass(init=False)
class GraphParams(OptionalParams):
    """Parameters for graph factories"""

    pc_gauss: PCGaussParams = dataclasses.field(
        default_factory=PCGaussParams, metadata={Params.IS_ITERABLE: False}
    )
    pc_gsq: PCGSquareParams = dataclasses.field(
        default_factory=PCGSquareParams, metadata={Params.IS_ITERABLE: False}
    )
    pcts: PCTSParams = dataclasses.field(
        default_factory=PCTSParams, metadata={Params.IS_ITERABLE: False}
    )
    structural: StructuralGraphParams = dataclasses.field(
        default_factory=StructuralGraphParams, metadata={Params.IS_ITERABLE: False}
    )


@dataclasses.dataclass
class ScorerParams(Params):
    """Parameters for scorer with an extra graph fields"""

    graph: GraphParams = dataclasses.field(
        default_factory=GraphParams, metadata={Params.IS_ITERABLE: False}
    )


@dataclasses.dataclass
class NSigmaParams(Params):
    """Parameters for n-sigma"""


@dataclasses.dataclass
class SPOTParams(Params):
    """Parameters for SPOT"""

    risk: Tuple[float, ...] = dataclasses.field(
        default=(1e-2, 1e-3, 1e-4),
        metadata={
            Params.HELP_MESSAGE: "The probability of risk",
            Params.ABBREVIATION: "p",
            Params.TARGET: "proba",
        },
    )


@dataclasses.dataclass(init=False)
class ADParams(OptionalParams):
    """Parameters for anomaly detection scorers"""

    nsigma: NSigmaParams = dataclasses.field(
        default_factory=NSigmaParams, metadata={Params.IS_ITERABLE: False}
    )
    spot: SPOTParams = dataclasses.field(
        default_factory=SPOTParams, metadata={Params.IS_ITERABLE: False}
    )


@dataclasses.dataclass
class DFSParams(ScorerParams):
    """Parameters for DFS-based scorers"""

    detector: ADParams = dataclasses.field(
        default_factory=ADParams, metadata={Params.IS_ITERABLE: False}
    )


@dataclasses.dataclass
class MicroscopeParams(DFSParams):
    """Parameters for Microscope"""


@dataclasses.dataclass
class MicroHECLParams(DFSParams):
    """Parameters for MicroHECL"""

    stop_threshold: Tuple[float, ...] = dataclasses.field(
        default=_ZERO_TO_ONE,
        metadata={
            Params.HELP_MESSAGE: "Threshold for stopping",
            Params.ABBREVIATION: "s",
        },
    )


@dataclasses.dataclass
class RandomWalkParams(ScorerParams):
    """Parameters for random walk-based scorers"""

    rho: Tuple[float, ...] = dataclasses.field(
        default=_ZERO_TO_ONE,
        metadata={
            Params.HELP_MESSAGE: "Back-ward probability",
            Params.ABBREVIATION: "r",
        },
    )


@dataclasses.dataclass
class MicroCauseParams(RandomWalkParams):
    """Parameters for MicroCause"""


@dataclasses.dataclass
class CloudRangerParams(RandomWalkParams):
    """Parameters for CloudRanger"""

    beta: Tuple[float, ...] = dataclasses.field(
        default=_ZERO_TO_ONE,
        metadata={
            Params.HELP_MESSAGE: "For second order random walk",
            Params.ABBREVIATION: "b",
        },
    )


@dataclasses.dataclass
class InvariantNetworkParams(Params):
    """Parameters for invariant network-based scorers"""

    discrete: Tuple[bool, ...] = dataclasses.field(
        default=_TRUE_AND_FALSE,
        metadata={
            Params.HELP_MESSAGE: "Whether to use binary or continous value for correlation",
            Params.ABBREVIATION: "d",
        },
    )
    gamma: Tuple[float, ...] = dataclasses.field(
        default=(0.2, 0.5, 0.8),
        metadata={
            Params.HELP_MESSAGE: "A parameter to balance "
            "the award for neighboring nodes to have similar status scores, and"
            "the penalty of large bias from the initial seeds",
            Params.ABBREVIATION: "c",
        },
    )
    tau: Tuple[float, ...] = dataclasses.field(
        default=(0.1, 1),
        metadata={
            Params.HELP_MESSAGE: "A larger tau typically results in more zeros in e",
            Params.ABBREVIATION: "t",
        },
    )


@dataclasses.dataclass
class ENMFParams(InvariantNetworkParams):
    """Parameters for ENMF"""

    use_softmax: Tuple[bool, ...] = dataclasses.field(
        default=_TRUE_AND_FALSE,
        metadata={
            Params.HELP_MESSAGE: "Whether to use confidence or anomaly score",
            Params.ABBREVIATION: "soft",
        },
    )


@dataclasses.dataclass
class CRDParams(InvariantNetworkParams):
    """Parameters for CRD"""

    num_cluster: Tuple[int, ...] = dataclasses.field(
        default=(2, 3, 4, 5),
        metadata={
            Params.HELP_MESSAGE: "The number of clusters in an invariant network",
            Params.ABBREVIATION: "nc",
        },
    )
    alpha: Tuple[float, ...] = dataclasses.field(
        default=(1.2, 1.5, 2),
        metadata={
            Params.HELP_MESSAGE: "A parameter in the Dirichlet distribution with alpha >= 1",
            Params.ABBREVIATION: "a",
        },
    )
    beta: Tuple[float, ...] = dataclasses.field(
        default=(0.1, 1, 10),
        metadata={
            Params.HELP_MESSAGE: "Balance the object functions for network clustering"
            "and broken score",
            Params.ABBREVIATION: "b",
        },
    )
    learning_rate: Tuple[float, ...] = dataclasses.field(
        default=(0.1, 1), metadata={Params.ABBREVIATION: "lr"}
    )


@dataclasses.dataclass
class CIParams(ScorerParams):
    """Parameters for causal inference-based scorers"""

    tau_max: Tuple[int, ...] = dataclasses.field(
        default=_TAU_MAXS,
        metadata={
            Params.HELP_MESSAGE: "The maximum lag to be considered",
            Params.ABBREVIATION: "t",
        },
    )
    regressor: Tuple[str, ...] = dataclasses.field(
        default=("linear", "svr", "rf", "mdn"),
        metadata={
            Params.HELP_MESSAGE: "Regressor",
        },
    )


@dataclasses.dataclass
class RHTParams(CIParams):
    """Parameters for RHT"""


@dataclasses.dataclass
class RHTDAParams(CIParams):
    """Parameters for RHT-DA"""


@dataclasses.dataclass
class ModelParams(ADParams):
    # pylint: disable=too-many-instance-attributes
    """
    Specify model parameters
    """

    dfs: DFSParams = dataclasses.field(
        default=None, metadata={Params.IS_ITERABLE: False}
    )
    micro_scope: MicroscopeParams = dataclasses.field(
        default=None, metadata={Params.IS_ITERABLE: False}
    )
    micro_hecl: MicroHECLParams = dataclasses.field(
        default=None, metadata={Params.IS_ITERABLE: False}
    )
    micro_cause: MicroCauseParams = dataclasses.field(
        default=None, metadata={Params.IS_ITERABLE: False}
    )
    cloud_ranger: CloudRangerParams = dataclasses.field(
        default=None, metadata={Params.IS_ITERABLE: False}
    )
    enmf: ENMFParams = dataclasses.field(
        default=None, metadata={Params.IS_ITERABLE: False}
    )
    crd: CRDParams = dataclasses.field(
        default=None, metadata={Params.IS_ITERABLE: False}
    )
    rht: RHTParams = dataclasses.field(
        default=None, metadata={Params.IS_ITERABLE: False}
    )
    rht_da: RHTDAParams = dataclasses.field(
        default=None, metadata={Params.IS_ITERABLE: False}
    )

    def __init__(self, params: Union[Dict[str, dict], str] = None):
        # pylint: disable=super-init-not-called
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
        super().__init__(**params)


ModelParams.__init__.__doc__ += indent(
    "\n".join(["\nParameters:\n", Params.dataclass_doc(ModelParams)]), " " * 8
)
