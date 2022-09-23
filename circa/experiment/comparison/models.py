"""
Utilities
"""
from abc import ABC
from itertools import product
from typing import Dict
from typing import Generator
from typing import List
from typing import Tuple

from sklearn.svm import SVR

from . import utils
from ...alg.base import Scorer
from ...alg.common import Model
from ...alg.common import NSigmaScorer
from ...alg.correlation import CorrelationScorer
from ...alg.correlation import PartialCorrelationScorer
from ...alg.invariant_network import CRDScorer
from ...alg.invariant_network import ENMFScorer
from ...alg.dfs import DFSScorer
from ...alg.dfs import MicroHECLScorer
from ...alg.evt import SPOTScorer
from ...alg.random_walk import RandomWalkScorer
from ...alg.random_walk import SecondOrderRandomWalkScorer
from ...alg.ci import DAScorer
from ...alg.ci import RHTScorer
from ...alg.ci.anm import ANMRegressor
from ...alg.ci.gmm import GMMRegressor
from ...alg.ci.gmm.mdn import MDNPredictor
from ...graph import GraphFactory
from ...graph.common import EmptyGraphFactory
from ...graph.pcts import PCTSFactory
from ...graph.r import PCAlgFactory
from ...graph.structural import StructuralGraphFactory


EMPTY_GRAPH_NAME = "Empty"


class ModelGetter(ABC):
    """
    Abstract interface to get models
    """

    def __init__(self, params: utils.Params):
        self._params = params

    @staticmethod
    def compose_parameters(
        values: tuple, names: tuple, abbrs: tuple
    ) -> Tuple[str, dict]:
        """
        Wrap parameters as (suffix str and parameter dict)
        """
        suffix = ""
        params = {}
        for value, name, abbr in zip(values, names, abbrs):
            if value is None:
                continue
            params[name] = value
            if isinstance(value, bool):
                if value:
                    suffix += f"_{abbr}"
            else:
                suffix += f"_{abbr}{value}"
        return (suffix, params)

    @staticmethod
    def compose_fields(
        fields: Tuple[tuple], names: tuple, abbrs: tuple
    ) -> Generator[Tuple[str, dict], None, None]:
        """
        Generate parameter combinations
        """
        for values in product(*fields):
            yield ModelGetter.compose_parameters(values, names=names, abbrs=abbrs)

    def _get(
        self,
        graph_factories: Dict[str, GraphFactory] = None,
        **scorer_params,
    ) -> List[Model]:
        raise NotImplementedError

    def get(
        self,
        graph_factory_params: dict,
        graph_factories: Dict[str, GraphFactory] = None,
        **scorer_params,
    ) -> Tuple[List[Model], Dict[str, GraphFactory]]:
        # pylint: disable=unused-argument
        """
        List models and used graph factories
        """
        if self._params is None:
            return [], {}
        return self._get(graph_factories=None, **scorer_params), {}


def _get_graph_factories(
    graph_params: utils.GraphParams, seed: int, structural_graph_params: dict = None
):
    graph_factories: Dict[str, GraphFactory] = {}

    if graph_params.pc_gauss:
        for suffix, params in ModelGetter.compose_fields(
            (graph_params.pc_gauss.alpha, graph_params.pc_gauss.max_conds_dim),
            names=("alpha", "max_conds_dim"),
            abbrs=("a", "m"),
        ):
            graph_factories["PC_gauss" + suffix] = PCAlgFactory(
                method="PC-gauss",
                num_cores=graph_params.pc_gauss.num_cores,
                seed=seed,
                **params,
            )
    if graph_params.pc_gsq:
        for suffix, params in ModelGetter.compose_fields(
            (graph_params.pc_gsq.alpha, graph_params.pc_gsq.max_conds_dim),
            names=("alpha", "max_conds_dim"),
            abbrs=("a", "m"),
        ):
            graph_factories["PC_gsq" + suffix] = PCAlgFactory(
                method="PC-gsq",
                num_cores=graph_params.pc_gsq.num_cores,
                seed=seed,
                **params,
            )
    if graph_params.pcts:
        for suffix, params in ModelGetter.compose_fields(
            (
                graph_params.pcts.alpha,
                graph_params.pcts.max_conds_dim,
                graph_params.pcts.tau_max,
            ),
            names=("alpha", "max_conds_dim", "tau_max"),
            abbrs=("a", "m", "t"),
        ):
            graph_factories["PCTS" + suffix] = PCTSFactory(seed=seed, **params)
    if graph_params.structural and structural_graph_params is not None:
        graph_factories["Structural"] = StructuralGraphFactory(
            **structural_graph_params, seed=seed
        )

    return graph_factories


def _get_detectors(ad_params: utils.ADParams, **scorer_params):
    """
    Map a detector name to a pair of Scorer and threshold
    """
    detectors: Dict[str, Tuple[Scorer, float]] = {}

    if ad_params.nsigma:
        detectors["NSigma"] = (NSigmaScorer(**scorer_params), 3)
    if ad_params.spot:
        for suffix, params in ModelGetter.compose_fields(
            (ad_params.spot.risk,), names=("proba",), abbrs=("p",)
        ):
            detectors["SPOT" + suffix] = (SPOTScorer(**params, **scorer_params), 0)

    return detectors


class NSigmaGetter(ModelGetter):
    """
    Get NSigma models
    """

    def __init__(self, params: utils.NSigmaParams):
        super().__init__(params=params)
        self._params = params

    def _get(
        self,
        graph_factories: Dict[str, GraphFactory] = None,
        **scorer_params,
    ) -> List[Model]:
        graph_factory = EmptyGraphFactory()
        return [
            Model(
                graph_factory=graph_factory,
                scorers=[NSigmaScorer(**scorer_params)],
                names=(EMPTY_GRAPH_NAME, "NSigma"),
            )
        ]


class SPOTGetter(ModelGetter):
    """
    Get SPOT models
    """

    def __init__(self, params: utils.SPOTParams):
        super().__init__(params=params)
        self._params = params

    def _get(
        self,
        graph_factories: Dict[str, GraphFactory] = None,
        **scorer_params,
    ) -> List[Model]:
        graph_factory = EmptyGraphFactory()
        return [
            Model(
                graph_factory=graph_factory,
                scorers=[SPOTScorer(**params, **scorer_params)],
                names=(EMPTY_GRAPH_NAME, "SPOT" + suffix),
            )
            for suffix, params in ModelGetter.compose_fields(
                (self._params.risk,), names=("proba",), abbrs=("p",)
            )
        ]


class ScorerModelGetter(ModelGetter):
    """
    Get graph-based models
    """

    def __init__(self, params: utils.ScorerParams):
        super().__init__(params=params)
        self._params = params

    def _compose_parameters(self, **scorer_params):
        raise NotImplementedError

    def _get_model_base(
        self,
        graph_factories: Dict[str, GraphFactory],
        **scorer_params,
    ):
        for suffix, params in self._compose_parameters(**scorer_params):
            for graph_name, graph_factory in graph_factories.items():
                yield graph_factory, graph_name, suffix, params

    def _get(
        self,
        graph_factories: Dict[str, GraphFactory] = None,
        **scorer_params,
    ) -> List[Model]:
        raise NotImplementedError

    def get(
        self,
        graph_factory_params: dict,
        graph_factories: Dict[str, GraphFactory] = None,
        **scorer_params,
    ) -> Tuple[List[Model], Dict[str, GraphFactory]]:
        if self._params is None:
            return [], {}
        if graph_factories is None:
            graph_factories = _get_graph_factories(
                graph_params=self._params.graph, **graph_factory_params
            )
        return (
            self._get(graph_factories=graph_factories, **scorer_params),
            graph_factories,
        )


class DFSGetter(ScorerModelGetter):
    """
    Get DFS models
    """

    def __init__(self, params: utils.DFSParams):
        super().__init__(params=params)
        self._params = params

    def _compose_parameters(self, **scorer_params):
        detectors = _get_detectors(ad_params=self._params.detector, **scorer_params)
        for detector_name, (detector, anomaly_threshold) in detectors.items():
            suffix, params = self.compose_parameters(
                values=(anomaly_threshold,), names=("anomaly_threshold",), abbrs=("a",)
            )
            params.update(scorer_params)
            yield detector, detector_name, suffix, params

    def _get_model_base(
        self,
        graph_factories: Dict[str, GraphFactory],
        **scorer_params,
    ):
        for detector, detector_name, suffix, params in self._compose_parameters(
            **scorer_params
        ):
            for graph_name, graph_factory in graph_factories.items():
                yield (
                    graph_factory,
                    detector,
                    [graph_name, detector_name],
                    suffix,
                    params,
                )

    def _get(
        self,
        graph_factories: Dict[str, GraphFactory] = None,
        **scorer_params,
    ) -> List[Model]:
        return [
            Model(
                graph_factory=graph_factory,
                scorers=[detector, DFSScorer(**params)],
                names=[*names, "DFS" + suffix],
            )
            for graph_factory, detector, names, suffix, params in self._get_model_base(
                graph_factories=graph_factories, **scorer_params
            )
        ]


class MicroscopeGetter(DFSGetter):
    """
    Get Microscope models
    """

    def _get(
        self,
        graph_factories: Dict[str, GraphFactory] = None,
        **scorer_params,
    ) -> List[Model]:
        return [
            Model(
                graph_factory=graph_factory,
                scorers=[
                    detector,
                    DFSScorer(**params),
                    CorrelationScorer(**scorer_params),
                ],
                names=[*names, "DFS" + suffix, "Pearson"],
            )
            for graph_factory, detector, names, suffix, params in self._get_model_base(
                graph_factories=graph_factories, **scorer_params
            )
        ]


class MicroHECLGetter(DFSGetter):
    """
    Get MicroHECL models
    """

    def __init__(self, params: utils.MicroHECLParams):
        super().__init__(params=params)
        self._params = params

    def _compose_parameters(self, **scorer_params):
        detectors = _get_detectors(ad_params=self._params.detector, **scorer_params)
        for detector_name, (detector, anomaly_threshold) in detectors.items():
            for stop_threshold in self._params.stop_threshold:
                suffix, params = self.compose_parameters(
                    values=(anomaly_threshold, stop_threshold),
                    names=("anomaly_threshold", "stop_threshold"),
                    abbrs=("a", "s"),
                )
                params.update(scorer_params)
                yield detector, detector_name, suffix, params

    def _get(
        self,
        graph_factories: Dict[str, GraphFactory] = None,
        **scorer_params,
    ) -> List[Model]:
        return [
            Model(
                graph_factory=graph_factory,
                scorers=[detector, MicroHECLScorer(**params)],
                names=[*names, "MicroHECL" + suffix],
            )
            for graph_factory, detector, names, suffix, params in self._get_model_base(
                graph_factories=graph_factories, **scorer_params
            )
        ]


class RWModelGetter(ScorerModelGetter):
    """
    Get random walk-based models
    """

    def __init__(self, params: utils.RandomWalkParams):
        super().__init__(params=params)
        self._params = params

    def _compose_parameters(self, **scorer_params):
        for suffix, params in self.compose_fields(
            (self._params.rho,), names=("rho",), abbrs=("r",)
        ):
            params.update(scorer_params)
            yield suffix, params


class MicroCauseGetter(RWModelGetter):
    """
    Get MicroCause models
    """

    def __init__(self, params: utils.MicroCauseParams):
        super().__init__(params=params)
        self._params = params

    def _get(
        self,
        graph_factories: Dict[str, GraphFactory] = None,
        **scorer_params,
    ) -> List[Model]:
        return [
            Model(
                graph_factory=graph_factory,
                scorers=[
                    PartialCorrelationScorer(**scorer_params),
                    RandomWalkScorer(**params),
                ],
                names=[graph_name, "PartialCorrelation", "RW" + suffix],
            )
            for graph_factory, graph_name, suffix, params in self._get_model_base(
                graph_factories=graph_factories, **scorer_params
            )
        ]


class CloudRangerGetter(RWModelGetter):
    """
    Get CloudRanger models
    """

    def __init__(self, params: utils.CloudRangerParams):
        super().__init__(params=params)
        self._params = params

    def _compose_parameters(self, **scorer_params):
        for suffix, params in self.compose_fields(
            (self._params.rho, self._params.beta),
            names=("rho", "beta"),
            abbrs=("r", "b"),
        ):
            params.update(scorer_params)
            yield suffix, params

    def _get(
        self,
        graph_factories: Dict[str, GraphFactory] = None,
        **scorer_params,
    ) -> List[Model]:
        return [
            Model(
                graph_factory=graph_factory,
                scorers=[
                    CorrelationScorer(**scorer_params),
                    SecondOrderRandomWalkScorer(**params),
                ],
                names=[
                    graph_name,
                    "Pearson",
                    "RW_2" + suffix,
                ],
            )
            for graph_factory, graph_name, suffix, params in self._get_model_base(
                graph_factories=graph_factories, **scorer_params
            )
        ]


class ENMFGetter(ModelGetter):
    """
    Get ENMF models
    """

    def __init__(self, params: utils.ENMFParams):
        super().__init__(params=params)
        self._params = params

    def _compose_parameters(self, **scorer_params):
        for suffix, params in self.compose_fields(
            (
                self._params.use_softmax,
                self._params.gamma,
                self._params.tau,
                self._params.discrete,
            ),
            names=("use_softmax", "gamma", "tau", "discrete"),
            abbrs=("soft", "c", "t", "d"),
        ):
            use_softmax: bool = params.pop("use_softmax")
            discrete: bool = params.pop("discrete")
            params = dict(
                model_params=params,
                use_softmax=use_softmax,
                discrete=discrete,
                **scorer_params,
            )
            yield suffix, params

    def _get(
        self,
        graph_factories: Dict[str, GraphFactory] = None,
        **scorer_params,
    ) -> List[Model]:
        return [
            Model(
                graph_factory=EmptyGraphFactory(),
                scorers=[ENMFScorer(**params)],
                names=(EMPTY_GRAPH_NAME, "ENMF" + suffix),
            )
            for suffix, params in self._compose_parameters(**scorer_params)
        ]


class CRDGetter(ModelGetter):
    """
    Get CRD models
    """

    def __init__(self, params: utils.CRDParams):
        super().__init__(params=params)
        self._params = params

    def _compose_parameters(self, **scorer_params):
        for suffix, params in self.compose_fields(
            (
                self._params.gamma,
                self._params.tau,
                self._params.discrete,
                self._params.num_cluster,
                self._params.alpha,
                self._params.beta,
                self._params.learning_rate,
            ),
            names=(
                "gamma",
                "tau",
                "discrete",
                "num_cluster",
                "alpha",
                "beta",
                "learning_rate",
            ),
            abbrs=("c", "t", "d", "nc", "a", "b", "lr"),
        ):
            discrete: bool = params.pop("discrete")
            params = dict(
                model_params=params,
                discrete=discrete,
                **scorer_params,
            )
            yield suffix, params

    def _get(
        self,
        graph_factories: Dict[str, GraphFactory] = None,
        **scorer_params,
    ) -> List[Model]:
        return [
            Model(
                graph_factory=EmptyGraphFactory(),
                scorers=[CRDScorer(**params)],
                names=(EMPTY_GRAPH_NAME, "CRD" + suffix),
            )
            for suffix, params in self._compose_parameters(**scorer_params)
        ]


class CIModelGetter(ScorerModelGetter):
    """
    Get causal inference-based models
    """

    def __init__(self, params: utils.CIParams):
        super().__init__(params=params)
        self._params = params

    def _compose_parameters(self, **scorer_params):
        for suffix, params in self.compose_fields(
            (self._params.tau_max,), names=("tau_max",), abbrs=("t",)
        ):
            params.update(scorer_params)
            for regressor in self._params.regressor:
                if regressor == "linear":
                    yield suffix, params
                elif regressor == "svr":
                    yield f"_svr{suffix}", dict(
                        regressor=ANMRegressor(regressor=SVR(kernel="sigmoid")),
                        **params,
                    )
                elif regressor == "rf":
                    yield f"_rf{suffix}", dict(regressor=GMMRegressor(), **params)
                elif regressor == "mdn":
                    yield f"_mdn{suffix}", dict(
                        regressor=GMMRegressor(regressor=MDNPredictor()), **params
                    )


class RHTGetter(CIModelGetter):
    """
    Get RHT models
    """

    def _get(
        self,
        graph_factories: Dict[str, GraphFactory] = None,
        **scorer_params,
    ) -> List[Model]:
        return [
            Model(
                graph_factory=graph_factory,
                scorers=[RHTScorer(**params)],
                names=(graph_name, "RHT" + suffix),
            )
            for graph_factory, graph_name, suffix, params in self._get_model_base(
                graph_factories=graph_factories, **scorer_params
            )
        ]


class RHTDAGetter(CIModelGetter):
    """
    Get RHT-DA models
    """

    def _get(
        self,
        graph_factories: Dict[str, GraphFactory] = None,
        **scorer_params,
    ) -> List[Model]:
        # The three-sigma rule of thumb, as use_confidence=False by default
        threshold = 3
        ranker = DAScorer(threshold=threshold)
        return [
            Model(
                graph_factory=graph_factory,
                scorers=[RHTScorer(**params), ranker],
                names=(graph_name, "RHT" + suffix, "DA"),
            )
            for graph_factory, graph_name, suffix, params in self._get_model_base(
                graph_factories=graph_factories, **scorer_params
            )
        ]


def get_models(
    structural_graph_params: dict = None,
    graph_factories: Dict[str, GraphFactory] = None,
    params: utils.ModelParams = None,
    seed: int = 0,
    **scorer_params,
) -> Tuple[List[Model], Dict[str, GraphFactory]]:
    """
    Prepare algorithm candidates

    Parameters:
        graph_factories: Specify GraphFactory for models
        params: Specify options for model parameters
    """
    if params is None:
        params = utils.ModelParams()

    graph_factory_params = dict(
        structural_graph_params=structural_graph_params, seed=seed
    )
    getter_params = dict(
        graph_factory_params=graph_factory_params,
        graph_factories=graph_factories,
        seed=seed,
        **scorer_params,
    )
    getters: List[ModelGetter] = [
        NSigmaGetter(params.nsigma),
        SPOTGetter(params.spot),
        DFSGetter(params.dfs),
        MicroscopeGetter(params.micro_scope),
        MicroHECLGetter(params.micro_hecl),
        MicroCauseGetter(params.micro_cause),
        CloudRangerGetter(params.cloud_ranger),
        ENMFGetter(params.enmf),
        CRDGetter(params.crd),
        RHTGetter(params.rht),
        RHTDAGetter(params.rht_da),
    ]

    models: List[Model] = []
    graph2cache: Dict[str, GraphFactory] = {}
    for getter in getters:
        submodels, sub_graph_factories = getter.get(**getter_params)
        models += submodels
        graph2cache.update(sub_graph_factories)

    return models, graph2cache
