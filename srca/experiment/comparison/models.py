"""
Utilities
"""
from abc import ABC
from itertools import product
from typing import Dict
from typing import List
from typing import Tuple

from . import utils
from ...alg.base import GraphFactory
from ...alg.base import Scorer
from ...alg.common import EmptyGraphFactory
from ...alg.common import Model
from ...alg.common import NSigmaScorer
from ...alg.correlation import CorrelationScorer
from ...alg.correlation import PartialCorrelationScorer
from ...alg.crd import CRDScorer
from ...alg.dfs import DFSScorer
from ...alg.dfs import MicroHECLScorer
from ...alg.evt import SPOTScorer
from ...alg.graph.pcts import PCTSFactory
from ...alg.graph.r import PCAlgFactory
from ...alg.random_walk import RandomWalkScorer
from ...alg.random_walk import SecondOrderRandomWalkScorer


EMPTY_GRAPH_NAME = "Empty"


class ModelGetter(ABC):
    """
    Abstract interface to get models
    """

    def get(
        self,
        graph_factory_params: dict,
        graph_factories: Dict[str, GraphFactory] = None,
        **scorer_params,
    ) -> Tuple[List[Model], Dict[str, GraphFactory]]:
        """
        List models and used graph factories
        """
        raise NotImplementedError


def _get_graph_factories(graph_params: utils.GraphParams, seed: int):
    pc_params: List[Tuple[str, dict]] = []
    for alpha, max_conds_dim in product(graph_params.alpha, graph_params.max_conds_dim):
        params = dict(alpha=alpha, seed=seed)
        suffix = f"_a{alpha}"
        if max_conds_dim is not None:
            params["max_conds_dim"] = max_conds_dim
            suffix += f"_m{max_conds_dim}"
        pc_params.append((suffix, params))
    graph_factories: Dict[str, GraphFactory] = {}

    if utils.GraphMethod.PC_GAUSS in graph_params.method:
        for suffix, params in pc_params:
            graph_factories["PC_gauss" + suffix] = PCAlgFactory(
                method="PC-gauss", num_cores=graph_params.num_cores, **params
            )
    if utils.GraphMethod.PC_GSQ in graph_params.method:
        for suffix, params in pc_params:
            graph_factories["PC_gsq" + suffix] = PCAlgFactory(
                method="PC-gsq", num_cores=graph_params.num_cores, **params
            )
    if utils.GraphMethod.PCTS in graph_params.method:
        for (suffix, params), tau_max in product(pc_params, graph_params.tau_max):
            graph_factories["PCTS" + suffix + f"_t{tau_max}"] = PCTSFactory(
                tau_max=tau_max, **params
            )

    return graph_factories


def _get_detectors(params: utils.ADParams, **scorer_params):
    """
    Map a detector name to a pair of Scorer and threshold
    """
    detectors: Dict[str, Tuple[Scorer, float]] = {}

    if utils.ADMethod.NSIGMA in params.method:
        detectors["NSigma"] = (NSigmaScorer(**scorer_params), 3)
    if utils.ADMethod.SPOT in params.method:
        for risk in params.risk:
            detectors[f"SPOT_p{risk}"] = (SPOTScorer(proba=risk, **scorer_params), 0)

    return detectors


class ADModelGetter(ModelGetter):
    """
    Get anomaly detection models
    """

    def __init__(self, params: utils.ADParams):
        self._params = params

    def get(
        self,
        graph_factory_params: dict,
        graph_factories: Dict[str, GraphFactory] = None,
        **scorer_params,
    ) -> Tuple[List[Model], Dict[str, GraphFactory]]:
        detectors = _get_detectors(params=self._params, **scorer_params)
        graph_factory = EmptyGraphFactory()
        return [
            Model(
                graph_factory=graph_factory,
                scorers=[detector],
                names=(EMPTY_GRAPH_NAME, name),
            )
            for name, (detector, _) in detectors.items()
        ], {}


class DFSModelGetter(ModelGetter):
    """
    Get dfs-based models
    """

    def __init__(self, params: utils.DFSParams):
        self._params = params

    def get(
        self,
        graph_factory_params: dict,
        graph_factories: Dict[str, GraphFactory] = None,
        **scorer_params,
    ) -> Tuple[List[Model], Dict[str, GraphFactory]]:
        # pylint: disable=too-many-locals
        if graph_factories is None:
            graph_factories = _get_graph_factories(
                graph_params=self._params.graph, **graph_factory_params
            )
        detectors = _get_detectors(params=self._params.detector, **scorer_params)
        model_base: List[Tuple[GraphFactory, Scorer, List[str], str, dict]] = []
        for detector_name, (detector, anomaly_threshold) in detectors.items():
            params = dict(anomaly_threshold=anomaly_threshold, **scorer_params)
            suffix = f"_a{anomaly_threshold}"
            for graph_name, graph_factory in graph_factories.items():
                model_base.append(
                    (
                        graph_factory,
                        detector,
                        [graph_name, detector_name],
                        suffix,
                        params,
                    )
                )
        models: List[Model] = []

        if utils.DFSMethod.DFS in self._params.method:
            for graph_factory, detector, names, suffix, params in model_base:
                models.append(
                    Model(
                        graph_factory=graph_factory,
                        scorers=[detector, DFSScorer(**params)],
                        names=[*names, "DFS" + suffix],
                    ),
                )
        if utils.DFSMethod.MICRO_SCOPE in self._params.method:
            for graph_factory, detector, names, suffix, params in model_base:
                models.append(
                    Model(
                        graph_factory=graph_factory,
                        scorers=[
                            detector,
                            DFSScorer(**params),
                            CorrelationScorer(**scorer_params),
                        ],
                        names=[*names, "DFS" + suffix, "Pearson"],
                    ),
                )
        if utils.DFSMethod.MICRO_HECL in self._params.method:
            for graph_factory, detector, names, suffix, params in model_base:
                for stop_threshold in self._params.stop_threshold:
                    scorer = MicroHECLScorer(stop_threshold=stop_threshold, **params)
                    models.append(
                        Model(
                            graph_factory=graph_factory,
                            scorers=[detector, scorer],
                            names=[*names, "MicroHECL" + f"{suffix}_s{stop_threshold}"],
                        ),
                    )

        return models, graph_factories


class RWModelGetter(ModelGetter):
    """
    Get random walk-based models
    """

    def __init__(self, params: utils.RandomWalkParams):
        self._params = params

    def get(
        self,
        graph_factory_params: dict,
        graph_factories: Dict[str, GraphFactory] = None,
        **scorer_params,
    ) -> Tuple[List[Model], Dict[str, GraphFactory]]:
        if graph_factories is None:
            graph_factories = _get_graph_factories(
                graph_params=self._params.graph, **graph_factory_params
            )
        model_base: List[Tuple[GraphFactory, str, str, dict]] = []
        for (
            rho,
            remove_sla,
        ) in product(self._params.rho, self._params.remove_sla):
            params = dict(rho=rho, remove_sla=remove_sla, **scorer_params)
            suffix = f"_r{rho}"
            if remove_sla:
                suffix += "_nosla"
            for graph_name, graph_factory in graph_factories.items():
                model_base.append((graph_factory, graph_name, suffix, params))
        models: List[Model] = []

        if utils.RandomWalkMethod.MICRO_CAUSE in self._params.method:
            for graph_factory, graph_name, suffix, params in model_base:
                models.append(
                    Model(
                        graph_factory=graph_factory,
                        scorers=[
                            PartialCorrelationScorer(**scorer_params),
                            RandomWalkScorer(**params),
                        ],
                        names=[graph_name, "PartialCorrelation", "RW" + suffix],
                    ),
                )
        if utils.RandomWalkMethod.CLOUD_RANGER in self._params.method:
            for graph_factory, graph_name, suffix, params in model_base:
                for beta in self._params.beta:
                    models.append(
                        Model(
                            graph_factory=graph_factory,
                            scorers=[
                                CorrelationScorer(**scorer_params),
                                SecondOrderRandomWalkScorer(beta=beta, **params),
                            ],
                            names=[
                                graph_name,
                                "Pearson",
                                "RW_2" + f"{suffix}_b{beta}",
                            ],
                        ),
                    )

        return models, graph_factories


class OtherModelGetter(ModelGetter):
    """
    Get other models
    """

    def __init__(self, params: utils.OtherParams):
        self._params = params

    def get(
        self,
        graph_factory_params: dict,
        graph_factories: Dict[str, GraphFactory] = None,
        **scorer_params,
    ) -> Tuple[List[Model], Dict[str, GraphFactory]]:
        models: List[Model] = []

        if utils.OtherMethod.CRD in self._params.method:
            models.append(
                Model(
                    graph_factory=EmptyGraphFactory(),
                    scorers=[CRDScorer(**scorer_params)],
                    names=(EMPTY_GRAPH_NAME, "CRD"),
                )
            )

        return models, {}


def get_models(
    graph_factories: Dict[str, GraphFactory] = None,
    params: utils.ModelParams = None,
    seed: int = 0,
) -> Tuple[List[Model], Dict[str, GraphFactory]]:
    """
    Prepare algorithm candidates

    Parameters:
        graph_factories: Specify GraphFactory for models
        params: Specify options for model parameters
    """
    if params is None:
        params = utils.ModelParams()

    graph_factory_params = dict(seed=seed)
    getter_params = dict(
        graph_factory_params=graph_factory_params,
        graph_factories=graph_factories,
        seed=seed,
    )
    getters: List[ModelGetter] = [
        ADModelGetter(params.anomaly_detection),
        DFSModelGetter(params.dfs),
        RWModelGetter(params.random_walk),
        OtherModelGetter(params.other),
    ]

    models: List[Model] = []
    graph2cache: Dict[str, GraphFactory] = {}
    for getter in getters:
        submodels, sub_graph_factories = getter.get(**getter_params)
        models += submodels
        graph2cache.update(sub_graph_factories)

    return models, graph2cache
