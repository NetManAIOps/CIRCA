"""
Structral graph
"""
import dataclasses
from itertools import chain
from itertools import product
import logging
from typing import Dict
from typing import List
from typing import Set

import networkx as nx

from .base import GraphFactory
from ..model.case import CaseData
from ..model.graph import Graph
from ..model.graph import MemoryGraph
from ..model.graph import Node
from ..utils import YamlComposeLoader
from ..utils import topological_sort


@dataclasses.dataclass
class MetaVariable:
    """
    MetaVariable includes the following information
    - type: The meta variable type
    - component: Name of the component
    """

    type: str = dataclasses.field(default_factory=str)
    component: str = dataclasses.field(default_factory=str)

    def __hash__(self) -> int:
        return hash((self.type, self.component))


@dataclasses.dataclass
class ComponentConfig:
    """
    ComponentConfig includes the following information
    - name: Name of the component
    - dependencies: A mapping from one caller to its callees, and
    - mappings: A mapping from one variable to meta variables
    """

    name: str = dataclasses.field(default_factory=str)
    dependencies: Dict[str, Set[str]] = dataclasses.field(default_factory=dict)
    mappings: Dict[str, List[MetaVariable]] = dataclasses.field(default_factory=dict)

    def __init__(self, component: dict, types: set):
        self.name = component["name"]
        dependencies: dict = component.get("dependencies", {})
        self.dependencies = {
            caller: set(callees) for caller, callees in dependencies.items()
        }
        mappings: dict = component.get("mappings", {})
        self.mappings = self.parse_mappings(mappings, types)

    @staticmethod
    def parse_mappings(
        mappings: Dict[str, List[dict]], types: set
    ) -> Dict[str, List[MetaVariable]]:
        """
        Check whether a meta variable is valid
        """
        invalid_meta_variables = []
        ret: dict = {}
        for variable, meta_variables in mappings.items():
            mapped_meta_variables = []
            for meta_variable in meta_variables:
                mv_type = meta_variable.get("type", None)
                if mv_type not in types or "component" not in meta_variable:
                    invalid_meta_variables.append(meta_variable)
                    continue
                mapped_meta_variables.append(
                    MetaVariable(type=mv_type, component=meta_variable["component"])
                )
            ret[variable] = mapped_meta_variables

        if invalid_meta_variables:
            logger = logging.getLogger(__name__)
            logger.warning(
                "There are unknown meta variables in the metric mapping: %s",
                invalid_meta_variables,
            )
        return ret


@dataclasses.dataclass
class Config:
    """
    Config includes the following information to construct the structural graph
    - causal assumptions,
    - the component call graph, and
    - the mapping between variables and meta variables
    """

    types: Set[str] = dataclasses.field(default_factory=set)

    assumed_graph: Dict[str, Set[str]] = dataclasses.field(default_factory=dict)
    assumed_parents: Dict[str, Set[str]] = dataclasses.field(default_factory=dict)
    assumed_children: Dict[str, Set[str]] = dataclasses.field(default_factory=dict)
    assumed_ancestors: Dict[str, Set[str]] = dataclasses.field(default_factory=dict)
    assumed_descendents: Dict[str, Set[str]] = dataclasses.field(default_factory=dict)

    components: List[ComponentConfig] = dataclasses.field(default_factory=list)

    def __init__(self, filename: str):
        config = YamlComposeLoader.load(filename)
        self.types = set(config.get("types", []))
        assumptions: dict = config.get("assumptions", {})
        for item in ["graph", "parents", "children", "ancestors", "descendents"]:
            setattr(
                self,
                f"assumed_{item}",
                self.parse_assumptions(assumptions.get(item, {}), self.types),
            )
        components: list = config.get("components", [])
        self.components = [
            ComponentConfig(component, self.types) for component in components
        ]

    @staticmethod
    def parse_assumptions(assumptions: dict, types: set) -> Dict[str, set]:
        """
        A set of assumptions is a mapping from one meta variable type to its effects
        """
        logger = logging.getLogger(__name__)
        extra_causes = []
        extra_effects = {}
        ret = {mv_type: set() for mv_type in types}
        for cause, effects in assumptions.items():
            if cause not in types:
                extra_causes.append(cause)
                continue
            effects = set(effects)
            extra = effects - types
            if extra:
                extra_effects[cause] = extra
            ret[cause] = effects & types

        if extra_causes:
            logger.warning(
                "There are causes with unknown meta variables types"
                " in the assumptions: %s",
                extra_causes,
            )
        if extra_effects:
            logger.warning(
                "There are effects with unknown meta variables types"
                " in the assumptions: %s",
                extra_effects,
            )
        return ret

    def call_graph(self) -> nx.DiGraph:
        """
        Summary the dependencies of all components as the call graph
        """
        graph = nx.DiGraph()
        for component in self.components:
            graph.add_node(component.name)
            for caller, callees in component.dependencies.items():
                graph.add_edges_from(product([caller], callees))
        return graph


class StructuralGraph:
    """
    Structural graph derived from architecture
    """

    def __init__(self, filename: str = None, config: Config = None):
        """
        config: The necessary information to construct the structural graph
        filename: Mandatory if config is missing
        """
        if config is None:
            config = Config(filename)
        self._config = config
        self._skeleton = self.create_skeleton(config)

    @staticmethod
    def create_skeleton(config: Config) -> nx.DiGraph:
        """
        Create skeleton among meta variables
        """
        # 1. Collect the call graph
        call_graph = config.call_graph()

        skeleton = nx.DiGraph()
        # 2. Instantiate meta variables
        for mv_type in config.types:
            for component in call_graph.nodes:
                skeleton.add_node(MetaVariable(type=mv_type, component=component))

        # 3. Add edges based on causal assumptions
        for callee in call_graph.nodes:
            # 3.1 Within a component
            for cause_mv_type, effect_mv_types in config.assumed_graph.items():
                for effect_mv_type in effect_mv_types:
                    skeleton.add_edge(
                        MetaVariable(type=cause_mv_type, component=callee),
                        MetaVariable(type=effect_mv_type, component=callee),
                    )
            for callee_mv_type in config.types:
                # 3.2 For one-hop relations
                for caller in call_graph.predecessors(callee):
                    for caller_mv_type in config.assumed_parents[callee_mv_type]:
                        skeleton.add_edge(
                            MetaVariable(type=caller_mv_type, component=caller),
                            MetaVariable(type=callee_mv_type, component=callee),
                        )
                    for caller_mv_type in config.assumed_children[callee_mv_type]:
                        skeleton.add_edge(
                            MetaVariable(type=callee_mv_type, component=callee),
                            MetaVariable(type=caller_mv_type, component=caller),
                        )
                # 3.3 For multi-hop relations
                for caller in nx.ancestors(call_graph, callee):
                    for caller_mv_type in config.assumed_ancestors[callee_mv_type]:
                        skeleton.add_edge(
                            MetaVariable(type=caller_mv_type, component=caller),
                            MetaVariable(type=callee_mv_type, component=callee),
                        )
                    for caller_mv_type in config.assumed_descendents[callee_mv_type]:
                        skeleton.add_edge(
                            MetaVariable(type=callee_mv_type, component=callee),
                            MetaVariable(type=caller_mv_type, component=caller),
                        )

        return skeleton

    def _map_variable_meta(self, mask: Dict[str, Set[str]] = None):
        if mask is None:
            mask = {}

        variable2meta: Dict[Node, List[MetaVariable]] = {}
        meta2variable: Dict[MetaVariable, List[Node]] = {
            meta_variable: [] for meta_variable in self._skeleton.nodes
        }
        for component in self._config.components:
            for variable_name, meta_variables in component.mappings.items():
                variable = Node(entity=component.name, metric=variable_name)
                if component.name not in mask or variable_name in mask[component.name]:
                    variable2meta[variable] = meta_variables
                    for meta_variable in meta_variables:
                        meta2variable[meta_variable].append(variable)
        return variable2meta, meta2variable

    def visit(self, mask: Dict[str, Set[str]] = None) -> nx.DiGraph:
        """
        Generate the graph among the (component, variable) tuples
        """
        # 1. Set up mappings between variables and meta variables with the mask
        variable2meta, meta2variable = self._map_variable_meta(mask)

        graph = nx.DiGraph()
        visible_variables: Dict[MetaVariable, Set[Node]] = {
            meta_variable: set() for meta_variable in meta2variable
        }
        counter = {
            variable: len(meta_variables)
            for variable, meta_variables in variable2meta.items()
        }
        variable2visited: Dict[Node, Set[MetaVariable]] = {
            variable: set() for variable in counter
        }
        # 2. Iterate over meta variables in the topological order
        meta_variables: List[Set[MetaVariable]] = topological_sort(
            nodes=self._skeleton.nodes,
            predecessors=self._skeleton.predecessors,
            successors=self._skeleton.successors,
        )
        for meta_variable in chain(*meta_variables):
            current: Set[Node] = set()
            parents: Set[Node] = set()

            # 2.1 Handle multi-mapping variables
            for variable in meta2variable[meta_variable]:
                if counter[variable] == 1:
                    current.add(variable)
                elif len(variable2visited[variable]) == counter[variable] - 1:
                    # This is the last time to vist this variable
                    parents.add(variable)
                    for visited_mv in variable2visited[variable]:
                        graph.add_edges_from(
                            product(visible_variables[visited_mv], [variable])
                        )
                else:
                    # Skip to avoid self-loop
                    variable2visited[variable].add(meta_variable)

            # 2.2 Collect parents
            for cause_meta_variable in self._skeleton.predecessors(meta_variable):
                parents |= visible_variables[cause_meta_variable]

            # 2.3 Link from parents to current
            graph.add_edges_from(product(parents, current))

            # 2.4 Set visible variables for the following meta variables
            visible_variables[meta_variable] = current if current else parents

        return graph


class StructuralGraphFactory(GraphFactory):
    """
    Create Graph instances based on StructuralGraph
    """

    def __init__(self, structural_graph: StructuralGraph, **kwargs):
        super().__init__(**kwargs)
        self._structural_graph = structural_graph

    def create(self, data: CaseData, current: float) -> Graph:
        graph = self._structural_graph.visit(
            {
                entity: set(metric_names)
                for entity, metric_names in data.data_loader.metrics.items()
            }
        )
        return MemoryGraph(graph)
