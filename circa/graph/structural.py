"""
Structral graph
"""
from collections import Counter
from enum import Enum
from itertools import chain
from itertools import product
import os
from typing import Dict
from typing import List
from typing import Set
from typing import Tuple

import networkx as nx

from .base import GraphFactory
from ..model.case import CaseData
from ..model.graph import Graph
from ..model.graph import MemoryGraph
from ..model.graph import Node
from ..utils import load_csv


class MetricType(Enum):
    """
    Metric types
    """

    TRAFFIC = "T"
    SATURATION = "S"
    ERROR = "E"
    LATENCY = "L"


class Component:
    """
    Node in StructuralGraph
    """

    def __init__(self, root: str, name: str):
        self._name = name
        self._sequential = self.create_components(root, "sequential")
        if os.path.isfile(os.path.join(root, "parallel")):
            self._parallel = self.create_components(root, "parallel")
        else:
            sequential = {component.name for component in self._sequential}
            self._parallel: List["Component"] = []
            for filename in os.listdir(root):
                subdir = os.path.join(root, filename)
                if os.path.isdir(subdir) and filename not in sequential:
                    self._parallel.append(Component(subdir, filename))

        metrics: Dict[MetricType, Set[str]] = {kind: set() for kind in MetricType}
        metric_list = os.path.join(root, "metric.csv")
        if os.path.isfile(metric_list):
            for row in load_csv(metric_list):
                if len(row) != 2:
                    continue
                metric, kind = row
                try:
                    kind = MetricType(kind)
                    metrics[kind].add(metric)
                except ValueError:
                    pass
        self._metrics = metrics

        self._id = id(self)

    @property
    def identity(self) -> int:
        """
        Identity
        """
        return self._id

    @property
    def _traffic(self):
        return (self._id, MetricType.TRAFFIC)

    @property
    def _saturation(self):
        return (self._id, MetricType.SATURATION)

    @property
    def _latency(self):
        return (self._id, MetricType.LATENCY)

    @property
    def _error(self):
        return (self._id, MetricType.ERROR)

    @staticmethod
    def create_components(root: str, filename: str) -> List["Component"]:
        """
        Create components listed in the given file
        """
        filename_list = os.path.join(root, filename)
        components: List[Component] = []
        if os.path.isfile(filename_list):
            for (sub_component,) in load_csv(filename_list):
                subdir = os.path.join(root, sub_component)
                if os.path.isdir(subdir):
                    components.append(Component(subdir, sub_component))
        return components

    @property
    def name(self) -> str:
        """
        Component name
        """
        return self._name

    @property
    def sequential(self) -> List["Component"]:
        """
        Sequential components
        """
        return self._sequential

    @property
    def parallel(self) -> List["Component"]:
        """
        Parallel components
        """
        return self._parallel

    def __repr__(self) -> str:
        return f"Component(name='{self.name}')"

    def list_metrics(self) -> List[str]:
        """
        List metrics, including those of sub-components
        """
        metrics: List[str] = []
        for kind in MetricType:
            metrics += list(self._metrics[kind])
        for component in chain(self._sequential, self._parallel):
            metrics += component.list_metrics()
        return metrics

    def metrics(self, metric_type: MetricType, mask: Set[str] = None) -> Set[str]:
        """
        Fetch metrics
        """
        if mask is None:
            return self._metrics[metric_type]
        return self._metrics[metric_type] & mask

    def visit(
        self,
        graph: nx.DiGraph,
        components: Dict[int, "Component"],
        library: Dict[str, "Component"],
    ) -> Tuple[nx.DiGraph, Dict[int, "Component"]]:
        """
        Visit sub-components and record relations among MetricType
        """
        components[self._id] = self

        lib_component = library.get(self.name, None)
        if lib_component and self != lib_component:
            lib_id = lib_component.identity
            graph.add_edge(self._traffic, (lib_id, MetricType.TRAFFIC))
            graph.add_edge((lib_id, MetricType.LATENCY), self._latency)
            graph.add_edge((lib_id, MetricType.ERROR), self._error)
            return lib_component.visit(graph, components, library)

        # 1. Inter components
        for component in chain(self._parallel, self._sequential):
            graph.add_edge(self._traffic, (component.identity, MetricType.TRAFFIC))
            graph.add_edge((component.identity, MetricType.LATENCY), self._latency)
            graph.add_edge((component.identity, MetricType.ERROR), self._error)
            graph, components = component.visit(graph, components, library)

        # 1.1 Inter sequential components
        # WARNING: This part will introduce cycles
        # if self._sequential:
        #     pre_id = self._sequential[0].identity
        #     for component in self._sequential[1:]:
        #         next_id = component.identity
        #         graph.add_edge(
        #             (pre_id, MetricType.ERROR), (next_id, MetricType.TRAFFIC)
        #         )
        #         pre_id = next_id

        # 2. Intra component
        default = {
            self._traffic: {self._saturation, self._latency, self._error},
            self._saturation: {self._latency, self._error},
            self._latency: {self._error},
            # self._error: set(),
        }
        for parent, children in default.items():
            for child in children:
                graph.add_edge(parent, child)

        return graph, components


class StructuralGraph:
    """
    Graph derived from architecture
    """

    def __init__(self, root: str, name: str):
        self._name = name
        self._root = Component(root, name)
        metrics = self._root.list_metrics()

        self._components = {
            component.name: component
            for component in chain(self._root.sequential, self._root.parallel)
        }

        for filename in os.listdir(root):
            subdir = os.path.join(root, filename)
            if filename not in self._components and os.path.isdir(subdir):
                component = Component(subdir, filename)
                self._components[filename] = component
                metrics += component.list_metrics()

        self._metrics = Counter(metrics)

    def visit(self, mask: Set[str] = None) -> nx.DiGraph:
        """
        Generate the graph among metrics
        """
        skeleton, components = self._root.visit(
            graph=nx.DiGraph(), components={}, library=self._components
        )

        metric_counter = {
            metric: count
            for metric, count in self._metrics.items()
            if mask is None or metric in mask
        }
        metric_node: Dict[str, Set[Tuple[int, MetricType]]] = {
            metric: set() for metric in metric_counter
        }
        node_metrics: Dict[Tuple[int, MetricType], Set[str]] = {}
        graph = nx.DiGraph()
        for node in nx.topological_sort(skeleton):
            component_id, metric_type = node
            component = components[component_id]
            parent_metric: Set[str] = set()
            node_metric: Set[str] = set()

            # 1. Gather metrics in a node
            # 1.1 Handle metrics in multiple nodes
            for metric in component.metrics(metric_type, mask):
                if metric_counter[metric] > 1:
                    metric_counter[metric] -= 1
                    metric_node[metric].add(node)
                elif self._metrics[metric] > 1:
                    parent_metric.add(metric)
                    graph.add_edges_from(
                        product(
                            chain(
                                *[node_metrics[node] for node in metric_node[metric]]
                            ),
                            {metric},
                        )
                    )
                else:
                    node_metric.add(metric)

            if metric_type in {MetricType.LATENCY, MetricType.ERROR}:
                # 1.2 Gather metrics from reference children
                for ch_component_id, ch_metric_type in skeleton.successors(node):
                    ch_component = components[ch_component_id]
                    if (
                        ch_component.name == component.name
                        and metric_type == ch_metric_type
                    ):
                        # The child is a reference of component
                        assert list(
                            skeleton.predecessors((ch_component_id, ch_metric_type))
                        ) == [node]
                        node_metric.update(ch_component.metrics(metric_type, mask))

            # 2. Gather metrics of parents
            parents: List[Tuple[int, MetricType]] = list(skeleton.predecessors(node))

            for pa_component_id, pa_metric_type in parents:
                parent = components[pa_component_id]
                is_reference = (
                    parent.name == component.name and pa_metric_type == metric_type
                )
                if (
                    is_reference
                    and metric_type == MetricType.TRAFFIC
                    and parent.metrics(metric_type, mask)
                ):
                    # 2.1 Gather metrics from refered parents
                    assert list(
                        skeleton.successors((pa_component_id, pa_metric_type))
                    ) == [node]
                    node_metric.update(parent.metrics(metric_type, mask))
                elif is_reference and metric_type in {
                    MetricType.LATENCY,
                    MetricType.ERROR,
                }:
                    # 2.2 Override metrics of the only refered parent
                    assert len(parents) == 1
                    if not node_metric:
                        node_metric.update(
                            node_metrics[(pa_component_id, pa_metric_type)]
                        )
                else:
                    parent_metric |= node_metrics[(pa_component_id, pa_metric_type)]

            # 3. Link
            graph.add_edges_from(product(parent_metric, node_metric))

            # 4. Bypass ERROR
            if metric_type == MetricType.ERROR:
                for pa_component_id, pa_metric_type in parents:
                    if pa_metric_type == metric_type:
                        node_metric.update(
                            node_metrics[(pa_component_id, pa_metric_type)]
                        )

            node_metrics[node] = node_metric if node_metric else parent_metric

        return graph


class StructrualGraphFactory(GraphFactory):
    """
    Create Graph instances based on StructuralGraph
    """

    def __init__(self, structural_graph: StructuralGraph, entity: str = "DB", **kwargs):
        super().__init__(**kwargs)
        self._structural_graph = structural_graph
        self._entity = entity

    def create(self, data: CaseData, current: float) -> Graph:
        graph = self._structural_graph.visit(
            set(data.data_loader.metrics[self._entity])
        )
        return MemoryGraph(
            nx.DiGraph(
                {
                    Node(self._entity, reason): [
                        Node(self._entity, result)
                        for result in graph.successors(reason)
                    ]
                    for reason in graph.nodes
                }
            )
        )
