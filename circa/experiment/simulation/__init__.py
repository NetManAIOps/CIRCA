"""
Simluation with vector auto-regression model
"""
import datetime
import os
import pickle
from typing import Callable
from typing import List
from typing import Set

import networkx as nx
import numpy as np
from sklearn.preprocessing import StandardScaler

from ...graph.common import StaticGraphFactory
from ...model.case import Case
from ...model.case import CaseData
from ...model.data_loader import MemoryDataLoader
from ...model.graph import MemoryGraph
from ...model.graph import Node
from ...utils import ENCODING
from ...utils import dump_csv
from ...utils import dump_json
from ...utils import load_csv
from ...utils import load_json


_SLI = 0
ENTITY = "SIM"


class SimCase(Case):
    """
    Simulated case data for evaluation
    """

    DATA_FILENAME = "data.csv"
    INFO_FILENAME = "info.json"
    DETAIL_FILENAME = "details.pk"

    def __init__(
        self,
        data: np.ndarray,
        causes: Set[int],
        length_normal: int,
        interval: datetime.timedelta = datetime.timedelta(minutes=1),
        case_data_params: dict = None,
        details: dict = None,
    ):
        # pylint: disable=too-many-arguments
        super().__init__(
            data=None,
            answer={Node(entity=ENTITY, metric=str(cause)) for cause in causes},
        )
        self._time_series = data
        self._causes = causes
        self._length_normal = length_normal
        self._interval = interval
        self._case_data_params = {} if case_data_params is None else case_data_params
        self._details = details

    @property
    def data(self) -> CaseData:
        if self._data is not None:
            return self._data

        _, num_node = self._time_series.shape
        interval = self._interval.total_seconds()
        data = {
            ENTITY: {
                str(node): [
                    (i * interval, value)
                    for i, value in enumerate(self._time_series[:, node])
                ]
                for node in range(num_node)
            }
        }
        self._data = CaseData(
            data_loader=MemoryDataLoader(data=data),
            sli=Node(entity=ENTITY, metric=str(_SLI)),
            detect_time=self._length_normal * interval,
            interval=self._interval,
            **self._case_data_params,
        )
        return self._data

    @classmethod
    def load(cls, folder: str, **kwargs) -> "SimCase":
        """
        Load from a folder

        Parameters:
            folder: where the case is dumped
            **kwargs: Other parameters will be passed to construct SimCase
        """
        filename = os.path.join(folder, cls.DATA_FILENAME)
        data = np.array(list(load_csv(filename))).astype(float)
        info: dict = load_json(os.path.join(folder, cls.INFO_FILENAME))
        return SimCase(
            data=data,
            causes=info["causes"],
            length_normal=info["length_normal"],
            **kwargs,
        )

    def dump(self, folder: str):
        """
        Dump into a folder
        """
        # Dump data into csv
        os.makedirs(folder, exist_ok=True)
        dump_csv(
            filename=os.path.join(folder, self.DATA_FILENAME), data=self._time_series
        )
        dump_json(
            filename=os.path.join(folder, self.INFO_FILENAME),
            data=dict(causes=list(self._causes), length_normal=self._length_normal),
        )
        if self._details:
            with open(os.path.join(folder, self.DETAIL_FILENAME), "wb") as obj:
                pickle.dump(self._details, obj)


class SimDataset:
    """
    A combination of graph and simulated cases
    """

    _INDEX_FILENAME = "index"
    GRAPH_FILENAME = "graph.json"
    CASES_FOLDER = "cases"

    def __init__(self, graph: MemoryGraph, cases: List[SimCase]):
        self._cases = cases
        self._graph = graph
        self._graph_factory = StaticGraphFactory(graph=graph)

    @property
    def cases(self) -> List[Case]:
        """
        Simulated cases
        """
        return self._cases

    @property
    def graph_factory(self) -> StaticGraphFactory:
        """
        The causal graph in the data generation process
        """
        return self._graph_factory

    @classmethod
    def load(cls, folder: str, **kwargs) -> "SimDataset":
        """
        Load from a folder

        Parameters:
            folder: where the case is dumped
            **kwargs: Other parameters will be passed to construct SimCase
        """
        cases_folder = os.path.join(folder, cls.CASES_FOLDER)
        graph = MemoryGraph.load(os.path.join(folder, cls.GRAPH_FILENAME))
        with open(
            os.path.join(cases_folder, cls._INDEX_FILENAME), encoding=ENCODING
        ) as obj:
            num_cases = int(next(obj))
        cases = [
            SimCase.load(os.path.join(cases_folder, str(index)), **kwargs)
            for index in range(num_cases)
        ]
        return SimDataset(graph=graph, cases=cases)

    def dump(self, folder: str):
        """
        Dump into a folder
        """
        # Dump data into csv
        cases_folder = os.path.join(folder, self.CASES_FOLDER)
        os.makedirs(cases_folder, exist_ok=True)
        self._graph.dump(os.path.join(folder, self.GRAPH_FILENAME))

        with open(
            os.path.join(cases_folder, self._INDEX_FILENAME), "w", encoding=ENCODING
        ) as obj:
            obj.write(str(len(self._cases)))
        for index, case in enumerate(self._cases):
            case.dump(os.path.join(cases_folder, str(index)))


def _normal_weight(rng: np.random.Generator) -> float:
    weight = rng.standard_normal()
    return np.sign(weight) * (abs(weight) + 0.2)


def _uniform_weight(rng: np.random.Generator) -> float:
    segments = [(-2.0, -0.5), (0.5, 2.0)]
    low, high = rng.choice(segments)
    return rng.uniform(low=low, high=high)


def generate_sedag(
    num_node: int,
    num_edge: int,
    weight_generator: Callable[[np.random.Generator], float] = _uniform_weight,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """
    Generate a weighted directed acyclic graph with a single end.

    The first node with index of 0 is the only end that does not have results.

    Returns:
        a matrix, where matrix[i, j] != 0 means j is the cause of i
    """
    num_edge = min(max(num_edge, num_node - 1), int(num_node * (num_node - 1) / 2))
    if rng is None:
        rng = np.random.default_rng()
    matrix = np.zeros((num_node, num_node))
    # Make the graph connected
    for cause in range(1, num_node):
        result = rng.integers(low=0, high=cause)
        matrix[result, cause] = weight_generator(rng)
    num_edge -= num_node - 1
    while num_edge > 0:
        cause = rng.integers(low=1, high=num_node)
        result = rng.integers(low=0, high=cause)
        if not matrix[result, cause]:
            matrix[result, cause] = weight_generator(rng)
            num_edge -= 1
    return matrix


def generate_case(
    weight: np.ndarray,
    length_normal: int = 1440,
    fault_duration: int = 2,
    length_abnormal: int = 10,
    beta: float = 1e-1,
    tau: float = 3,
    sigmas: np.ndarray = None,
    fault: np.ndarray = None,
    rng: np.random.Generator = None,
) -> SimCase:
    # pylint: disable=too-many-arguments, too-many-locals
    """
    Generate a case

    Parameters:
        weight: The inversed matrix of I - A
    """
    if rng is None:
        rng = np.random.default_rng()
    length_abnormal = max(length_abnormal, fault_duration)

    num_node, _ = weight.shape
    data: np.ndarray = np.zeros((0, num_node))
    if sigmas is None:
        sigmas = rng.standard_exponential(num_node)

    values = rng.standard_normal(num_node) * sigmas

    # Generate a series of x with x^{(t)} = A x^{(t)} + x^{(t - 1)} + epsilon^{(t)}
    # or x^{(t)} = A^{(\prime)} (x^{(t - 1)} + epsilon^{(t)})
    # where A^{(\prime)} = \sum_{i=0}^{num_node} A^{i}
    for _ in range(length_normal):
        values = weight @ (beta * values + rng.standard_normal(num_node) * sigmas)
        data = np.append(data, [values], axis=0)

    sli_mean: float = data[:, _SLI].mean()
    sli_sigma: float = data[:, _SLI].std()
    # Inject a fault
    if fault is None:
        num_causes = min(rng.poisson(1) + 1, num_node)
        causes = rng.choice(num_node, size=num_causes, replace=False)
        fault = np.zeros(num_node)
        alpha = rng.standard_exponential(size=num_causes)
        epsilon = rng.standard_normal(num_node)
        while True:
            fault[causes] = alpha
            sli_value: float = np.dot(
                weight[_SLI, :], beta * values + (epsilon + fault) * sigmas
            )
            if abs(sli_value - sli_mean) > tau * sli_sigma:
                break
            alpha *= 2
    else:
        causes: np.ndarray = np.where(fault)[0]
        assert causes.size

    # Faulty data
    for _ in range(fault_duration):
        values = weight @ (
            beta * values + (rng.standard_normal(num_node) + fault) * sigmas
        )
        data = np.append(data, [values], axis=0)
    for _ in range(length_abnormal - fault_duration):
        values = weight @ (beta * values + rng.standard_normal(num_node) * sigmas)
        data = np.append(data, [values], axis=0)

    scaler = StandardScaler().fit(data[:length_normal, :])
    data = np.around(scaler.transform(data), decimals=3)

    details = dict(
        fault=fault,
        sigmas=sigmas,
        stds=scaler.scale_,
        weight=weight,
    )
    return SimCase(
        data=data,
        causes=set(causes.tolist()),
        length_normal=length_normal,
        details=details,
    )


def generate(
    num_node: int, num_edge: int, num_cases: int = 100, rng: np.random.Generator = None
) -> SimDataset:
    """
    Generate a dataset with the same graph and serveral cases
    """
    if rng is None:
        rng = np.random.default_rng()

    # A = Generate weighted DAG
    matrix = generate_sedag(num_node=num_node, num_edge=num_edge, rng=rng)
    prod = np.eye(num_node)
    weight = np.eye(num_node)  # The reversed matrix of I - A
    for _ in range(1, num_node):
        prod = prod @ matrix
        weight += prod

    cases = [generate_case(weight=weight, rng=rng) for _ in range(num_cases)]
    graph = nx.DiGraph(
        (
            (
                Node(entity=ENTITY, metric=str(cause)),
                Node(entity=ENTITY, metric=str(result)),
            )
            for result, cause in zip(*np.where(matrix))
        )
    )
    return SimDataset(cases=cases, graph=MemoryGraph(graph))
