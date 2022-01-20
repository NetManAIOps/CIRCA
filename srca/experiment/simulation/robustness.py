"""
Robustness evaluation
"""
from enum import Enum
import logging
import os
import pickle
from typing import Dict
from typing import List

from . import _SLI
from . import SimCase
from . import SimDataset
from ..comparison import IDEAL_METHOD
from ..comparison import report_evaluation
from ...alg.common import Evaluation
from ...model.case import Case
from ...model.graph import Node
from ...utils import dump_csv
from ...utils import load_json


class Intensity(Enum):
    """
    Strong dependency intensity leads to unobvious root cause,
    dealing with which shows strong robustness.
    """

    WEAK = "weak"
    STRONG = "strong"
    MIXED = "mixed"

    @staticmethod
    def classify(data_dir: str) -> "Intensity":
        """
        Classify a case
        """
        detail_filename = os.path.join(data_dir, SimCase.DETAIL_FILENAME)
        with open(detail_filename, "rb") as obj:
            data = pickle.load(obj)
        ratio = abs(data["stds"] * data["weight"][_SLI, :] / data["stds"][_SLI])
        root_cause = data["fault"] != 0
        if not any((ratio <= 1) & root_cause):
            return Intensity.STRONG
        if any((ratio > 1) & root_cause):
            return Intensity.MIXED
        return Intensity.WEAK


def _evaluate_graph(
    cases: List[Case],
    reports: Dict[Intensity, Dict[str, Evaluation]],
    cache_dir: str,
    data_dir: str,
):
    """
    Evaluate for a single graph
    """
    answer_groups: Dict[str, List[Case]] = {intensity: [] for intensity in Intensity}
    index_groups: Dict[Intensity, List[int]] = {
        intensity: [] for intensity in Intensity
    }
    for i, case in enumerate(cases):
        intensity = Intensity.classify(
            data_dir=os.path.join(data_dir, SimDataset.CASES_FOLDER, str(i))
        )
        answer_groups[intensity].append(case.answer)
        index_groups[intensity].append(i)

    for filename in os.listdir(cache_dir):
        if not filename.endswith(".json"):
            continue
        method = filename[:-5]
        ranks = [
            [Node(entity=node["entity"], metric=node["metric"]) for node in rank]
            for rank in load_json(os.path.join(cache_dir, filename))
        ]
        for intensity, answers in answer_groups.items():
            if method not in reports[intensity]:
                reports[intensity][method] = Evaluation()
            for i, answer in zip(index_groups[intensity], answers):
                reports[intensity][method](ranks=ranks[i], answers=answer)

    for intensity, answers in answer_groups.items():
        if IDEAL_METHOD not in reports[intensity]:
            reports[intensity][IDEAL_METHOD] = Evaluation()
        for answer in answers:
            reports[intensity][IDEAL_METHOD](ranks=list(answer), answers=answer)
    return reports


def evaluate(num_graph: int, cache_dir: str, data_dir: str, report_dir: str):
    """
    Classify cases based on dependency intensities and evaluate separately
    """
    os.makedirs(report_dir, exist_ok=True)
    reports: Dict[Intensity, Dict[str, Evaluation]] = {
        intensity: {} for intensity in Intensity
    }

    logger = logging.getLogger(evaluate.__module__)
    for i in range(num_graph):
        cases_dir = os.path.join(data_dir, str(i))
        logger.info("Loading from %s", cases_dir)
        cases = SimDataset.load(folder=cases_dir).cases
        reports = _evaluate_graph(
            cases=cases,
            reports=reports,
            cache_dir=os.path.join(cache_dir, str(i)),
            data_dir=cases_dir,
        )

    for intensity, sub_reports in reports.items():
        dump_csv(
            filename=os.path.join(report_dir, f"{intensity.value}.csv"),
            data=[
                report_evaluation(method, report)
                for method, report in sub_reports.items()
            ],
            headers=["method", "AC@1", "AC@3", "AC@5", "Avg@5"],
        )
    dump_csv(
        filename=os.path.join(report_dir, "count.csv"),
        data=[
            [intensity.value, list(sub_reports.values())[0].num]
            for intensity, sub_reports in reports.items()
        ],
        headers=["intensity", "count"],
    )
