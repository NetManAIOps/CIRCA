"""
Smoke tests for algorithms
"""
import pytest

from srca.alg.base import Ranker
from srca.alg.base import Scorer
from srca.alg.common import NSigmaScorer
from srca.alg.common import ScoreRanker
from srca.alg.common import analyze
from srca.model.case import CaseData


@pytest.mark.parametrize(
    ("scorer", "ranker"),
    [
        (NSigmaScorer(lookup_window=4, detect_window=2), ScoreRanker()),
    ],
)
def test_smoke(scorer: Scorer, ranker: Ranker, case_data: CaseData):
    """
    Smoke tests
    """
    assert analyze(
        scorer=scorer, ranker=ranker, data=case_data, current=case_data.detect_time + 60
    )
