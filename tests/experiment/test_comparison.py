"""
Test suites for the comparison experiment
"""
import logging
import os

import pytest

from srca.experiment.comparison import run
from srca.experiment.comparison.models import ModelGetter
from srca.experiment.comparison.models import get_models
from srca.model.case import Case
from srca.model.case import CaseData


@pytest.mark.parametrize(("max_workers",), [(1,), (2,)])
def test_logging(
    max_workers: int, case_data: CaseData, tempdir: str, capfd: pytest.CaptureFixture
):
    """
    The forked process shall have the same logging level as the main one
    """
    models, graph_factories = get_models()
    cases = [Case(data=case_data, answer={case_data.sli})]
    delay = 60

    params = dict(
        models=models[:5],
        cases=cases,
        graph_factories=dict(list(graph_factories.items())[:5]),
        output_dir=tempdir,
        delay=delay,
        max_workers=max_workers,
    )
    logging.basicConfig(level=logging.WARNING, force=True)
    run(report_filename=os.path.join(tempdir, "report-warn.csv"), **params)
    assert "INFO:srca" not in capfd.readouterr().err

    logging.basicConfig(level=logging.INFO, force=True)
    report_filename = os.path.join(tempdir, "report-info.csv")
    run(report_filename=report_filename, **params)
    assert "INFO:srca" in capfd.readouterr().err
    run(report_filename=report_filename, **params)
    assert "INFO:srca" not in capfd.readouterr().err, "Historical report is lost"


def test_compose_parameters():
    """
    ModelGetter.compose_parameters shall generate a tuple of suffix and named parameters
    """
    suffix, params = ModelGetter.compose_parameters(
        values=[None, False, True, 3],
        names=["1", "2", "3", "4"],
        abbrs=["a", "b", "c", "d"],
    )
    assert suffix == "_c_d3"
    assert params == {"2": False, "3": True, "4": 3}
