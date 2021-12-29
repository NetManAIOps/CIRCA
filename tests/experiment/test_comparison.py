"""
Test suites for the comparison experiment
"""
import logging
import os

import pytest

from srca.experiment.comparison import run
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
    report_filename = os.path.join(tempdir, "report.csv")
    models, graph_factories = get_models()
    cases = [Case(data=case_data, answer={case_data.sla})]
    delay = 60

    params = dict(
        models=models[:5],
        cases=cases,
        graph_factories=dict(list(graph_factories.items())[:5]),
        output_dir=tempdir,
        delay=delay,
        max_workers=max_workers,
        report_filename=report_filename,
    )
    logging.basicConfig(level=logging.WARNING, force=True)
    run(**params)
    assert "INFO:srca" not in capfd.readouterr().err

    logging.basicConfig(level=logging.INFO, force=True)
    run(**params)
    assert "INFO:srca" in capfd.readouterr().err
