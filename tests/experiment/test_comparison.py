"""
Test suites for the comparison experiment
"""
import logging
import os

import pytest

from srca.experiment import comparison
from srca.model.case import Case
from srca.model.case import CaseData


def test_logging(case_data: CaseData, tempdir: str, capfd: pytest.CaptureFixture):
    """
    The forked process shall have the same logging level as the main one
    """
    report_filename = os.path.join(tempdir, "report.csv")
    models, _ = comparison.get_models()
    cases = [Case(data=case_data, answer={case_data.sla})]
    delay, max_workers = 60, 2

    logging.basicConfig(level=logging.WARNING)
    comparison.run(
        models=models[:5],
        cases=cases,
        delay=delay,
        max_workers=max_workers,
        report_filename=report_filename,
    )
    assert "INFO:srca" not in capfd.readouterr().err

    logging.basicConfig(level=logging.INFO, force=True)
    comparison.run(
        models=models[:5],
        cases=cases,
        delay=delay,
        max_workers=max_workers,
        report_filename=report_filename,
    )
    assert "INFO:srca" in capfd.readouterr().err
