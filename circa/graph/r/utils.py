"""
Utilities
"""
from functools import partial

import bayesian_changepoint_detection.online_changepoint_detection as oncd
import numpy as np


def bcp(data: np.ndarray, tau_max: int = 3, confidence: float = 0.5) -> np.ndarray:
    """
    Bayesian change-point detection

    Used in CauseInfer in INFOCOM'14
    https://doi.org/10.1109/INFOCOM.2014.6848128
    """
    probs, _ = oncd.online_changepoint_detection(
        data, partial(oncd.constant_hazard, 250), oncd.StudentT(0.1, 0.01, 1, 0)
    )
    change_points = np.zeros(data.shape)
    change_points[tau_max:] = probs[tau_max, tau_max + 1 :] > confidence
    return change_points
