"""
Simple evaluation metrics for the experiments.
"""

import numpy as np


def _as_array(x):
    """
    Convert input to a NumPy array with float dtype.
    """
    return np.asarray(x, dtype=float)


def rmse(pred, true):
    """
    Root mean squared error.

    Parameters
    ----------
    pred : array-like
        Predicted values.
    true : array-like
        True values.

    Returns
    -------
    float
        Root mean squared error.
    """
    pred = _as_array(pred).ravel()
    true = _as_array(true).ravel()

    if pred.shape != true.shape:
        raise ValueError(
            f"pred and true must have the same shape, got {pred.shape} and {true.shape}."
        )

    return np.sqrt(np.mean((pred - true) ** 2))


def mae(pred, true):
    """
    Mean absolute error.
    """
    pred = _as_array(pred).ravel()
    true = _as_array(true).ravel()

    if pred.shape != true.shape:
        raise ValueError(
            f"pred and true must have the same shape, got {pred.shape} and {true.shape}."
        )

    return np.mean(np.abs(pred - true))


def pde_rmse(residual):
    """
    Root mean squared PDE residual.
    """
    residual = _as_array(residual).ravel()
    return np.sqrt(np.mean(residual ** 2))


def coverage_rate(lower, upper, target):
    """
    Empirical coverage rate for confidence intervals.

    Returns the fraction of target values satisfying

        lower <= target <= upper.
    """
    lower = _as_array(lower).ravel()
    upper = _as_array(upper).ravel()
    target = _as_array(target).ravel()

    if lower.shape != upper.shape or lower.shape != target.shape:
        raise ValueError("lower, upper, and target must have the same shape.")

    return np.mean((lower <= target) & (target <= upper))


def average_ci_width(lower, upper):
    """
    Average confidence interval width.
    """
    lower = _as_array(lower).ravel()
    upper = _as_array(upper).ravel()

    if lower.shape != upper.shape:
        raise ValueError("lower and upper must have the same shape.")

    return np.mean(upper - lower)


def standard_error_calibration(error, se):
    """
    Diagnostic ratio:

        std(error) / mean(se)

    If the standard errors are well-calibrated, this should be roughly near 1.

    This is not a formal test; it is just a useful simulation diagnostic.
    """
    error = _as_array(error).ravel()
    se = _as_array(se).ravel()

    if error.shape != se.shape:
        raise ValueError("error and se must have the same shape.")

    mean_se = np.mean(se)

    if mean_se <= 0:
        return np.nan

    return np.std(error, ddof=1) / mean_se
