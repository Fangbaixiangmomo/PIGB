"""
Finite-basis physics-informed estimator.

The estimator has the form

    u_hat(S, t) = baseline(S, t) + Psi(S, t) theta.

For the Black-Scholes PDE, the residual is linear in theta:

    L[u_hat] = L[baseline] + R theta,

where R is the matrix obtained by applying the Black-Scholes operator
to each basis function.

The estimator solves

    min_theta
        1/2 ||y - baseline - Psi theta||^2
        + lambda_pde / 2 ||baseline_pde + R theta||^2
        + lambda_ridge / 2 ||theta||^2.

This gives the closed-form normal equation

    (Psi'Psi + lambda_pde R'R + lambda_ridge I) theta
        =
    Psi'(y - baseline) - lambda_pde R' baseline_pde.

No optimizer is used.
"""

import numpy as np

from src.basis import (
    rbf_features,
    rbf_dS,
    rbf_dSS,
    rbf_black_scholes_pde_matrix,
)


def _as_array(x):
    """
    Convert input to a NumPy array with float dtype.
    """
    return np.asarray(x, dtype=float)


def _check_same_length(*arrays):
    """
    Check that all arrays have the same length.
    """
    lengths = [len(_as_array(a).ravel()) for a in arrays]

    if len(set(lengths)) != 1:
        raise ValueError(f"Arrays must have the same length, got lengths {lengths}.")


def build_design_matrices(
    S,
    t,
    centers,
    bandwidth,
    K,
    r,
    sigma,
    T,
):
    """
    Build the supervised basis matrix and PDE basis matrix.

    Returns
    -------
    design : dict
        Contains:
        - Psi: basis matrix
        - R: Black-Scholes PDE matrix applied to basis functions
    """
    Psi = rbf_features(
        S=S,
        t=t,
        centers=centers,
        bandwidth=bandwidth,
        K=K,
        T=T,
    )

    R = rbf_black_scholes_pde_matrix(
        S=S,
        t=t,
        centers=centers,
        bandwidth=bandwidth,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
    )

    return {
        "Psi": Psi,
        "R": R,
    }


def fit_finite_basis(
    S_train,
    t_train,
    y_train,
    baseline_train,
    baseline_pde_train,
    centers,
    bandwidth,
    K,
    r,
    sigma,
    T,
    lambda_pde=0.0,
    lambda_ridge=0.0,
):
    """
    Fit the finite-basis physics-informed estimator.

    Parameters
    ----------
    S_train, t_train : array-like
        Training design points.
    y_train : array-like
        Noisy observed prices.
    baseline_train : array-like
        Baseline values Phi(S_i, t_i).
    baseline_pde_train : array-like
        PDE residual values L[Phi](S_i, t_i).
        For the zero baseline, this should be zero.
    centers : np.ndarray
        RBF centers.
    bandwidth : float
        RBF bandwidth.
    K, r, sigma, T : float
        Black-Scholes parameters.
    lambda_pde : float
        PDE penalty weight.
    lambda_ridge : float
        Small ridge penalty for numerical stability.

    Returns
    -------
    fit : dict
        Contains theta and useful matrices for later inference.
    """
    S_train = _as_array(S_train).ravel()
    t_train = _as_array(t_train).ravel()
    y_train = _as_array(y_train).ravel()
    baseline_train = _as_array(baseline_train).ravel()
    baseline_pde_train = _as_array(baseline_pde_train).ravel()

    _check_same_length(S_train, t_train, y_train, baseline_train, baseline_pde_train)

    design = build_design_matrices(
        S=S_train,
        t=t_train,
        centers=centers,
        bandwidth=bandwidth,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
    )

    Psi = design["Psi"]
    R = design["R"]

    n_basis = Psi.shape[1]

    y_centered = y_train - baseline_train

    # ------------------------------------------------------------
    # Solve the same least-squares problem using the augmented design.
    #
    # This is numerically more stable than directly solving the
    # normal equations. When lambda_ridge = 0, this does NOT add ridge.
    # ------------------------------------------------------------

    A = Psi
    b = y_centered

    if lambda_pde > 0:
        A = np.vstack([
            A,
            np.sqrt(lambda_pde) * R,
        ])

        b = np.concatenate([
            b,
            -np.sqrt(lambda_pde) * baseline_pde_train,
        ])

    if lambda_ridge > 0:
        A = np.vstack([
            A,
            np.sqrt(lambda_ridge) * np.eye(n_basis),
        ])

        b = np.concatenate([
            b,
            np.zeros(n_basis),
        ])

    theta, *_ = np.linalg.lstsq(A, b, rcond=None)

    # Keep lhs for diagnostics and covariance formulas.
    lhs = (
        Psi.T @ Psi
        + lambda_pde * (R.T @ R)
        + lambda_ridge * np.eye(n_basis)
    )

    fitted_price_train = baseline_train + Psi @ theta
    fitted_pde_train = baseline_pde_train + R @ theta

    price_residual_train = y_train - fitted_price_train

    return {
        "theta": theta,
        "lhs": lhs,
        "Psi_train": Psi,
        "R_train": R,
        "fitted_price_train": fitted_price_train,
        "fitted_pde_train": fitted_pde_train,
        "price_residual_train": price_residual_train,
        "lambda_pde": lambda_pde,
        "lambda_ridge": lambda_ridge,
        "augmented_design": A,
    }


def predict_price(
    S,
    t,
    baseline,
    theta,
    centers,
    bandwidth,
    K,
    T,
):
    """
    Predict price:

        baseline(S,t) + Psi(S,t) theta.
    """
    baseline = _as_array(baseline).ravel()
    theta = _as_array(theta).ravel()

    Psi = rbf_features(
        S=S,
        t=t,
        centers=centers,
        bandwidth=bandwidth,
        K=K,
        T=T,
    )

    return baseline + Psi @ theta


def predict_delta(
    S,
    t,
    baseline_delta,
    theta,
    centers,
    bandwidth,
    K,
    T,
):
    """
    Predict Delta:

        baseline_delta(S,t) + partial_S Psi(S,t) theta.
    """
    baseline_delta = _as_array(baseline_delta).ravel()
    theta = _as_array(theta).ravel()

    Psi_S = rbf_dS(
        S=S,
        t=t,
        centers=centers,
        bandwidth=bandwidth,
        K=K,
        T=T,
    )

    return baseline_delta + Psi_S @ theta


def predict_gamma(
    S,
    t,
    baseline_gamma,
    theta,
    centers,
    bandwidth,
    K,
    T,
):
    """
    Predict Gamma:

        baseline_gamma(S,t) + partial_SS Psi(S,t) theta.
    """
    baseline_gamma = _as_array(baseline_gamma).ravel()
    theta = _as_array(theta).ravel()

    Psi_SS = rbf_dSS(
        S=S,
        t=t,
        centers=centers,
        bandwidth=bandwidth,
        K=K,
        T=T,
    )

    return baseline_gamma + Psi_SS @ theta


def pde_residual(
    S,
    t,
    baseline_pde,
    theta,
    centers,
    bandwidth,
    K,
    r,
    sigma,
    T,
):
    """
    Compute Black-Scholes PDE residual of the fitted surface:

        L[u_hat](S,t) = L[baseline](S,t) + R(S,t) theta.
    """
    baseline_pde = _as_array(baseline_pde).ravel()
    theta = _as_array(theta).ravel()

    R = rbf_black_scholes_pde_matrix(
        S=S,
        t=t,
        centers=centers,
        bandwidth=bandwidth,
        K=K,
        r=r,
        sigma=sigma,
        T=T,
    )

    return baseline_pde + R @ theta


def sandwich_covariance(fit):
    """
    Simple heteroskedastic sandwich covariance estimate for theta.

    This is mainly for the later coverage experiment.

    The estimating equation perturbation comes from the projected price noise.
    The PDE penalty changes the curvature but does not directly add response noise.

    Returns
    -------
    theta_cov : np.ndarray
        Estimated covariance matrix of theta_hat.
    """
    lhs = fit["lhs"]
    Psi = fit["Psi_train"]
    residual = fit["price_residual_train"]

    # Meat: Psi' diag(residual^2) Psi
    meat = Psi.T @ ((residual ** 2)[:, None] * Psi)

    lhs_inv = np.linalg.inv(lhs)

    theta_cov = lhs_inv @ meat @ lhs_inv.T

    return theta_cov


def prediction_standard_error(
    S,
    t,
    theta_cov,
    centers,
    bandwidth,
    K,
    T,
):
    """
    Standard error for the fitted correction Psi(S,t)' theta.

    The baseline is deterministic, so it contributes no sampling variance.
    """
    Psi = rbf_features(
        S=S,
        t=t,
        centers=centers,
        bandwidth=bandwidth,
        K=K,
        T=T,
    )

    variances = np.sum((Psi @ theta_cov) * Psi, axis=1)

    # Numerical guard against tiny negative values caused by floating point error.
    variances = np.maximum(variances, 0.0)

    return np.sqrt(variances)