"""
RBF basis functions and analytical derivatives.

We use normalized coordinates

    x = log(S / K),
    q = t / T.

The j-th basis function is

    psi_j(S, t)
        = exp(- ((x - c_xj)^2 + (q - c_tj)^2) / (2 h^2)).

This file provides:
- RBF center construction
- RBF feature matrix
- first S-derivative
- second S-derivative
- time derivative
- Black-Scholes PDE matrix for the basis

The implementation is deliberately simple and vectorized.
"""

import numpy as np


def _as_array(x):
    """
    Convert input to a NumPy array with float dtype.
    """
    return np.asarray(x, dtype=float)


def _prepare_inputs(S, t, K, T):
    """
    Prepare S and t as flat arrays.

    Parameters
    ----------
    S : array-like
        Underlying prices. Must be positive.
    t : array-like
        Calendar times.
    K : float
        Strike.
    T : float
        Maturity.

    Returns
    -------
    S_flat : np.ndarray
        Flat array of S values.
    t_flat : np.ndarray
        Flat array of t values.
    x : np.ndarray
        log-moneyness coordinates log(S/K).
    q : np.ndarray
        normalized time coordinates t/T.
    """
    S = _as_array(S)
    t = _as_array(t)

    S, t = np.broadcast_arrays(S, t)

    S_flat = S.ravel()
    t_flat = t.ravel()

    if np.any(S_flat <= 0):
        raise ValueError("All S values must be positive.")

    if K <= 0:
        raise ValueError("K must be positive.")

    if T <= 0:
        raise ValueError("T must be positive.")

    x = np.log(S_flat / K)
    q = t_flat / T

    return S_flat, t_flat, x, q


def _check_centers_and_bandwidth(centers, bandwidth):
    """
    Validate centers and bandwidth.
    """
    centers = _as_array(centers)

    if centers.ndim != 2 or centers.shape[1] != 2:
        raise ValueError("centers must have shape (n_centers, 2).")

    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive.")

    return centers


# ============================================================
# RBF center construction
# ============================================================

def make_rbf_centers(
    S_min,
    S_max,
    t_min,
    t_max,
    n_s,
    n_t,
    K,
    T,
):
    """
    Create a tensor grid of RBF centers in normalized coordinates.

    The first coordinate is log(S/K).
    The second coordinate is t/T.

    Parameters
    ----------
    S_min, S_max : float
        Range of underlying prices used for placing centers.
    t_min, t_max : float
        Range of times used for placing centers.
    n_s : int
        Number of centers along the S/log-moneyness direction.
    n_t : int
        Number of centers along the time direction.
    K : float
        Strike.
    T : float
        Maturity.

    Returns
    -------
    centers : np.ndarray
        Array of shape (n_s * n_t, 2).
    """
    if S_min <= 0 or S_max <= 0:
        raise ValueError("S_min and S_max must be positive.")

    if S_min >= S_max:
        raise ValueError("S_min must be smaller than S_max.")

    if t_min >= t_max:
        raise ValueError("t_min must be smaller than t_max.")

    if n_s <= 0 or n_t <= 0:
        raise ValueError("n_s and n_t must be positive integers.")

    x_grid = np.linspace(np.log(S_min / K), np.log(S_max / K), n_s)
    q_grid = np.linspace(t_min / T, t_max / T, n_t)

    X, Q = np.meshgrid(x_grid, q_grid, indexing="ij")

    centers = np.column_stack([X.ravel(), Q.ravel()])

    return centers


# ============================================================
# RBF feature matrix
# ============================================================

def rbf_features(S, t, centers, bandwidth, K, T):
    """
    Compute the RBF feature matrix.

    Returns
    -------
    Psi : np.ndarray
        Matrix of shape (n_points, n_basis).
    """
    centers = _check_centers_and_bandwidth(centers, bandwidth)
    _, _, x, q = _prepare_inputs(S, t, K, T)

    cx = centers[:, 0]
    cq = centers[:, 1]

    dx = x[:, None] - cx[None, :]
    dq = q[:, None] - cq[None, :]

    h2 = bandwidth ** 2

    Psi = np.exp(-0.5 * (dx ** 2 + dq ** 2) / h2)

    return Psi


def rbf_dS(S, t, centers, bandwidth, K, T):
    """
    First derivative of RBF features with respect to S.

    Since x = log(S/K),

        d psi / dS
        = psi * (-(x - c_x) / h^2) * (1/S).
    """
    centers = _check_centers_and_bandwidth(centers, bandwidth)
    S_flat, _, x, q = _prepare_inputs(S, t, K, T)

    cx = centers[:, 0]
    cq = centers[:, 1]

    dx = x[:, None] - cx[None, :]
    dq = q[:, None] - cq[None, :]

    h2 = bandwidth ** 2

    Psi = np.exp(-0.5 * (dx ** 2 + dq ** 2) / h2)

    out = Psi * (-dx / h2) / S_flat[:, None]

    return out


def rbf_dSS(S, t, centers, bandwidth, K, T):
    """
    Second derivative of RBF features with respect to S.

    Let x = log(S/K), a = x - c_x, and h2 = h^2.

    Then

        d psi / dS = psi * ( -a / (h2 S) ).

    Differentiating again gives

        d2 psi / dS2
        = psi / S^2 * { a^2 / h2^2 + (a - 1) / h2 }.
    """
    centers = _check_centers_and_bandwidth(centers, bandwidth)
    S_flat, _, x, q = _prepare_inputs(S, t, K, T)

    cx = centers[:, 0]
    cq = centers[:, 1]

    dx = x[:, None] - cx[None, :]
    dq = q[:, None] - cq[None, :]

    h2 = bandwidth ** 2

    Psi = np.exp(-0.5 * (dx ** 2 + dq ** 2) / h2)

    out = (
        Psi
        / (S_flat[:, None] ** 2)
        * ((dx ** 2) / (h2 ** 2) + (dx - 1.0) / h2)
    )

    return out


def rbf_dt(S, t, centers, bandwidth, K, T):
    """
    Time derivative of RBF features with respect to calendar time t.

    Since q = t/T,

        d psi / dt
        = psi * (-(q - c_q) / h^2) * (1/T).
    """
    centers = _check_centers_and_bandwidth(centers, bandwidth)
    _, _, x, q = _prepare_inputs(S, t, K, T)

    cx = centers[:, 0]
    cq = centers[:, 1]

    dx = x[:, None] - cx[None, :]
    dq = q[:, None] - cq[None, :]

    h2 = bandwidth ** 2

    Psi = np.exp(-0.5 * (dx ** 2 + dq ** 2) / h2)

    out = Psi * (-dq / h2) / T

    return out


# ============================================================
# Black-Scholes PDE matrix for the basis
# ============================================================

def rbf_black_scholes_pde_matrix(
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
    Compute the Black-Scholes PDE applied to each RBF basis function.

    For each basis function psi_j, compute

        L[psi_j]
        = psi_{j,t}
          + r S psi_{j,S}
          + 0.5 sigma^2 S^2 psi_{j,SS}
          - r psi_j.

    Returns
    -------
    R : np.ndarray
        Matrix of shape (n_points, n_basis), where R[i, j] = L[psi_j](S_i, t_i).
    """
    S_flat, _, _, _ = _prepare_inputs(S, t, K, T)

    Psi = rbf_features(S, t, centers, bandwidth, K, T)
    Psi_t = rbf_dt(S, t, centers, bandwidth, K, T)
    Psi_S = rbf_dS(S, t, centers, bandwidth, K, T)
    Psi_SS = rbf_dSS(S, t, centers, bandwidth, K, T)

    R = (
        Psi_t
        + r * S_flat[:, None] * Psi_S
        + 0.5 * sigma ** 2 * (S_flat[:, None] ** 2) * Psi_SS
        - r * Psi
    )

    return R