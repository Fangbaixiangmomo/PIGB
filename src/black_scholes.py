"""
Black-Scholes pricing formulas used for simulation and evaluation.

This file is intentionally simple:
- It provides analytical prices and Greeks.
- These functions are used to generate ground truth in experiments.
- The estimator itself should not depend on the analytical solution during fitting.

We use t as calendar time and T as maturity, so tau = T - t.
"""

import numpy as np
from scipy.stats import norm


def _as_array(x):
    """
    Convert input to a NumPy array with float dtype.
    This allows the functions to work with both scalars and arrays.
    """
    return np.asarray(x, dtype=float)


def _time_to_maturity(t, T, eps=1e-10):
    """
    Compute tau = T - t, clipped away from zero for numerical stability.

    In the experiments, we usually avoid sampling exactly at maturity.
    Still, this helper prevents division by zero.
    """
    t = _as_array(t)
    return np.maximum(T - t, eps)


def _d1_d2(S, t, K, r, sigma, T):
    """
    Compute the Black-Scholes d1 and d2 quantities.
    """
    S = _as_array(S)
    tau = _time_to_maturity(t, T)

    sqrt_tau = np.sqrt(tau)

    d1 = (
        np.log(S / K)
        + (r + 0.5 * sigma ** 2) * tau
    ) / (sigma * sqrt_tau)

    d2 = d1 - sigma * sqrt_tau

    return d1, d2


# ============================================================
# European call option
# ============================================================

def bs_call_price(S, t, K, r, sigma, T):
    """
    Black-Scholes price of a European call option.

    Parameters
    ----------
    S : array-like
        Underlying price.
    t : array-like
        Current time.
    K : float
        Strike price.
    r : float
        Risk-free rate.
    sigma : float
        Volatility.
    T : float
        Maturity.

    Returns
    -------
    price : np.ndarray
        Call option price.
    """
    S = _as_array(S)
    tau = _time_to_maturity(t, T)

    d1, d2 = _d1_d2(S, t, K, r, sigma, T)

    price = S * norm.cdf(d1) - K * np.exp(-r * tau) * norm.cdf(d2)

    return price


def bs_call_delta(S, t, K, r, sigma, T):
    """
    Delta of a European call option. Derivative of price with respect to S.
    """
    d1, _ = _d1_d2(S, t, K, r, sigma, T)
    return norm.cdf(d1)


def bs_call_gamma(S, t, K, r, sigma, T):
    """
    Gamma of a European call option. Derative of delta with respect to S.
    """
    S = _as_array(S)
    tau = _time_to_maturity(t, T)

    d1, _ = _d1_d2(S, t, K, r, sigma, T)

    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(tau))

    return gamma


# ============================================================
# Cash-or-nothing digital call option
# ============================================================

def bs_digital_price(S, t, K, r, sigma, T):
    """
    Black-Scholes price of a cash-or-nothing digital call.

    The payoff is:

        1{S_T > K}

    so the price is:

        exp(-r tau) Phi(d2).
    """
    tau = _time_to_maturity(t, T)
    _, d2 = _d1_d2(S, t, K, r, sigma, T)

    # The discount factor exp(-r tau)
    price = np.exp(-r * tau) * norm.cdf(d2)

    return price


def bs_digital_delta(S, t, K, r, sigma, T):
    """
    Delta of a cash-or-nothing digital call. Derivative of price with respect to S.
    """
    S = _as_array(S)
    tau = _time_to_maturity(t, T)

    _, d2 = _d1_d2(S, t, K, r, sigma, T)

    delta = (
        np.exp(-r * tau)
        * norm.pdf(d2)
        / (S * sigma * np.sqrt(tau))
    )

    return delta


def bs_digital_gamma(S, t, K, r, sigma, T):
    """
    Gamma of a cash-or-nothing digital call. Derivative of delta with respect to S.

    If a = sigma * sqrt(tau), then

        Delta = exp(-r tau) phi(d2) / (S a).

    Differentiating with respect to S gives

        Gamma = - exp(-r tau) phi(d2) * (d2 / a + 1) / (S^2 a).
    """
    S = _as_array(S)
    tau = _time_to_maturity(t, T)

    a = sigma * np.sqrt(tau)
    _, d2 = _d1_d2(S, t, K, r, sigma, T)

    gamma = (
        -np.exp(-r * tau)
        * norm.pdf(d2)
        * (d2 / a + 1.0)
        / (S ** 2 * a)
    )

    return gamma