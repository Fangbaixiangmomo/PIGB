"""
Boundary-informed baseline functions.

The baseline is chosen before training and does not use observed prices
or the analytical Black-Scholes solution.

For one-dimensional Black-Scholes experiments, we use the transport baseline

    Phi(S, t) = exp(-r (T - t)) H_smooth(S exp(r (T - t))),

where H_smooth is a smoothed payoff.

This baseline solves the first-order transport-discount part of the PDE,
but intentionally ignores the diffusion term. Therefore the learned finite-basis
correction still has meaningful work to do.
"""

import numpy as np


def _as_array(x):
    """
    Convert input to a NumPy array with float dtype.
    """
    return np.asarray(x, dtype=float)


def _time_to_maturity(t, T):
    """
    Compute tau = T - t.
    """
    t = _as_array(t)
    return T - t


def _sigmoid(x):
    """
    Numerically stable sigmoid.
    """
    x = _as_array(x)

    out = np.empty_like(x, dtype=float)

    positive = x >= 0
    negative = ~positive

    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))

    exp_x = np.exp(x[negative])
    out[negative] = exp_x / (1.0 + exp_x)

    return out


def _softplus(x):
    """
    Numerically stable softplus: log(1 + exp(x)).
    """
    x = _as_array(x)
    return np.logaddexp(0.0, x)


# ============================================================
# Smoothed call payoff
# ============================================================

def softplus_call_payoff(S, K, smoothness=2.0):
    """
    Smooth approximation to the call payoff:

        (S - K)^+

    using

        smoothness * log(1 + exp((S - K) / smoothness)).

    A smaller smoothness gives a sharper approximation to the kink.
    """
    S = _as_array(S)
    x = (S - K) / smoothness
    return smoothness * _softplus(x)


def softplus_call_payoff_derivative(S, K, smoothness=2.0):
    """
    First derivative of the smoothed call payoff with respect to S.
    """
    S = _as_array(S)
    x = (S - K) / smoothness
    return _sigmoid(x)


def softplus_call_payoff_second_derivative(S, K, smoothness=2.0):
    """
    Second derivative of the smoothed call payoff with respect to S.
    """
    S = _as_array(S)
    x = (S - K) / smoothness
    p = _sigmoid(x)
    return p * (1.0 - p) / smoothness


# ============================================================
# Smoothed digital payoff
# ============================================================

def logistic_digital_payoff(S, K, smoothness=2.0):
    """
    Smooth approximation to the cash-or-nothing digital payoff:

        1{S > K}

    using

        sigmoid((S - K) / smoothness).
    """
    S = _as_array(S)
    x = (S - K) / smoothness
    return _sigmoid(x)


def logistic_digital_payoff_derivative(S, K, smoothness=2.0):
    """
    First derivative of the smoothed digital payoff with respect to S.
    """
    S = _as_array(S)
    x = (S - K) / smoothness
    p = _sigmoid(x)
    return p * (1.0 - p) / smoothness


def logistic_digital_payoff_second_derivative(S, K, smoothness=2.0):
    """
    Second derivative of the smoothed digital payoff with respect to S.
    """
    S = _as_array(S)
    x = (S - K) / smoothness
    p = _sigmoid(x)
    return p * (1.0 - p) * (1.0 - 2.0 * p) / (smoothness ** 2)


# ============================================================
# Generic transport baseline helper
# ============================================================

def _transport_baseline(
    S,
    t,
    K,
    r,
    T,
    payoff_func,
    payoff_derivative_func,
    payoff_second_derivative_func,
    smoothness=2.0,
):
    """
    Compute transport baseline and its derivatives.

    Let tau = T - t,
        q = S exp(r tau),
        d = exp(-r tau).

    Then

        Phi(S,t) = d H(q).

    Derivatives:

        Phi_S  = H'(q),
        Phi_SS = exp(r tau) H''(q).

    The time derivative is

        Phi_t = r d {H(q) - q H'(q)}.

    The Black-Scholes PDE residual is

        Phi_t + r S Phi_S + 0.5 sigma^2 S^2 Phi_SS - r Phi.

    The drift-discount terms cancel by construction, so only the diffusion
    term remains once sigma is supplied.
    """
    S = _as_array(S)
    tau = _time_to_maturity(t, T)

    growth = np.exp(r * tau)
    discount = np.exp(-r * tau)

    q = S * growth

    H = payoff_func(q, K=K, smoothness=smoothness)
    H_prime = payoff_derivative_func(q, K=K, smoothness=smoothness)
    H_second = payoff_second_derivative_func(q, K=K, smoothness=smoothness)

    value = discount * H
    delta = H_prime
    gamma = growth * H_second
    dt = r * discount * (H - q * H_prime)

    return {
        "value": value,
        "delta": delta,
        "gamma": gamma,
        "dt": dt,
    }


# ============================================================
# Call transport baseline
# ============================================================

def transport_call_baseline(S, t, K, r, T, smoothness=2.0):
    """
    Boundary-informed transport baseline for a European call.
    """
    out = _transport_baseline(
        S=S,
        t=t,
        K=K,
        r=r,
        T=T,
        payoff_func=softplus_call_payoff,
        payoff_derivative_func=softplus_call_payoff_derivative,
        payoff_second_derivative_func=softplus_call_payoff_second_derivative,
        smoothness=smoothness,
    )
    return out["value"]


def transport_call_delta(S, t, K, r, T, smoothness=2.0):
    """
    S-derivative of the call transport baseline.
    """
    out = _transport_baseline(
        S=S,
        t=t,
        K=K,
        r=r,
        T=T,
        payoff_func=softplus_call_payoff,
        payoff_derivative_func=softplus_call_payoff_derivative,
        payoff_second_derivative_func=softplus_call_payoff_second_derivative,
        smoothness=smoothness,
    )
    return out["delta"]


def transport_call_gamma(S, t, K, r, T, smoothness=2.0):
    """
    Second S-derivative of the call transport baseline.
    """
    out = _transport_baseline(
        S=S,
        t=t,
        K=K,
        r=r,
        T=T,
        payoff_func=softplus_call_payoff,
        payoff_derivative_func=softplus_call_payoff_derivative,
        payoff_second_derivative_func=softplus_call_payoff_second_derivative,
        smoothness=smoothness,
    )
    return out["gamma"]


def transport_call_dt(S, t, K, r, T, smoothness=2.0):
    """
    Time derivative of the call transport baseline.
    """
    out = _transport_baseline(
        S=S,
        t=t,
        K=K,
        r=r,
        T=T,
        payoff_func=softplus_call_payoff,
        payoff_derivative_func=softplus_call_payoff_derivative,
        payoff_second_derivative_func=softplus_call_payoff_second_derivative,
        smoothness=smoothness,
    )
    return out["dt"]


def transport_call_pde_residual(S, t, K, r, sigma, T, smoothness=2.0):
    """
    Black-Scholes PDE residual of the call transport baseline.

    Since the transport baseline solves the drift-discount part exactly,
    the residual is only the diffusion contribution:

        0.5 * sigma^2 * S^2 * Phi_SS.
    """
    S = _as_array(S)
    gamma = transport_call_gamma(
        S=S,
        t=t,
        K=K,
        r=r,
        T=T,
        smoothness=smoothness,
    )
    return 0.5 * sigma ** 2 * S ** 2 * gamma


# ============================================================
# Digital transport baseline
# ============================================================

def transport_digital_baseline(S, t, K, r, T, smoothness=2.0):
    """
    Boundary-informed transport baseline for a cash-or-nothing digital call.
    """
    out = _transport_baseline(
        S=S,
        t=t,
        K=K,
        r=r,
        T=T,
        payoff_func=logistic_digital_payoff,
        payoff_derivative_func=logistic_digital_payoff_derivative,
        payoff_second_derivative_func=logistic_digital_payoff_second_derivative,
        smoothness=smoothness,
    )
    return out["value"]


def transport_digital_delta(S, t, K, r, T, smoothness=2.0):
    """
    S-derivative of the digital transport baseline.
    """
    out = _transport_baseline(
        S=S,
        t=t,
        K=K,
        r=r,
        T=T,
        payoff_func=logistic_digital_payoff,
        payoff_derivative_func=logistic_digital_payoff_derivative,
        payoff_second_derivative_func=logistic_digital_payoff_second_derivative,
        smoothness=smoothness,
    )
    return out["delta"]


def transport_digital_gamma(S, t, K, r, T, smoothness=2.0):
    """
    Second S-derivative of the digital transport baseline.
    """
    out = _transport_baseline(
        S=S,
        t=t,
        K=K,
        r=r,
        T=T,
        payoff_func=logistic_digital_payoff,
        payoff_derivative_func=logistic_digital_payoff_derivative,
        payoff_second_derivative_func=logistic_digital_payoff_second_derivative,
        smoothness=smoothness,
    )
    return out["gamma"]


def transport_digital_dt(S, t, K, r, T, smoothness=2.0):
    """
    Time derivative of the digital transport baseline.
    """
    out = _transport_baseline(
        S=S,
        t=t,
        K=K,
        r=r,
        T=T,
        payoff_func=logistic_digital_payoff,
        payoff_derivative_func=logistic_digital_payoff_derivative,
        payoff_second_derivative_func=logistic_digital_payoff_second_derivative,
        smoothness=smoothness,
    )
    return out["dt"]


def transport_digital_pde_residual(S, t, K, r, sigma, T, smoothness=2.0):
    """
    Black-Scholes PDE residual of the digital transport baseline.
    """
    S = _as_array(S)
    gamma = transport_digital_gamma(
        S=S,
        t=t,
        K=K,
        r=r,
        T=T,
        smoothness=smoothness,
    )
    return 0.5 * sigma ** 2 * S ** 2 * gamma