"""
experiments/exp_bs_1d.py

European Black–Scholes PDE (1D) experiment for PIGB (mixed learners).

What it does:
- samples interior collocation points Z_pde = (x,tau), x=log(S/K), tau=T-t
- samples boundary points Z_bc in transformed coordinates
- trains PIGB using:
    * interior step: fit smooth learner to negative PDE residual
    * boundary step: fit smooth learner to boundary residual (debug baseline)
- evaluates:
    * price RMSE vs closed-form BS
    * delta RMSE (finite diff on model) vs closed-form BS delta
    * PDE residual RMS on probe points
- saves:
    results/bs1d_metrics.csv
    results/figures/bs1d_*.png

Run:
  python experiments/exp_bs_1d.py --rounds 200 --n_pde 4000 --n_bc 2000 --n_test 4000
"""

import os
import time
import math
import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.pigb import PIGB
from src.models.weak_learners import RidgeBasisLearner
from src.models.basis_functions import RBFMap


# ----------------------------
# 1) Black–Scholes closed form
# ----------------------------

def _norm_cdf(x: np.ndarray) -> np.ndarray:
    # Standard normal CDF via erf; vectorized
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def bs_call_price(S, K, r, sigma, tau):
    """
    European call price at time-to-maturity tau = T - t.
    """
    S = np.asarray(S, dtype=float)
    tau = np.asarray(tau, dtype=float)
    eps = 1e-12
    tau = np.maximum(tau, eps)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    return S * _norm_cdf(d1) - K * np.exp(-r * tau) * _norm_cdf(d2)


def bs_call_delta(S, K, r, sigma, tau):
    """
    Delta of European call.
    """
    S = np.asarray(S, dtype=float)
    tau = np.asarray(tau, dtype=float)
    eps = 1e-12
    tau = np.maximum(tau, eps)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    return _norm_cdf(d1)


# ----------------------------
# 2) Coordinate transform + finite differences
# ----------------------------

def st_to_z(S: np.ndarray, t: np.ndarray, p: "BSParams") -> np.ndarray:
    S = np.asarray(S, dtype=float)
    t = np.asarray(t, dtype=float)
    x = np.log(np.maximum(S, 1e-12) / p.K)
    tau = np.maximum(p.T - t, 0.0)
    return np.column_stack([x.reshape(-1), tau.reshape(-1)])


def z_to_st(Z: np.ndarray, p: "BSParams") -> tuple[np.ndarray, np.ndarray]:
    Z = np.asarray(Z, dtype=float)
    x = Z[:, 0]
    tau = Z[:, 1]
    S = p.K * np.exp(x)
    t = p.T - tau
    return S, t

def fd_uS(predict_ST, S, t, rel=1e-3, abs_min=1e-2):
    """
    Central diff in S with relative step:
      h = max(abs_min, rel * max(|S|, 1))
    """
    S = np.asarray(S, float)
    h = np.maximum(abs_min, rel * np.maximum(np.abs(S), 1.0))
    return (predict_ST(S + h, t) - predict_ST(S - h, t)) / (2.0 * h)


def fd_uSS(predict_ST, S, t, rel=1e-3, abs_min=1e-2):
    S = np.asarray(S, float)
    h = np.maximum(abs_min, rel * np.maximum(np.abs(S), 1.0))
    return (predict_ST(S + h, t) - 2.0 * predict_ST(S, t) + predict_ST(S - h, t)) / (h * h)


def fd_ut(predict_ST, S, t, h=1e-4):
    return (predict_ST(S, t + h) - predict_ST(S, t - h)) / (2.0 * h)


def fd_ux(predict_Z, x, tau, h=1e-3):
    return (predict_Z(x + h, tau) - predict_Z(x - h, tau)) / (2.0 * h)


def fd_uxx(predict_Z, x, tau, h=1e-3):
    return (predict_Z(x + h, tau) - 2.0 * predict_Z(x, tau) + predict_Z(x - h, tau)) / (h * h)


def fd_utau(predict_Z, x, tau, h=1e-4):
    return (predict_Z(x, tau + h) - predict_Z(x, tau - h)) / (2.0 * h)


# ----------------------------
# 3) Sampling
# ----------------------------

@dataclass
class BSParams:
    K: float = 100.0
    r: float = 0.05
    sigma: float = 0.2
    T: float = 1.0
    S_min: float = 1e-3
    S_max: float = 300.0


def sample_interior(n: int, p: BSParams, rng: np.random.Generator) -> np.ndarray:
    """
    Interior collocation points in transformed coordinates:
      x ~ Unif(log(S_min/K), log(S_max/K))
      tau ~ Unif(0,T)
    Returns Z_pde with columns [x, tau].
    """
    tau_eps = 1e-4
    tau = rng.uniform(tau_eps, max(tau_eps, p.T - tau_eps), size=n)
    x = rng.uniform(np.log(p.S_min / p.K), np.log(p.S_max / p.K), size=n)
    return np.column_stack([x, tau])


def sample_terminal(n: int, p: BSParams, rng: np.random.Generator) -> np.ndarray:
    """
    Terminal boundary in transformed coordinates:
      tau = 0
      x ~ Unif(log(S_min/K), log(S_max/K))
    """
    x = rng.uniform(np.log(p.S_min / p.K), np.log(p.S_max / p.K), size=n)
    tau = np.zeros(n, dtype=float)
    return np.column_stack([x, tau])


def sample_side_boundaries(n: int, p: BSParams, rng: np.random.Generator) -> np.ndarray:
    """
    Side boundaries over transformed time:
      x = log(S_min/K) or x = log(S_max/K), tau ~ Unif(0,T)
    """
    n_lo = n // 2
    n_hi = n - n_lo

    tau_lo = rng.uniform(0.0, p.T, size=n_lo)
    tau_hi = rng.uniform(0.0, p.T, size=n_hi)
    x_lo = np.full(n_lo, np.log(p.S_min / p.K), dtype=float)
    x_hi = np.full(n_hi, np.log(p.S_max / p.K), dtype=float)

    Z_lo = np.column_stack([x_lo, tau_lo])
    Z_hi = np.column_stack([x_hi, tau_hi])
    return np.vstack([Z_lo, Z_hi])


def sample_boundary(n: int, p: BSParams, rng: np.random.Generator, side_ratio: float) -> np.ndarray:
    """
    Combined boundary set:
      - terminal condition at t=T
      - side boundaries at S=S_min and S=S_max
    """
    n_side = int(round(n * side_ratio))
    n_side = max(0, min(n, n_side))
    n_term = n - n_side
    X_term = sample_terminal(n_term, p, rng)
    X_side = sample_side_boundaries(n_side, p, rng) if n_side > 0 else np.empty((0, 2), dtype=float)
    return np.vstack([X_term, X_side])


def sample_test(n: int, p: BSParams, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """
    Test points across domain:
      t ~ Unif(0,T)
      log S ~ Unif(log S_min, log S_max)
    Returns (S_test, t_test).
    """
    t = rng.uniform(0.0, p.T, size=n)
    logS = rng.uniform(np.log(p.S_min), np.log(p.S_max), size=n)
    S = np.exp(logS)
    return S, t


# ----------------------------
# 4) PDE residual + BC target functions
# ----------------------------

def make_bc_target_fn(p: BSParams):
    """
    Mixed boundary targets:
      terminal: u(S,T) = max(S-K, 0)
      low side: u(S_min, t) ≈ 0
      high side: u(S_max, t) ≈ S_max - K exp(-r(T-t))
    """
    x_min = np.log(p.S_min / p.K)
    x_max = np.log(p.S_max / p.K)

    def bc_target_fn(X_bc: np.ndarray) -> np.ndarray:
        x = X_bc[:, 0]
        tau = X_bc[:, 1]
        S = p.K * np.exp(x)

        y = np.maximum(S - p.K, 0.0)  # terminal target (tau=0)

        is_low = np.isclose(x, x_min, atol=1e-14, rtol=0.0)
        is_high = np.isclose(x, x_max, atol=1e-14, rtol=0.0)

        y[is_low] = 0.0
        y[is_high] = p.S_max - p.K * np.exp(-p.r * tau[is_high])
        return y
    return bc_target_fn


def make_pde_residual_fn(p: BSParams):
    """
    Black–Scholes PDE in transformed variables (x, tau):
      u_tau = 0.5*sigma^2*u_xx + (r - 0.5*sigma^2)*u_x - r*u
    Residual:
      R = u_tau - 0.5*sigma^2*u_xx - (r - 0.5*sigma^2)*u_x + r*u
    """
    def pde_residual_fn(predict_fn, X_pde: np.ndarray) -> np.ndarray:
        x = X_pde[:, 0]
        tau = X_pde[:, 1]

        def predict_Z(xv, tauv):
            X = np.column_stack([
                np.asarray(xv, float).reshape(-1),
                np.asarray(tauv, float).reshape(-1),
            ])
            return np.asarray(predict_fn(X), float).reshape(-1)

        utau = fd_utau(predict_Z, x, tau, h=1e-4)
        ux = fd_ux(predict_Z, x, tau, h=1e-3)
        uxx = fd_uxx(predict_Z, x, tau, h=1e-3)
        u = predict_Z(x, tau)

        return utau - 0.5 * (p.sigma ** 2) * uxx - (p.r - 0.5 * p.sigma ** 2) * ux + p.r * u

    return pde_residual_fn


# ----------------------------
# 5) Main experiment
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--n_pde", type=int, default=4000)
    ap.add_argument("--n_bc", type=int, default=2000)
    ap.add_argument("--n_test", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="results")

    # learner / basis knobs
    ap.add_argument("--rbf_centers", type=int, default=200)
    ap.add_argument("--rbf_sigma", type=float, default=1.0)
    ap.add_argument("--ridge_lam", type=float, default=1e-4)
    ap.add_argument("--nu_int", type=float, default=0.01)
    ap.add_argument("--nu_bdry", type=float, default=0.01)
    ap.add_argument("--bc_side_ratio", type=float, default=0.5)

    args = ap.parse_args()
    print(
        f"Starting BS-1D experiment: rounds={args.rounds}, n_pde={args.n_pde}, n_bc={args.n_bc}, n_test={args.n_test}",
        flush=True,
    )

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "figures"), exist_ok=True)

    rng = np.random.default_rng(args.seed)
    p = BSParams()

    # Data
    X_pde = sample_interior(args.n_pde, p, rng)
    X_bc = sample_boundary(args.n_bc, p, rng, side_ratio=args.bc_side_ratio)
    print(f"Sampled datasets: X_pde={X_pde.shape}, X_bc={X_bc.shape}", flush=True)

    S_te, t_te = sample_test(args.n_test, p, rng)
    tau_te = p.T - t_te
    u_true = bs_call_price(S_te, p.K, p.r, p.sigma, tau_te)
    d_true = bs_call_delta(S_te, p.K, p.r, p.sigma, tau_te)

    Z_te = st_to_z(S_te, t_te, p)
    print(f"Prepared test set: Z_te={Z_te.shape}", flush=True)

    bc_target_fn = make_bc_target_fn(p)
    pde_residual_fn = make_pde_residual_fn(p)

    def make_fmap():
        return RBFMap(
            n_centers=args.rbf_centers,
            sigma=args.rbf_sigma,
            add_bias=True,
            standardize=True,
            center_seed=args.seed,
        )

    def interior_factory():
        return RidgeBasisLearner(feature_map=make_fmap(), lam=args.ridge_lam)

    def boundary_factory():
        # Debug baseline: keep boundary smooth too.
        return RidgeBasisLearner(feature_map=make_fmap(), lam=args.ridge_lam)

    model = PIGB(
        B=args.rounds,
        interior_learner_factory=interior_factory,
        boundary_learner_factory=boundary_factory,
        pde_residual_fn=pde_residual_fn,
        bc_target_fn=bc_target_fn,
        nu_int=args.nu_int,
        nu_bdry=args.nu_bdry,
        f0=None,
        verbose=True,
    )

    rows = []
    t0 = time.time()

    def eval_callback(b: int, mdl: PIGB):
        u_pred = mdl.predict(Z_te)

        def predict_Z(xv, tauv):
            Z = np.column_stack([np.asarray(xv, float).reshape(-1), np.asarray(tauv, float).reshape(-1)])
            return np.asarray(mdl.predict(Z), float).reshape(-1)

        ux_pred = fd_ux(predict_Z, Z_te[:, 0], Z_te[:, 1], h=1e-3)
        delta_pred = ux_pred / np.maximum(S_te, 1e-8)

        rmse_u = float(np.sqrt(np.mean((u_pred - u_true) ** 2)))
        rmse_d = float(np.sqrt(np.mean((delta_pred - d_true) ** 2)))

        X_probe = sample_interior(min(800, args.n_pde), p, rng)
        res = pde_residual_fn(lambda X: mdl.predict(X), X_probe)
        res_rms = float(np.sqrt(np.mean(res ** 2)))

        elapsed = float(time.time() - t0)

        rows.append({
            "round": b,
            "rmse_price": rmse_u,
            "rmse_delta": rmse_d,
            "pde_residual_rms": res_rms,
            "time_sec": elapsed,
        })

        if b == 1 or b % max(1, args.rounds // 10) == 0:
            print(
                f"[eval {b:4d}] rmse_u={rmse_u:.6g}  rmse_delta={rmse_d:.6g}  res_rms={res_rms:.6g}  time={elapsed:.2f}s",
                flush=True,
            )

    print("Training started...", flush=True)
    model.fit(X_pde=X_pde, X_bc=X_bc, eval_callback=eval_callback)
    print("Training finished.", flush=True)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.outdir, "bs1d_metrics.csv")
    df.to_csv(csv_path, index=False)
    print("Saved:", csv_path, flush=True)

    def save_plot(ycol: str, ylabel: str, fname: str):
        plt.figure()
        plt.plot(df["round"], df[ycol])
        plt.xlabel("Boosting rounds")
        plt.ylabel(ylabel)
        plt.title(f"BS 1D: {ylabel} vs rounds")
        out = os.path.join(args.outdir, "figures", fname)
        plt.tight_layout()
        plt.savefig(out, dpi=200)
        plt.close()
        print("Saved:", out, flush=True)

    save_plot("rmse_price", "Price RMSE", "bs1d_rmse_vs_rounds.png")
    save_plot("rmse_delta", "Delta RMSE", "bs1d_delta_rmse_vs_rounds.png")
    save_plot("pde_residual_rms", "PDE residual RMS", "bs1d_pde_residual_vs_rounds.png")

    plt.figure()
    plt.plot(df["round"], df["time_sec"])
    plt.xlabel("Boosting rounds")
    plt.ylabel("Elapsed time (sec)")
    plt.title("BS 1D: time vs rounds")
    out = os.path.join(args.outdir, "figures", "bs1d_time_vs_rounds.png")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print("Saved:", out, flush=True)


if __name__ == "__main__":
    main()
