"""
experiments/exp_bs_1d.py

European Black–Scholes PDE (1D) experiment for PIGB (mixed learners).

What it does:
- samples interior collocation points X_pde = (S,t)
- samples terminal points X_bc = (S,T)
- trains PIGB using:
    * interior step: fit smooth learner to negative PDE residual
    * boundary step: fit tree learner to boundary residual
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
from src.models.weak_learners import RidgeBasisLearner, TreeLearner
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
# 2) Finite differences (for PDE residual + delta)
# ----------------------------

def fd_uS(predict_ST, S, t, h=1e-2):
    # Central diff in S
    return (predict_ST(S + h, t) - predict_ST(S - h, t)) / (2.0 * h)


def fd_uSS(predict_ST, S, t, h=1e-2):
    return (predict_ST(S + h, t) - 2.0 * predict_ST(S, t) + predict_ST(S - h, t)) / (h * h)


def fd_ut(predict_ST, S, t, h=1e-4):
    return (predict_ST(S, t + h) - predict_ST(S, t - h)) / (2.0 * h)


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
    Interior collocation points:
      t ~ Unif(0,T)
      log S ~ Unif(log S_min, log S_max)
    Returns X_pde with columns [S, t].
    """
    t = rng.uniform(0.0, p.T, size=n)
    logS = rng.uniform(np.log(p.S_min), np.log(p.S_max), size=n)
    S = np.exp(logS)
    return np.column_stack([S, t])


def sample_terminal(n: int, p: BSParams, rng: np.random.Generator) -> np.ndarray:
    """
    Terminal points (boundary/terminal condition):
      t = T
      log S ~ Unif(log S_min, log S_max)
    Returns X_bc with columns [S, t].
    """
    logS = rng.uniform(np.log(p.S_min), np.log(p.S_max), size=n)
    S = np.exp(logS)
    t = np.full(n, p.T, dtype=float)
    return np.column_stack([S, t])


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
    Terminal condition for European call:
      u(S,T) = max(S-K, 0)
    """
    def bc_target_fn(X_bc: np.ndarray) -> np.ndarray:
        S = X_bc[:, 0]
        return np.maximum(S - p.K, 0.0)
    return bc_target_fn


def make_pde_residual_fn(p: BSParams):
    """
    Black–Scholes PDE operator:
      L u = u_t + 0.5*sigma^2*S^2*u_SS + r*S*u_S - r*u
    Residual defined as L f(S,t) (should be 0).
    We approximate derivatives via finite differences on the model predictor.
    """
    def pde_residual_fn(predict_fn, X_pde: np.ndarray) -> np.ndarray:
        S = X_pde[:, 0]
        t = X_pde[:, 1]

        # wrap predict_fn(X) into predict_ST(S,t)
        def predict_ST(Sv, tv):
            X = np.column_stack([np.asarray(Sv, float).reshape(-1), np.asarray(tv, float).reshape(-1)])
            return np.asarray(predict_fn(X), float).reshape(-1)

        ut = fd_ut(predict_ST, S, t, h=1e-4)
        uS = fd_uS(predict_ST, S, t, h=1e-2)
        uSS = fd_uSS(predict_ST, S, t, h=1e-2)
        u = predict_ST(S, t)

        return ut + 0.5 * (p.sigma ** 2) * (S ** 2) * uSS + p.r * S * uS - p.r * u

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
    ap.add_argument("--tree_depth", type=int, default=3)
    ap.add_argument("--tree_leaf", type=int, default=30)
    ap.add_argument("--nu_int", type=float, default=0.1)
    ap.add_argument("--nu_bdry", type=float, default=0.1)

    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "figures"), exist_ok=True)

    rng = np.random.default_rng(args.seed)
    p = BSParams()

    # Data
    X_pde = sample_interior(args.n_pde, p, rng)
    X_bc = sample_terminal(args.n_bc, p, rng)

    S_te, t_te = sample_test(args.n_test, p, rng)
    tau_te = p.T - t_te
    u_true = bs_call_price(S_te, p.K, p.r, p.sigma, tau_te)
    d_true = bs_call_delta(S_te, p.K, p.r, p.sigma, tau_te)

    X_te = np.column_stack([S_te, t_te])

    bc_target_fn = make_bc_target_fn(p)
    pde_residual_fn = make_pde_residual_fn(p)

    # Learner factories (mixed learners exactly as your algorithm)
    def interior_factory():
        fmap = RBFMap(n_centers=args.rbf_centers, sigma=args.rbf_sigma, add_bias=True, standardize=True, center_seed=args.seed)
        return RidgeBasisLearner(feature_map=fmap, lam=args.ridge_lam)

    def boundary_factory():
        return TreeLearner(max_depth=args.tree_depth, min_samples_leaf=args.tree_leaf, random_state=args.seed)

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

    # Evaluation callback per round
    def eval_callback(b: int, mdl: PIGB):
        # price prediction
        u_pred = mdl.predict(X_te)

        # delta prediction by finite diff on model predict
        def predict_ST(Sv, tv):
            X = np.column_stack([np.asarray(Sv, float).reshape(-1), np.asarray(tv, float).reshape(-1)])
            return np.asarray(mdl.predict(X), float).reshape(-1)

        delta_pred = fd_uS(predict_ST, S_te, t_te, h=1e-2)

        rmse_u = float(np.sqrt(np.mean((u_pred - u_true) ** 2)))
        rmse_d = float(np.sqrt(np.mean((delta_pred - d_true) ** 2)))

        # probe PDE residual (keep small for speed)
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
            print(f"[eval {b:4d}] rmse_u={rmse_u:.6g}  rmse_delta={rmse_d:.6g}  res_rms={res_rms:.6g}  time={elapsed:.2f}s")

    # Train
    model.fit(X_pde=X_pde, X_bc=X_bc, eval_callback=eval_callback)

    # Save metrics
    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.outdir, "bs1d_metrics.csv")
    df.to_csv(csv_path, index=False)
    print("Saved:", csv_path)

    # Plots
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
        print("Saved:", out)

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
    print("Saved:", out)


if __name__ == "__main__":
    main()
