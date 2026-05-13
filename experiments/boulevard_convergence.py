"""
Exact full-basis Boulevard convergence experiment.

This script verifies the finite-dimensional characterization:

    theta_b -> {nu / (nu + 1)} theta_erm,

so that the rescaled coefficient

    ((nu + 1) / nu) theta_b

converges to the once-for-all finite-basis ERM coefficient.

Outputs:
    results/tables/boulevard_convergence.csv
    results/tables/boulevard_convergence_summary.csv
    results/figures/boulevard_convergence_rescaled.png
    results/figures/boulevard_convergence_rescaled.pdf
    results/figures/boulevard_convergence_diagnostics.png
    results/figures/boulevard_convergence_diagnostics.pdf

Run from repo root:
    python experiments/boulevard_convergence.py
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results" / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "results" / ".cache"))

import matplotlib.pyplot as plt


from src.baseline import transport_call_baseline, transport_call_pde_residual
from src.black_scholes import bs_call_price
from src.basis import make_rbf_centers, rbf_features
from src.estimator import fit_finite_basis
from src.metrics import rmse
from src.utils import ensure_dir, print_section, save_dataframe, set_seed


# ============================================================
# Global settings
# ============================================================

SEED = 123

S_MIN = 40.0
S_MAX = 160.0

T_MATURITY = 1.0

K = 100.0
R = 0.05
SIGMA = 0.2

N_TRAIN = 1000
NOISE_SD = 0.05
T_MAX_SAMPLE = 0.95 * T_MATURITY

N_BASIS_S = 14
N_BASIS_T = 8
RBF_BANDWIDTH = 0.30
BASELINE_SMOOTHNESS = 15.0

LAMBDA_PDE = 1e-3
LAMBDA_RIDGE = 0.0

NU = 0.5
N_ITERATIONS = 600

TABLE_DIR = ROOT / "results" / "tables"
FIG_DIR = ROOT / "results" / "figures"


# ============================================================
# Data and finite-basis helpers
# ============================================================

def sample_design(n, t_max):
    S = np.random.uniform(S_MIN, S_MAX, size=n)
    t = np.random.uniform(0.0, t_max, size=n)
    return S, t


def make_centers():
    return make_rbf_centers(
        S_min=S_MIN,
        S_max=S_MAX,
        t_min=0.0,
        t_max=T_MATURITY,
        n_s=N_BASIS_S,
        n_t=N_BASIS_T,
        K=K,
        T=T_MATURITY,
    )


def get_baseline(S, t):
    baseline = transport_call_baseline(
        S=S,
        t=t,
        K=K,
        r=R,
        T=T_MATURITY,
        smoothness=BASELINE_SMOOTHNESS,
    )
    baseline_pde = transport_call_pde_residual(
        S=S,
        t=t,
        K=K,
        r=R,
        sigma=SIGMA,
        T=T_MATURITY,
        smoothness=BASELINE_SMOOTHNESS,
    )
    return baseline, baseline_pde


def fit_call_erm(n_train, seed):
    set_seed(seed)

    centers = make_centers()

    S_train, t_train = sample_design(n_train, T_MAX_SAMPLE)
    true_train = bs_call_price(
        S_train,
        t_train,
        K=K,
        r=R,
        sigma=SIGMA,
        T=T_MATURITY,
    )
    y_train = true_train + NOISE_SD * np.random.randn(n_train)

    baseline_train, baseline_pde_train = get_baseline(S_train, t_train)

    fit = fit_finite_basis(
        S_train=S_train,
        t_train=t_train,
        y_train=y_train,
        baseline_train=baseline_train,
        baseline_pde_train=baseline_pde_train,
        centers=centers,
        bandwidth=RBF_BANDWIDTH,
        K=K,
        r=R,
        sigma=SIGMA,
        T=T_MATURITY,
        lambda_pde=LAMBDA_PDE,
        lambda_ridge=LAMBDA_RIDGE,
    )

    return {
        "centers": centers,
        "S_train": S_train,
        "t_train": t_train,
        "y_train": y_train,
        "baseline_train": baseline_train,
        "baseline_pde_train": baseline_pde_train,
        "fit": fit,
    }


def objective(theta, experiment):
    fit = experiment["fit"]

    theta = np.asarray(theta, dtype=float).ravel()
    y = experiment["y_train"]
    baseline = experiment["baseline_train"]
    baseline_pde = experiment["baseline_pde_train"]
    Psi = fit["Psi_train"]
    R_mat = fit["R_train"]

    price_residual = y - baseline - Psi @ theta
    pde_residual = baseline_pde + R_mat @ theta

    value = 0.5 * np.sum(price_residual ** 2)
    value += 0.5 * LAMBDA_PDE * np.sum(pde_residual ** 2)
    value += 0.5 * LAMBDA_RIDGE * np.sum(theta ** 2)

    return float(value)


def make_evaluation_design(experiment, n_s=100, n_t=70):
    S_grid = np.linspace(S_MIN, S_MAX, n_s)
    t_grid = np.linspace(0.0, T_MAX_SAMPLE, n_t)
    S_mesh, t_mesh = np.meshgrid(S_grid, t_grid, indexing="xy")
    S_eval = S_mesh.ravel()
    t_eval = t_mesh.ravel()
    Psi_eval = rbf_features(
        S=S_eval,
        t=t_eval,
        centers=experiment["centers"],
        bandwidth=RBF_BANDWIDTH,
        K=K,
        T=T_MATURITY,
    )

    return Psi_eval


def surface_rmse_between_coefficients(theta, target_theta, Psi_eval):
    prediction_difference = Psi_eval @ (theta - target_theta)
    return rmse(prediction_difference, np.zeros_like(prediction_difference))


# ============================================================
# Boulevard recursion
# ============================================================

def run_boulevard_recursion(experiment, nu, n_iterations):
    theta_erm = experiment["fit"]["theta"]
    theta_star = (nu / (nu + 1.0)) * theta_erm

    theta_b = np.zeros_like(theta_erm)
    theta_erm_norm = np.linalg.norm(theta_erm)
    theta_star_norm = np.linalg.norm(theta_star)
    objective_erm = objective(theta_erm, experiment)

    Psi_eval = make_evaluation_design(experiment)

    records = []

    for iteration in range(n_iterations + 1):
        theta_rescaled = ((nu + 1.0) / nu) * theta_b

        raw_error = np.linalg.norm(theta_b - theta_star)
        rescaled_error = np.linalg.norm(theta_rescaled - theta_erm)

        if theta_star_norm > 0:
            raw_relative_error = raw_error / theta_star_norm
        else:
            raw_relative_error = np.nan

        if theta_erm_norm > 0:
            rescaled_relative_error = rescaled_error / theta_erm_norm
        else:
            rescaled_relative_error = np.nan

        raw_surface_rmse = surface_rmse_between_coefficients(
            theta=theta_b,
            target_theta=theta_star,
            Psi_eval=Psi_eval,
        )
        rescaled_surface_rmse = surface_rmse_between_coefficients(
            theta=theta_rescaled,
            target_theta=theta_erm,
            Psi_eval=Psi_eval,
        )

        objective_rescaled = objective(theta_rescaled, experiment)
        objective_gap = objective_rescaled - objective_erm

        if iteration > 0:
            rate_scaled_raw_error = raw_error * (iteration ** (nu + 1.0))
        else:
            rate_scaled_raw_error = np.nan

        records.append(
            {
                "iteration": iteration,
                "nu": nu,
                "n_basis": theta_erm.size,
                "lambda_pde": LAMBDA_PDE,
                "lambda_ridge": LAMBDA_RIDGE,
                "theta_erm_norm": theta_erm_norm,
                "theta_star_norm": theta_star_norm,
                "theta_b_norm": np.linalg.norm(theta_b),
                "theta_rescaled_norm": np.linalg.norm(theta_rescaled),
                "raw_error_to_compressed_erm": raw_error,
                "raw_relative_error_to_compressed_erm": raw_relative_error,
                "raw_surface_rmse_to_compressed_erm": raw_surface_rmse,
                "rescaled_error_to_erm": rescaled_error,
                "rescaled_relative_error_to_erm": rescaled_relative_error,
                "rescaled_surface_rmse_to_erm": rescaled_surface_rmse,
                "objective_erm": objective_erm,
                "objective_rescaled": objective_rescaled,
                "objective_gap_rescaled_to_erm": objective_gap,
                "rate_scaled_raw_error": rate_scaled_raw_error,
            }
        )

        if iteration == n_iterations:
            break

        stage_theta = theta_erm - theta_b
        theta_b = theta_b + (nu * stage_theta - theta_b) / (iteration + 1.0)

    return pd.DataFrame(records)


def summarize_convergence(table):
    final = table.iloc[-1]
    positive = table[table["iteration"] > 0].copy()

    if len(positive) >= 10:
        tail = positive.tail(max(10, len(positive) // 5))
        x = np.log(tail["iteration"].to_numpy())
        y = np.log(tail["raw_error_to_compressed_erm"].to_numpy())
        empirical_loglog_slope = float(np.polyfit(x, y, deg=1)[0])
    else:
        empirical_loglog_slope = np.nan

    summary = pd.DataFrame(
        [
            {
                "nu": final["nu"],
                "iterations": int(final["iteration"]),
                "n_basis": int(final["n_basis"]),
                "lambda_pde": final["lambda_pde"],
                "lambda_ridge": final["lambda_ridge"],
                "theta_erm_norm": final["theta_erm_norm"],
                "theta_star_norm": final["theta_star_norm"],
                "final_raw_relative_error_to_compressed_erm": (
                    final["raw_relative_error_to_compressed_erm"]
                ),
                "final_rescaled_relative_error_to_erm": (
                    final["rescaled_relative_error_to_erm"]
                ),
                "final_rescaled_surface_rmse_to_erm": (
                    final["rescaled_surface_rmse_to_erm"]
                ),
                "final_objective_gap_rescaled_to_erm": (
                    final["objective_gap_rescaled_to_erm"]
                ),
                "theoretical_loglog_slope": -(final["nu"] + 1.0),
                "empirical_loglog_slope": empirical_loglog_slope,
            }
        ]
    )

    return summary


# ============================================================
# Plots
# ============================================================

def save_figure(fig, base_name):
    ensure_dir(FIG_DIR)

    png_path = FIG_DIR / f"{base_name}.png"
    pdf_path = FIG_DIR / f"{base_name}.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return [str(png_path), str(pdf_path)]


def plot_rescaled_convergence(table):
    plot_table = table[table["iteration"] > 0].copy()
    final = plot_table.iloc[-1]

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.loglog(
        plot_table["iteration"],
        plot_table["rescaled_relative_error_to_erm"],
        linewidth=2.2,
        marker="o",
        markersize=3,
        markevery=max(len(plot_table) // 25, 1),
        color="#355C7D",
    )

    ax.set_xlabel("Boulevard iteration")
    ax.set_ylabel("Relative coefficient error")
    ax.set_title("ERM-scale convergence of exact full-basis Boulevard")
    ax.grid(alpha=0.25, which="both")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.text(
        0.97,
        0.95,
        (
            r"$((\nu+1)/\nu)\theta_b \to \widehat{\theta}_{\rm ERM}$"
            "\n"
            f"final relative error = {final['rescaled_relative_error_to_erm']:.2e}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
    )

    fig.tight_layout()

    return save_figure(fig, "boulevard_convergence_rescaled")


def plot_diagnostics(table):
    plot_table = table[table["iteration"] > 0].copy()
    objective_gap = np.maximum(
        plot_table["objective_gap_rescaled_to_erm"].to_numpy(),
        1e-18,
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.loglog(
        plot_table["iteration"],
        plot_table["rescaled_surface_rmse_to_erm"],
        linewidth=2.2,
        marker="o",
        markersize=3,
        markevery=max(len(plot_table) // 25, 1),
        color="#2A9D8F",
    )
    ax.set_xlabel("Boulevard iteration")
    ax.set_ylabel("Surface RMSE to ERM predictor")
    ax.set_title("Prediction-level convergence")
    ax.grid(alpha=0.25, which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax = axes[1]
    ax.loglog(
        plot_table["iteration"],
        objective_gap,
        linewidth=2.2,
        marker="o",
        markersize=3,
        markevery=max(len(plot_table) // 25, 1),
        color="#B56576",
    )
    ax.set_xlabel("Boulevard iteration")
    ax.set_ylabel("Objective gap to ERM")
    ax.set_title("Objective-level convergence")
    ax.grid(alpha=0.25, which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Exact full-basis Boulevard convergence diagnostics",
        y=1.02,
        fontsize=13,
    )
    fig.tight_layout()

    return save_figure(fig, "boulevard_convergence_diagnostics")


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run exact full-basis Boulevard convergence diagnostics."
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-train", type=int, default=N_TRAIN)
    parser.add_argument("--nu", type=float, default=NU)
    parser.add_argument("--iterations", type=int, default=N_ITERATIONS)
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Only write tables, without regenerating figures.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    ensure_dir(TABLE_DIR)
    ensure_dir(FIG_DIR)

    print_section("Exact full-basis Boulevard convergence")
    print(f"seed: {args.seed}")
    print(f"training points: {args.n_train}")
    print(f"basis terms: {N_BASIS_S * N_BASIS_T}")
    print(f"lambda_pde: {LAMBDA_PDE}")
    print(f"nu: {args.nu}")
    print(f"iterations: {args.iterations}")

    experiment = fit_call_erm(n_train=args.n_train, seed=args.seed)
    table = run_boulevard_recursion(
        experiment=experiment,
        nu=args.nu,
        n_iterations=args.iterations,
    )
    summary = summarize_convergence(table)

    table_path = TABLE_DIR / "boulevard_convergence.csv"
    summary_path = TABLE_DIR / "boulevard_convergence_summary.csv"

    save_dataframe(table, table_path, index=False)
    save_dataframe(summary, summary_path, index=False)

    print("\nSaved tables:")
    print(table_path)
    print(summary_path)

    print("\nConvergence summary:")
    print(summary.to_string(index=False))

    if not args.skip_plots:
        saved = []
        saved.extend(plot_rescaled_convergence(table))
        saved.extend(plot_diagnostics(table))

        print("\nSaved figures:")
        for path in saved:
            print(path)


if __name__ == "__main__":
    main()
