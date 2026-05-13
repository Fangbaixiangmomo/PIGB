"""
Consolidated main finite-basis experiments.

This script runs the main Black-Scholes call and digital-call experiments,
then produces thesis-oriented diagnostics:

1. Existing per-option result tables:
       results/tables/call_main.csv
       results/tables/digital_main.csv
2. A combined result table:
       results/tables/main_finite_basis_combined.csv
3. A normalized multi-panel comparison figure:
       results/figures/main_finite_basis_normalized.{png,pdf}
4. A digital-call spatial error diagnostic:
       results/raw/digital_main_error_surface.csv
       results/figures/digital_main_error_diagnostic.{png,pdf}
5. Optional repeated-seed robustness tables:
       results/raw/main_finite_basis_robustness_raw.csv
       results/tables/main_finite_basis_robustness.csv

Run from repo root:
    python experiments/main_finite_basis.py

For a quicker run without robustness:
    python experiments/main_finite_basis.py --skip-robustness
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

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter


from src.black_scholes import (
    bs_call_delta,
    bs_call_gamma,
    bs_call_price,
    bs_digital_delta,
    bs_digital_gamma,
    bs_digital_price,
)
from src.baseline import (
    transport_call_baseline,
    transport_call_delta,
    transport_call_gamma,
    transport_call_pde_residual,
    transport_digital_baseline,
    transport_digital_delta,
    transport_digital_gamma,
    transport_digital_pde_residual,
)
from src.basis import make_rbf_centers
from src.estimator import (
    fit_finite_basis,
    pde_residual,
    predict_delta,
    predict_gamma,
    predict_price,
)
from src.metrics import pde_rmse, rmse
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

LAMBDA_RIDGE = 0.0

TABLE_DIR = "results/tables"
FIG_DIR = "results/figures"
RAW_DIR = "results/raw"

ACCURACY_METRICS = [
    "price_rmse",
    "delta_rmse",
    "gamma_rmse",
]

SUMMARY_METRICS = [
    "price_rmse",
    "delta_rmse",
    "gamma_rmse",
    "pde_rmse",
]

METRIC_LABELS = {
    "price_rmse": "Price",
    "delta_rmse": "Delta",
    "gamma_rmse": "Gamma",
    "pde_rmse": "PDE residual",
}

METHOD_LABELS = {
    "raw_basis": "Raw\nbasis",
    "baseline_alone": "Baseline\nalone",
    "baseline_only": "Baseline\n+ basis",
    "baseline_pde": "Baseline\n+ basis\n+ PDE",
}

OPTION_SETTINGS = {
    "call": {
        "label": "Call",
        "n_train": 800,
        "n_test": 5000,
        "noise_sd": 0.05,
        "t_max_sample": 0.95 * T_MATURITY,
        "n_basis_s": 8,
        "n_basis_t": 5,
        "rbf_bandwidth": 0.50,
        "baseline_smoothness": 25.0,
        "lambda_pde": 1e-3,
        "methods": [
            {
                "method": "raw_basis",
                "use_baseline": False,
                "fit_correction": True,
                "lambda_pde": 0.0,
            },
            {
                "method": "baseline_only",
                "use_baseline": True,
                "fit_correction": True,
                "lambda_pde": 0.0,
            },
            {
                "method": "baseline_pde",
                "use_baseline": True,
                "fit_correction": True,
                "lambda_pde": 1e-3,
            },
        ],
    },
    "digital": {
        "label": "Digital call",
        "n_train": 1000,
        "n_test": 5000,
        "noise_sd": 0.02,
        "t_max_sample": 0.90 * T_MATURITY,
        "n_basis_s": 14,
        "n_basis_t": 8,
        "rbf_bandwidth": 0.35,
        "baseline_smoothness": 8.0,
        "lambda_pde": 0.1,
        "methods": [
            {
                "method": "raw_basis",
                "use_baseline": False,
                "fit_correction": True,
                "lambda_pde": 0.0,
            },
            {
                "method": "baseline_alone",
                "use_baseline": True,
                "fit_correction": False,
                "lambda_pde": 0.0,
            },
            {
                "method": "baseline_only",
                "use_baseline": True,
                "fit_correction": True,
                "lambda_pde": 0.0,
            },
            {
                "method": "baseline_pde",
                "use_baseline": True,
                "fit_correction": True,
                "lambda_pde": 0.1,
            },
        ],
    },
}


CALL_TABLE_COLUMNS = [
    "method",
    "use_baseline",
    "lambda_pde",
    "price_rmse",
    "delta_rmse",
    "gamma_rmse",
    "pde_rmse",
    "train_price_rmse_noisy",
    "theta_norm",
    "augmented_condition_number",
]

DIGITAL_TABLE_COLUMNS = [
    "method",
    "use_baseline",
    "fit_correction",
    "lambda_pde",
    "price_rmse",
    "delta_rmse",
    "gamma_rmse",
    "pde_rmse",
    "train_price_rmse_noisy",
    "theta_norm",
    "augmented_condition_number",
]


# ============================================================
# Data and model helpers
# ============================================================

def sample_design(n, t_max):
    """
    Sample state-time design points.
    """
    S = np.random.uniform(S_MIN, S_MAX, size=n)
    t = np.random.uniform(0.0, t_max, size=n)
    return S, t


def zero_like(S):
    """
    Zero baseline and zero baseline derivatives/residual.
    """
    return np.zeros_like(S, dtype=float)


def get_truth(option_type, S, t):
    """
    Return true price, Delta, and Gamma for the selected option.
    """
    if option_type == "call":
        return {
            "price": bs_call_price(S, t, K=K, r=R, sigma=SIGMA, T=T_MATURITY),
            "delta": bs_call_delta(S, t, K=K, r=R, sigma=SIGMA, T=T_MATURITY),
            "gamma": bs_call_gamma(S, t, K=K, r=R, sigma=SIGMA, T=T_MATURITY),
        }

    if option_type == "digital":
        return {
            "price": bs_digital_price(S, t, K=K, r=R, sigma=SIGMA, T=T_MATURITY),
            "delta": bs_digital_delta(S, t, K=K, r=R, sigma=SIGMA, T=T_MATURITY),
            "gamma": bs_digital_gamma(S, t, K=K, r=R, sigma=SIGMA, T=T_MATURITY),
        }

    raise ValueError(f"Unknown option_type: {option_type}")


def get_baseline(option_type, S, t, settings):
    """
    Return baseline value, derivatives, and PDE residual.
    """
    smoothness = settings["baseline_smoothness"]

    if option_type == "call":
        return {
            "price": transport_call_baseline(
                S, t, K=K, r=R, T=T_MATURITY, smoothness=smoothness
            ),
            "delta": transport_call_delta(
                S, t, K=K, r=R, T=T_MATURITY, smoothness=smoothness
            ),
            "gamma": transport_call_gamma(
                S, t, K=K, r=R, T=T_MATURITY, smoothness=smoothness
            ),
            "pde": transport_call_pde_residual(
                S, t, K=K, r=R, sigma=SIGMA, T=T_MATURITY, smoothness=smoothness
            ),
        }

    if option_type == "digital":
        return {
            "price": transport_digital_baseline(
                S, t, K=K, r=R, T=T_MATURITY, smoothness=smoothness
            ),
            "delta": transport_digital_delta(
                S, t, K=K, r=R, T=T_MATURITY, smoothness=smoothness
            ),
            "gamma": transport_digital_gamma(
                S, t, K=K, r=R, T=T_MATURITY, smoothness=smoothness
            ),
            "pde": transport_digital_pde_residual(
                S, t, K=K, r=R, sigma=SIGMA, T=T_MATURITY, smoothness=smoothness
            ),
        }

    raise ValueError(f"Unknown option_type: {option_type}")


def make_option_centers(settings):
    """
    Build RBF centers for an option-specific setting.
    """
    return make_rbf_centers(
        S_min=S_MIN,
        S_max=S_MAX,
        t_min=0.0,
        t_max=T_MATURITY,
        n_s=settings["n_basis_s"],
        n_t=settings["n_basis_t"],
        K=K,
        T=T_MATURITY,
    )


def make_experiment_data(option_type, seed, n_test=None):
    """
    Generate one train/test split for the selected option.
    """
    settings = OPTION_SETTINGS[option_type]

    set_seed(seed)

    n_train = settings["n_train"]
    n_test = settings["n_test"] if n_test is None else n_test

    S_train, t_train = sample_design(n_train, settings["t_max_sample"])
    truth_train = get_truth(option_type, S_train, t_train)
    y_train = truth_train["price"] + settings["noise_sd"] * np.random.randn(n_train)

    S_test, t_test = sample_design(n_test, settings["t_max_sample"])
    truth_test = get_truth(option_type, S_test, t_test)

    return {
        "S_train": S_train,
        "t_train": t_train,
        "y_train": y_train,
        "S_test": S_test,
        "t_test": t_test,
        "truth_test": truth_test,
    }


def prepare_baselines(option_type, method, data, settings):
    """
    Build train/test baselines for one method.
    """
    if method["use_baseline"]:
        baseline_train = get_baseline(
            option_type,
            data["S_train"],
            data["t_train"],
            settings,
        )
        baseline_test = get_baseline(
            option_type,
            data["S_test"],
            data["t_test"],
            settings,
        )
    else:
        baseline_train = {
            "price": zero_like(data["S_train"]),
            "delta": zero_like(data["S_train"]),
            "gamma": zero_like(data["S_train"]),
            "pde": zero_like(data["S_train"]),
        }
        baseline_test = {
            "price": zero_like(data["S_test"]),
            "delta": zero_like(data["S_test"]),
            "gamma": zero_like(data["S_test"]),
            "pde": zero_like(data["S_test"]),
        }

    return baseline_train, baseline_test


def evaluate_method(
    option_type,
    method,
    data,
    centers,
    compute_condition=True,
    return_predictions=False,
):
    """
    Fit one method and compute test metrics.
    """
    settings = OPTION_SETTINGS[option_type]
    bandwidth = settings["rbf_bandwidth"]

    baseline_train, baseline_test = prepare_baselines(
        option_type,
        method,
        data,
        settings,
    )

    if method["fit_correction"]:
        fit = fit_finite_basis(
            S_train=data["S_train"],
            t_train=data["t_train"],
            y_train=data["y_train"],
            baseline_train=baseline_train["price"],
            baseline_pde_train=baseline_train["pde"],
            centers=centers,
            bandwidth=bandwidth,
            K=K,
            r=R,
            sigma=SIGMA,
            T=T_MATURITY,
            lambda_pde=method["lambda_pde"],
            lambda_ridge=LAMBDA_RIDGE,
        )

        theta = fit["theta"]
        train_price_rmse_noisy = rmse(fit["fitted_price_train"], data["y_train"])
        theta_norm = np.linalg.norm(theta)
        augmented_condition_number = (
            np.linalg.cond(fit["augmented_design"]) if compute_condition else np.nan
        )

        pred_price = predict_price(
            S=data["S_test"],
            t=data["t_test"],
            baseline=baseline_test["price"],
            theta=theta,
            centers=centers,
            bandwidth=bandwidth,
            K=K,
            T=T_MATURITY,
        )

        pred_delta = predict_delta(
            S=data["S_test"],
            t=data["t_test"],
            baseline_delta=baseline_test["delta"],
            theta=theta,
            centers=centers,
            bandwidth=bandwidth,
            K=K,
            T=T_MATURITY,
        )

        pred_gamma = predict_gamma(
            S=data["S_test"],
            t=data["t_test"],
            baseline_gamma=baseline_test["gamma"],
            theta=theta,
            centers=centers,
            bandwidth=bandwidth,
            K=K,
            T=T_MATURITY,
        )

        fitted_pde = pde_residual(
            S=data["S_test"],
            t=data["t_test"],
            baseline_pde=baseline_test["pde"],
            theta=theta,
            centers=centers,
            bandwidth=bandwidth,
            K=K,
            r=R,
            sigma=SIGMA,
            T=T_MATURITY,
        )
    else:
        pred_price = baseline_test["price"]
        pred_delta = baseline_test["delta"]
        pred_gamma = baseline_test["gamma"]
        fitted_pde = baseline_test["pde"]

        train_price_rmse_noisy = rmse(baseline_train["price"], data["y_train"])
        theta_norm = 0.0
        augmented_condition_number = np.nan

    truth = data["truth_test"]

    row = {
        "method": method["method"],
        "use_baseline": method["use_baseline"],
        "fit_correction": method["fit_correction"],
        "lambda_pde": method["lambda_pde"],
        "price_rmse": rmse(pred_price, truth["price"]),
        "delta_rmse": rmse(pred_delta, truth["delta"]),
        "gamma_rmse": rmse(pred_gamma, truth["gamma"]),
        "pde_rmse": pde_rmse(fitted_pde),
        "train_price_rmse_noisy": train_price_rmse_noisy,
        "theta_norm": theta_norm,
        "augmented_condition_number": augmented_condition_number,
    }

    predictions = None
    if return_predictions:
        predictions = {
            "pred_price": pred_price,
            "pred_delta": pred_delta,
            "pred_gamma": pred_gamma,
            "fitted_pde": fitted_pde,
        }

    return row, predictions


def run_option_experiment(
    option_type,
    seed=SEED,
    n_test=None,
    compute_condition=True,
    return_predictions=False,
):
    """
    Run all methods for one option type.
    """
    settings = OPTION_SETTINGS[option_type]
    data = make_experiment_data(option_type, seed=seed, n_test=n_test)
    centers = make_option_centers(settings)

    rows = []
    predictions = {}

    for method in settings["methods"]:
        row, method_predictions = evaluate_method(
            option_type=option_type,
            method=method,
            data=data,
            centers=centers,
            compute_condition=compute_condition,
            return_predictions=return_predictions,
        )

        rows.append(row)

        if return_predictions:
            predictions[method["method"]] = method_predictions

    table = pd.DataFrame(rows)

    return table, data, predictions


# ============================================================
# Tables
# ============================================================

def write_main_tables(call_table, digital_table):
    """
    Write per-option and combined result tables.
    """
    ensure_dir(TABLE_DIR)

    save_dataframe(
        call_table[CALL_TABLE_COLUMNS],
        f"{TABLE_DIR}/call_main.csv",
        index=False,
    )

    save_dataframe(
        digital_table[DIGITAL_TABLE_COLUMNS],
        f"{TABLE_DIR}/digital_main.csv",
        index=False,
    )

    combined = pd.concat(
        [
            call_table.assign(option_type="call"),
            digital_table.assign(option_type="digital"),
        ],
        ignore_index=True,
    )

    first_cols = ["option_type", "method"]
    other_cols = [c for c in combined.columns if c not in first_cols]
    combined = combined[first_cols + other_cols]

    save_dataframe(
        combined,
        f"{TABLE_DIR}/main_finite_basis_combined.csv",
        index=False,
    )

    return combined


def print_relative_improvements(table, option_type):
    """
    Print relative improvements versus raw_basis.
    """
    raw = table.loc[table["method"] == "raw_basis"].iloc[0]

    rows = []

    for _, row in table.iterrows():
        out = {
            "option_type": option_type,
            "method": row["method"],
        }

        for metric in SUMMARY_METRICS:
            out[f"{metric}_improvement_vs_raw_pct"] = (
                100.0 * (raw[metric] - row[metric]) / raw[metric]
            )

        rows.append(out)

    improvements = pd.DataFrame(rows)
    print(improvements.to_string(index=False))

    return improvements


# ============================================================
# Figures
# ============================================================

def save_figure(fig, base_name):
    """
    Save a figure as PNG and PDF.
    """
    ensure_dir(FIG_DIR)

    png_path = f"{FIG_DIR}/{base_name}.png"
    pdf_path = f"{FIG_DIR}/{base_name}.pdf"

    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    return png_path, pdf_path


def ratio_formatter(value, _position):
    """
    Format log-scale relative-error ticks compactly.
    """
    if value >= 1.0 and abs(value - round(value)) < 1e-10:
        return f"{int(round(value))}x"
    return f"{value:.2g}x"


def get_method_labels(table):
    """
    Return display labels in table order.
    """
    return [METHOD_LABELS[method] for method in table["method"]]


def set_ratio_axis(ax, ylim=None, yticks=None):
    """
    Apply consistent formatting to relative-error axes.
    """
    ax.set_yscale("log")
    if ylim is not None:
        ax.set_ylim(*ylim)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.yaxis.set_major_formatter(FuncFormatter(ratio_formatter))
    ax.yaxis.set_minor_formatter(NullFormatter())


def plot_normalized_accuracy(ax, table, option_title, ylim=None, yticks=None):
    """
    Plot price and Greek RMSE relative to raw_basis.
    """
    raw = table.loc[table["method"] == "raw_basis"].iloc[0]
    x = np.arange(len(table))

    for metric in ACCURACY_METRICS:
        ratio = table[metric].to_numpy() / raw[metric]
        ax.plot(
            x,
            ratio,
            marker="o",
            linewidth=2,
            label=METRIC_LABELS[metric],
        )

    ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(get_method_labels(table))
    set_ratio_axis(ax, ylim=ylim, yticks=yticks)
    ax.set_title(f"{option_title}: price and Greeks")
    ax.set_ylabel("RMSE / raw-basis RMSE")
    ax.legend(frameon=False, loc="best", fontsize=8)
    ax.grid(axis="y", alpha=0.25, which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_normalized_pde(ax, table, option_title, ylim=None, yticks=None):
    """
    Plot PDE residual RMSE relative to raw_basis.
    """
    raw = table.loc[table["method"] == "raw_basis"].iloc[0]
    x = np.arange(len(table))
    ratio = table["pde_rmse"].to_numpy() / raw["pde_rmse"]

    ax.plot(
        x,
        ratio,
        marker="o",
        linewidth=2,
        color="#4c78a8",
    )

    for x_i, y_i in zip(x, ratio):
        ax.text(
            x_i,
            y_i * 1.05,
            f"{y_i:.2f}x",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    ax.axhline(1.0, color="0.35", linestyle="--", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(get_method_labels(table))
    set_ratio_axis(ax, ylim=ylim, yticks=yticks)
    ax.set_title(f"{option_title}: PDE residual")
    ax.set_ylabel("PDE RMSE / raw-basis PDE RMSE")
    ax.grid(axis="y", alpha=0.25, which="both")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_normalized_summary(call_table, digital_table):
    """
    Build the compact replacement for the four old dot plots.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)

    plot_normalized_accuracy(
        axes[0, 0],
        call_table,
        "Call",
        ylim=(0.75, 1.12),
        yticks=[0.8, 0.9, 1.0, 1.1],
    )
    plot_normalized_pde(
        axes[0, 1],
        call_table,
        "Call",
        ylim=(0.55, 1.15),
        yticks=[0.6, 0.7, 0.8, 0.9, 1.0],
    )
    plot_normalized_accuracy(
        axes[1, 0],
        digital_table,
        "Digital call",
        ylim=(0.08, 6.5),
        yticks=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0],
    )
    plot_normalized_pde(
        axes[1, 1],
        digital_table,
        "Digital call",
        ylim=(0.05, 1.4),
        yticks=[0.05, 0.1, 0.25, 0.5, 1.0],
    )

    return save_figure(fig, "main_finite_basis_normalized")


def make_grid_data(option_type, S_grid, t_grid, truth_price):
    """
    Build a data dictionary that reuses train data but evaluates on a grid.
    """
    SS, TT = np.meshgrid(S_grid, t_grid)

    truth = get_truth(option_type, SS.ravel(), TT.ravel())
    truth["price"] = truth_price.ravel()

    return SS, TT, truth


def plot_digital_error_diagnostic(seed=SEED):
    """
    Compare raw-basis and PDE-regularized absolute error over the digital surface.
    """
    option_type = "digital"
    settings = OPTION_SETTINGS[option_type]

    data = make_experiment_data(option_type, seed=seed)
    centers = make_option_centers(settings)

    S_grid = np.linspace(60.0, 140.0, 85)
    t_grid = np.linspace(0.0, settings["t_max_sample"], 70)
    SS, TT = np.meshgrid(S_grid, t_grid)
    truth_grid = get_truth(option_type, SS.ravel(), TT.ravel())

    grid_data = {
        "S_train": data["S_train"],
        "t_train": data["t_train"],
        "y_train": data["y_train"],
        "S_test": SS.ravel(),
        "t_test": TT.ravel(),
        "truth_test": truth_grid,
    }

    selected_methods = {
        method["method"]: method
        for method in settings["methods"]
        if method["method"] in {"raw_basis", "baseline_pde"}
    }

    _, raw_pred = evaluate_method(
        option_type=option_type,
        method=selected_methods["raw_basis"],
        data=grid_data,
        centers=centers,
        compute_condition=False,
        return_predictions=True,
    )
    _, pde_pred = evaluate_method(
        option_type=option_type,
        method=selected_methods["baseline_pde"],
        data=grid_data,
        centers=centers,
        compute_condition=False,
        return_predictions=True,
    )

    true_price = truth_grid["price"]
    raw_error = raw_pred["pred_price"] - true_price
    pde_error = pde_pred["pred_price"] - true_price
    raw_abs_error = np.abs(raw_error).reshape(SS.shape)
    pde_abs_error = np.abs(pde_error).reshape(SS.shape)
    error_reduction = raw_abs_error - pde_abs_error

    surface = pd.DataFrame(
        {
            "S": SS.ravel(),
            "t": TT.ravel(),
            "true_price": true_price,
            "raw_basis_pred": raw_pred["pred_price"],
            "baseline_pde_pred": pde_pred["pred_price"],
            "raw_basis_abs_error": raw_abs_error.ravel(),
            "baseline_pde_abs_error": pde_abs_error.ravel(),
            "abs_error_reduction": error_reduction.ravel(),
        }
    )
    save_dataframe(
        surface,
        f"{RAW_DIR}/digital_main_error_surface.csv",
        index=False,
    )

    vmax = np.quantile(
        np.concatenate([raw_abs_error.ravel(), pde_abs_error.ravel()]),
        0.99,
    )
    vmax = max(vmax, 1e-8)

    diff_max = np.quantile(np.abs(error_reduction.ravel()), 0.99)
    diff_max = max(diff_max, 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)

    plots = [
        (
            axes[0],
            raw_abs_error,
            "Raw basis\nabsolute error",
            "magma",
            mcolors.Normalize(vmin=0.0, vmax=vmax),
        ),
        (
            axes[1],
            pde_abs_error,
            "Baseline + basis + PDE\nabsolute error",
            "magma",
            mcolors.Normalize(vmin=0.0, vmax=vmax),
        ),
        (
            axes[2],
            error_reduction,
            "Error reduction\nraw minus PDE",
            "coolwarm",
            mcolors.TwoSlopeNorm(vmin=-diff_max, vcenter=0.0, vmax=diff_max),
        ),
    ]

    for ax, values, title, cmap, norm in plots:
        im = ax.pcolormesh(
            S_grid,
            t_grid,
            values,
            shading="auto",
            cmap=cmap,
            norm=norm,
        )
        ax.set_title(title)
        ax.set_xlabel("Underlying price S")
        ax.set_ylabel("Time t")
        ax.axvline(K, color="white", linewidth=1.0, linestyle="--", alpha=0.85)
        ax.text(
            K + 1.0,
            t_grid[-1] * 0.96,
            "strike",
            color="white",
            fontsize=8,
            va="top",
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    return save_figure(fig, "digital_main_error_diagnostic")


# ============================================================
# Robustness
# ============================================================

def run_robustness(n_rep):
    """
    Repeat the main experiments over independent seeds.
    """
    if n_rep <= 0:
        return None

    rows = []

    for rep in range(n_rep):
        seed = SEED + rep
        for option_type in ["call", "digital"]:
            table, _, _ = run_option_experiment(
                option_type,
                seed=seed,
                compute_condition=False,
                return_predictions=False,
            )
            table = table.assign(option_type=option_type, rep=rep, seed=seed)
            rows.append(table)

    raw = pd.concat(rows, ignore_index=True)

    save_dataframe(
        raw,
        f"{RAW_DIR}/main_finite_basis_robustness_raw.csv",
        index=False,
    )

    summary = (
        raw.groupby(["option_type", "method"], as_index=False)[SUMMARY_METRICS]
        .agg(["mean", "std"])
    )

    summary.columns = [
        "_".join(col).strip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]

    method_order = {
        "raw_basis": 0,
        "baseline_alone": 1,
        "baseline_only": 2,
        "baseline_pde": 3,
    }
    option_order = {"call": 0, "digital": 1}

    summary["option_order"] = summary["option_type"].map(option_order)
    summary["method_order"] = summary["method"].map(method_order)
    summary = (
        summary.sort_values(["option_order", "method_order"])
        .drop(columns=["option_order", "method_order"])
        .reset_index(drop=True)
    )

    save_dataframe(
        summary,
        f"{TABLE_DIR}/main_finite_basis_robustness.csv",
        index=False,
    )

    return summary


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-robustness",
        action="store_true",
        help="Skip repeated-seed robustness tables.",
    )
    parser.add_argument(
        "--robustness-reps",
        type=int,
        default=25,
        help="Number of seeds for robustness tables.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    ensure_dir(TABLE_DIR)
    ensure_dir(FIG_DIR)
    ensure_dir(RAW_DIR)

    print_section("Main finite-basis experiments")

    call_table, _, _ = run_option_experiment("call")
    digital_table, _, _ = run_option_experiment("digital")

    combined = write_main_tables(call_table, digital_table)

    print_section("Call table")
    print(call_table[CALL_TABLE_COLUMNS].to_string(index=False))

    print_section("Digital table")
    print(digital_table[DIGITAL_TABLE_COLUMNS].to_string(index=False))

    print_section("Relative improvements versus raw basis")
    call_improvements = print_relative_improvements(call_table, "call")
    digital_improvements = print_relative_improvements(digital_table, "digital")
    improvements = pd.concat([call_improvements, digital_improvements], ignore_index=True)
    save_dataframe(
        improvements,
        f"{TABLE_DIR}/main_finite_basis_relative_improvements.csv",
        index=False,
    )

    print_section("Figures")
    saved = []
    saved.extend(plot_normalized_summary(call_table, digital_table))
    saved.extend(plot_digital_error_diagnostic())

    for path in saved:
        print(path)

    if args.skip_robustness:
        print_section("Robustness")
        print("Skipped repeated-seed robustness.")
    else:
        print_section("Robustness")
        robustness = run_robustness(args.robustness_reps)
        print(robustness.to_string(index=False))

    print_section("Saved tables")
    print(f"{TABLE_DIR}/call_main.csv")
    print(f"{TABLE_DIR}/digital_main.csv")
    print(f"{TABLE_DIR}/main_finite_basis_combined.csv")
    print(f"{TABLE_DIR}/main_finite_basis_relative_improvements.csv")
    if not args.skip_robustness:
        print(f"{TABLE_DIR}/main_finite_basis_robustness.csv")
    print(f"{RAW_DIR}/digital_main_error_surface.csv")

    _ = combined


if __name__ == "__main__":
    main()
