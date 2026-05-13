"""
Coverage diagnostics for the thesis coverage subsection.

This script produces the results needed to write the "Coverage and
approximation bias" subsection. It reports pointwise price interval coverage,
not simultaneous surface coverage and not direct coefficient-vector coverage.

Targets:
1. Finite-basis oracle target:
       the fitted value from the same estimator when the training responses
       are noiseless Black-Scholes prices.
2. True Black-Scholes surface:
       the analytical value u*(S, t).

Outputs:
    results/raw/coverage_call_raw.csv
    results/raw/coverage_digital_raw.csv
    results/raw/coverage_local_containment_grid.csv
    results/tables/coverage_call.csv
    results/tables/coverage_digital.csv
    results/tables/coverage_combined.csv
    results/tables/coverage_pointwise_table_rounded.csv
    results/tables/coverage_local_containment_summary.csv
    results/figures/coverage_pointwise_targets.png
    results/figures/coverage_pointwise_targets.pdf
    results/figures/coverage_bias_vs_half_width.png
    results/figures/coverage_bias_vs_half_width.pdf
    results/figures/coverage_local_3d_containment.png
    results/figures/coverage_local_3d_containment.pdf

Run from repo root:
    python experiments/coverage_subsection.py

For a quick plot-only pass using existing coverage tables:
    python experiments/coverage_subsection.py --skip-simulation
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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


from src.black_scholes import bs_call_price, bs_digital_price
from src.baseline import (
    transport_call_baseline,
    transport_call_pde_residual,
    transport_digital_baseline,
    transport_digital_pde_residual,
)
from src.basis import make_rbf_centers, rbf_features
from src.estimator import fit_finite_basis, predict_price
from src.metrics import average_ci_width, rmse
from src.utils import ensure_dir, print_section, save_dataframe, set_seed


# ============================================================
# Global settings
# ============================================================

SEED = 123
VISUAL_SEED = 101

N_REP = 300
N_TRAIN = 1000

S_MIN = 40.0
S_MAX = 160.0

T_MATURITY = 1.0

K = 100.0
R = 0.05
SIGMA = 0.2

LAMBDA_RIDGE = 0.0
Z_VALUE = 1.96

TABLE_DIR = "results/tables"
RAW_DIR = "results/raw"
FIG_DIR = "results/figures"

OPTION_ORDER = ["call", "digital"]
POINT_ORDER = ["low_S_mid_t", "atm_mid_t", "high_S_mid_t"]

OPTION_LABELS = {
    "call": "Call",
    "digital": "Digital call",
}

POINT_LABELS = {
    "low_S_mid_t": r"$S=80$, $t=0.5$",
    "atm_mid_t": r"$S=100$, $t=0.5$",
    "high_S_mid_t": r"$S=120$, $t=0.5$",
}

OPTION_SETTINGS = {
    "call": {
        "noise_sd": 0.05,
        "t_max_sample": 0.95 * T_MATURITY,
        "lambda_pde": 0.001,
        "baseline_smoothness": 15.0,
        "n_basis_s": 14,
        "n_basis_t": 8,
        "rbf_bandwidth": 0.30,
    },
    "digital": {
        "noise_sd": 0.02,
        "t_max_sample": 0.90 * T_MATURITY,
        "lambda_pde": 0.1,
        "baseline_smoothness": 8.0,
        "n_basis_s": 14,
        "n_basis_t": 8,
        "rbf_bandwidth": 0.35,
    },
}

EVAL_POINTS = [
    {"point_name": "low_S_mid_t", "S": 80.0, "t": 0.50},
    {"point_name": "atm_mid_t", "S": 100.0, "t": 0.50},
    {"point_name": "high_S_mid_t", "S": 120.0, "t": 0.50},
]

# Local boxes centered around points in the coverage table. These are narrow
# enough that the confidence band is visible on the price scale.
LOCAL_BOXES = [
    {
        "box_name": "call_atm_mid",
        "option_type": "call",
        "title": "Call near ATM",
        "S_min": 96.0,
        "S_max": 104.0,
        "t_min": 0.45,
        "t_max": 0.55,
    },
    {
        "box_name": "digital_low_tail",
        "option_type": "digital",
        "title": "Digital lower-side tail",
        "S_min": 76.0,
        "S_max": 84.0,
        "t_min": 0.45,
        "t_max": 0.55,
    },
    {
        "box_name": "digital_high_tail",
        "option_type": "digital",
        "title": "Digital upper-side tail",
        "S_min": 116.0,
        "S_max": 124.0,
        "t_min": 0.45,
        "t_max": 0.55,
    },
]


# ============================================================
# Core model helpers
# ============================================================

def sample_design(n, t_max):
    S = np.random.uniform(S_MIN, S_MAX, size=n)
    t = np.random.uniform(0.0, t_max, size=n)
    return S, t


def get_truth(option_type, S, t):
    if option_type == "call":
        return bs_call_price(S, t, K=K, r=R, sigma=SIGMA, T=T_MATURITY)

    if option_type == "digital":
        return bs_digital_price(S, t, K=K, r=R, sigma=SIGMA, T=T_MATURITY)

    raise ValueError(f"Unknown option_type: {option_type}")


def get_baseline(option_type, S, t):
    settings = OPTION_SETTINGS[option_type]
    smoothness = settings["baseline_smoothness"]

    if option_type == "call":
        baseline = transport_call_baseline(
            S, t, K=K, r=R, T=T_MATURITY, smoothness=smoothness
        )
        baseline_pde = transport_call_pde_residual(
            S, t, K=K, r=R, sigma=SIGMA, T=T_MATURITY, smoothness=smoothness
        )
        return baseline, baseline_pde

    if option_type == "digital":
        baseline = transport_digital_baseline(
            S, t, K=K, r=R, T=T_MATURITY, smoothness=smoothness
        )
        baseline_pde = transport_digital_pde_residual(
            S, t, K=K, r=R, sigma=SIGMA, T=T_MATURITY, smoothness=smoothness
        )
        return baseline, baseline_pde

    raise ValueError(f"Unknown option_type: {option_type}")


def make_option_centers(option_type):
    settings = OPTION_SETTINGS[option_type]

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


def fit_once(option_type, S_train, t_train, y_train, centers):
    settings = OPTION_SETTINGS[option_type]

    baseline_train, baseline_pde_train = get_baseline(
        option_type=option_type,
        S=S_train,
        t=t_train,
    )

    return fit_finite_basis(
        S_train=S_train,
        t_train=t_train,
        y_train=y_train,
        baseline_train=baseline_train,
        baseline_pde_train=baseline_pde_train,
        centers=centers,
        bandwidth=settings["rbf_bandwidth"],
        K=K,
        r=R,
        sigma=SIGMA,
        T=T_MATURITY,
        lambda_pde=settings["lambda_pde"],
        lambda_ridge=LAMBDA_RIDGE,
    )


def predict_price_and_se(option_type, fit, centers, S_eval, t_eval):
    """
    Predict prices and known-noise standard errors at evaluation points.
    """
    settings = OPTION_SETTINGS[option_type]

    baseline_eval, _ = get_baseline(
        option_type=option_type,
        S=S_eval,
        t=t_eval,
    )

    pred = predict_price(
        S=S_eval,
        t=t_eval,
        baseline=baseline_eval,
        theta=fit["theta"],
        centers=centers,
        bandwidth=settings["rbf_bandwidth"],
        K=K,
        T=T_MATURITY,
    )

    Psi_eval = rbf_features(
        S=S_eval,
        t=t_eval,
        centers=centers,
        bandwidth=settings["rbf_bandwidth"],
        K=K,
        T=T_MATURITY,
    )

    A = fit["augmented_design"]
    n_train = fit["Psi_train"].shape[0]
    A_pinv = np.linalg.pinv(A, rcond=1e-10)

    # Only the first n_train rows of the augmented response contain quote noise.
    influence = Psi_eval @ A_pinv[:, :n_train]
    variances = settings["noise_sd"] ** 2 * np.sum(influence ** 2, axis=1)
    variances = np.maximum(variances, 0.0)

    return pred, np.sqrt(variances)


# ============================================================
# Coverage simulation
# ============================================================

def run_coverage(option_type, n_rep):
    settings = OPTION_SETTINGS[option_type]
    centers = make_option_centers(option_type)

    S_eval = np.array([p["S"] for p in EVAL_POINTS])
    t_eval = np.array([p["t"] for p in EVAL_POINTS])
    point_names = [p["point_name"] for p in EVAL_POINTS]
    true_eval = get_truth(option_type, S_eval, t_eval)

    records = []

    print_section(f"Pointwise coverage simulation: {option_type}")
    print(f"replications: {n_rep}")
    print(f"training points: {N_TRAIN}")
    print(f"basis terms: {centers.shape[0]}")
    print(f"lambda_pde: {settings['lambda_pde']}")

    for rep in range(n_rep):
        if rep % max(1, n_rep // 10) == 0:
            print(f"  rep {rep + 1}/{n_rep}")

        S_train, t_train = sample_design(N_TRAIN, settings["t_max_sample"])
        true_train = get_truth(option_type, S_train, t_train)
        y_train = true_train + settings["noise_sd"] * np.random.randn(N_TRAIN)

        fit_noisy = fit_once(option_type, S_train, t_train, y_train, centers)
        pred_noisy, se = predict_price_and_se(
            option_type,
            fit_noisy,
            centers,
            S_eval,
            t_eval,
        )

        lower = pred_noisy - Z_VALUE * se
        upper = pred_noisy + Z_VALUE * se

        # Oracle finite-basis target: same estimator, same design, noiseless responses.
        fit_oracle = fit_once(option_type, S_train, t_train, true_train, centers)
        oracle_pred, _ = predict_price_and_se(
            option_type,
            fit_oracle,
            centers,
            S_eval,
            t_eval,
        )

        for j, point_name in enumerate(point_names):
            records.append(
                {
                    "rep": rep,
                    "option_type": option_type,
                    "point_name": point_name,
                    "S": S_eval[j],
                    "t": t_eval[j],
                    "lambda_pde": settings["lambda_pde"],
                    "lambda_ridge": LAMBDA_RIDGE,
                    "n_basis": centers.shape[0],
                    "rbf_bandwidth": settings["rbf_bandwidth"],
                    "baseline_smoothness": settings["baseline_smoothness"],
                    "estimate": pred_noisy[j],
                    "se": se[j],
                    "lower": lower[j],
                    "upper": upper[j],
                    "oracle_target": oracle_pred[j],
                    "true_target": true_eval[j],
                    "covered_oracle": lower[j] <= oracle_pred[j] <= upper[j],
                    "covered_true": lower[j] <= true_eval[j] <= upper[j],
                    "error_oracle": pred_noisy[j] - oracle_pred[j],
                    "error_true": pred_noisy[j] - true_eval[j],
                    "approx_bias": oracle_pred[j] - true_eval[j],
                }
            )

    raw = pd.DataFrame(records)
    summary = summarize_coverage(raw, option_type)

    save_dataframe(raw, f"{RAW_DIR}/coverage_{option_type}_raw.csv", index=False)
    save_dataframe(summary, f"{TABLE_DIR}/coverage_{option_type}.csv", index=False)

    return summary


def summarize_coverage(raw, option_type):
    rows = []

    for point_name, group in raw.groupby("point_name"):
        avg_width = average_ci_width(group["lower"], group["upper"])
        half_width = 0.5 * avg_width
        avg_approx_bias = group["approx_bias"].mean()

        if half_width > 0:
            bias_ratio = abs(avg_approx_bias) / half_width
        else:
            bias_ratio = np.nan

        rows.append(
            {
                "option_type": option_type,
                "point_name": point_name,
                "S": group["S"].iloc[0],
                "t": group["t"].iloc[0],
                "lambda_pde": group["lambda_pde"].iloc[0],
                "lambda_ridge": LAMBDA_RIDGE,
                "n_basis": group["n_basis"].iloc[0],
                "rbf_bandwidth": group["rbf_bandwidth"].iloc[0],
                "baseline_smoothness": group["baseline_smoothness"].iloc[0],
                "n_rep": len(group),
                "coverage_oracle": group["covered_oracle"].mean(),
                "coverage_true": group["covered_true"].mean(),
                "avg_ci_width": avg_width,
                "avg_ci_half_width": half_width,
                "rmse_oracle": rmse(group["estimate"], group["oracle_target"]),
                "rmse_true": rmse(group["estimate"], group["true_target"]),
                "avg_se": group["se"].mean(),
                "sd_error_oracle": group["error_oracle"].std(ddof=1),
                "sd_error_true": group["error_true"].std(ddof=1),
                "bias_oracle": group["error_oracle"].mean(),
                "bias_true": group["error_true"].mean(),
                "approx_bias": avg_approx_bias,
                "abs_approx_bias": abs(avg_approx_bias),
                "abs_approx_bias_over_half_width": bias_ratio,
            }
        )

    out = pd.DataFrame(rows)
    out["point_name"] = pd.Categorical(out["point_name"], POINT_ORDER, ordered=True)
    out = out.sort_values("point_name").reset_index(drop=True)
    out["point_name"] = out["point_name"].astype(str)

    return out


def load_or_run_coverage(skip_simulation, n_rep):
    if skip_simulation:
        path = Path(f"{TABLE_DIR}/coverage_combined.csv")
        if path.exists():
            return pd.read_csv(path)

        raw_paths = {
            option_type: Path(f"{RAW_DIR}/coverage_{option_type}_raw.csv")
            for option_type in OPTION_ORDER
        }

        if all(raw_path.exists() for raw_path in raw_paths.values()):
            summaries = []
            for option_type, raw_path in raw_paths.items():
                raw = pd.read_csv(raw_path)
                summary = summarize_coverage(raw, option_type)
                save_dataframe(
                    summary,
                    f"{TABLE_DIR}/coverage_{option_type}.csv",
                    index=False,
                )
                summaries.append(summary)

            combined = pd.concat(summaries, ignore_index=True)
            combined = order_coverage_table(combined)
            save_dataframe(combined, f"{TABLE_DIR}/coverage_combined.csv", index=False)
            return combined

        raise FileNotFoundError(
            f"Could not find {path} or raw coverage files in {RAW_DIR}. "
            "Run without --skip-simulation first."
        )

    call_summary = run_coverage("call", n_rep=n_rep)
    digital_summary = run_coverage("digital", n_rep=n_rep)

    combined = pd.concat([call_summary, digital_summary], ignore_index=True)
    combined = order_coverage_table(combined)
    save_dataframe(combined, f"{TABLE_DIR}/coverage_combined.csv", index=False)

    return combined


def order_coverage_table(table):
    out = table.copy()
    out["option_type"] = pd.Categorical(out["option_type"], OPTION_ORDER, ordered=True)
    out["point_name"] = pd.Categorical(out["point_name"], POINT_ORDER, ordered=True)
    out = out.sort_values(["option_type", "point_name"]).reset_index(drop=True)
    out["option_type"] = out["option_type"].astype(str)
    out["point_name"] = out["point_name"].astype(str)
    return out


def write_rounded_table(table):
    rows = []

    for _, row in table.iterrows():
        rows.append(
            {
                "Option": OPTION_LABELS[row["option_type"]],
                "Point": f"S={row['S']:.0f}, t={row['t']:.1f}",
                "Finite-basis target coverage": f"{row['coverage_oracle']:.3f}",
                "True-surface coverage": f"{row['coverage_true']:.3f}",
                "Mean CI half-width": f"{row['avg_ci_half_width']:.4f}",
                "Approx. bias / half-width": (
                    f"{row['abs_approx_bias_over_half_width']:.3f}"
                ),
            }
        )

    rounded = pd.DataFrame(rows)
    save_dataframe(
        rounded,
        f"{TABLE_DIR}/coverage_pointwise_table_rounded.csv",
        index=False,
    )
    return rounded


# ============================================================
# Static coverage figures
# ============================================================

def save_figure(fig, base_name):
    png_path = f"{FIG_DIR}/{base_name}.png"
    pdf_path = f"{FIG_DIR}/{base_name}.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return png_path, pdf_path


def make_group_labels(table):
    labels = []
    for _, row in table.iterrows():
        labels.append(
            f"{OPTION_LABELS[row['option_type']]}\n"
            f"{POINT_LABELS[row['point_name']]}"
        )
    return labels


def add_bar_labels(ax, bars, fmt="{:.2f}", y_pad_fraction=0.01):
    y_min, y_max = ax.get_ylim()
    y_pad = y_pad_fraction * (y_max - y_min)

    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + y_pad,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=8,
        )


def plot_coverage_targets(table):
    x = np.arange(len(table))
    width = 0.36

    fig, ax = plt.subplots(figsize=(10, 5))

    oracle_bars = ax.bar(
        x - width / 2,
        table["coverage_oracle"].to_numpy(),
        width,
        label="Finite-basis oracle target",
        color="#4c78a8",
    )
    true_bars = ax.bar(
        x + width / 2,
        table["coverage_true"].to_numpy(),
        width,
        label="True Black-Scholes surface",
        color="#f58518",
    )

    ax.axhline(0.95, linestyle="--", linewidth=1.5, color="0.25", label="Nominal 95%")

    add_bar_labels(ax, oracle_bars, fmt="{:.2f}", y_pad_fraction=0.014)
    add_bar_labels(ax, true_bars, fmt="{:.2f}", y_pad_fraction=0.014)

    ax.set_xticks(x)
    ax.set_xticklabels(make_group_labels(table))
    ax.set_ylim(0.0, 1.08)
    ax.set_ylabel("Pointwise coverage rate")
    ax.set_title("Pointwise price interval coverage")
    ax.legend(frameon=False, loc="lower right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.18)

    fig.tight_layout()

    return save_figure(fig, "coverage_pointwise_targets")


def plot_bias_ratio(table):
    x = np.arange(len(table))

    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.bar(
        x,
        table["abs_approx_bias_over_half_width"].to_numpy(),
        width=0.56,
        color="#4c78a8",
    )

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1.5,
        color="0.25",
        label="Approx. bias = CI half-width",
    )

    add_bar_labels(ax, bars, fmt="{:.2f}", y_pad_fraction=0.014)

    ax.set_xticks(x)
    ax.set_xticklabels(make_group_labels(table))
    ax.set_ylim(0.0, max(1.05, table["abs_approx_bias_over_half_width"].max() * 1.22))
    ax.set_ylabel("Absolute approximation bias / mean CI half-width")
    ax.set_title("Finite-basis approximation bias relative to interval width")
    ax.legend(frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.18)

    fig.tight_layout()

    return save_figure(fig, "coverage_bias_vs_half_width")


# ============================================================
# Local 3D containment diagnostic
# ============================================================

def fit_visual_dataset(option_type, seed):
    settings = OPTION_SETTINGS[option_type]

    set_seed(seed)
    centers = make_option_centers(option_type)

    S_train, t_train = sample_design(N_TRAIN, settings["t_max_sample"])
    true_train = get_truth(option_type, S_train, t_train)
    y_train = true_train + settings["noise_sd"] * np.random.randn(N_TRAIN)

    fit = fit_once(option_type, S_train, t_train, y_train, centers)

    return fit, centers


def evaluate_local_box(box, fit, centers, grid_n):
    option_type = box["option_type"]

    S_grid = np.linspace(box["S_min"], box["S_max"], grid_n)
    t_grid = np.linspace(box["t_min"], box["t_max"], grid_n)
    S_mesh, t_mesh = np.meshgrid(S_grid, t_grid, indexing="ij")

    S_flat = S_mesh.ravel()
    t_flat = t_mesh.ravel()

    true_flat = get_truth(option_type, S_flat, t_flat)
    pred_flat, se_flat = predict_price_and_se(
        option_type,
        fit,
        centers,
        S_flat,
        t_flat,
    )

    half_width_flat = Z_VALUE * se_flat
    lower_flat = pred_flat - half_width_flat
    upper_flat = pred_flat + half_width_flat
    covered_flat = (lower_flat <= true_flat) & (true_flat <= upper_flat)

    values = {
        "S_mesh": S_mesh,
        "t_mesh": t_mesh,
        "true": true_flat.reshape(S_mesh.shape),
        "pred": pred_flat.reshape(S_mesh.shape),
        "lower": lower_flat.reshape(S_mesh.shape),
        "upper": upper_flat.reshape(S_mesh.shape),
        "half_width": half_width_flat.reshape(S_mesh.shape),
        "covered": covered_flat.reshape(S_mesh.shape),
    }

    grid = pd.DataFrame(
        {
            "box_name": box["box_name"],
            "option_type": option_type,
            "S": S_flat,
            "t": t_flat,
            "true_price": true_flat,
            "pred_price": pred_flat,
            "se": se_flat,
            "ci_half_width": half_width_flat,
            "ci_lower": lower_flat,
            "ci_upper": upper_flat,
            "covered_true": covered_flat,
        }
    )

    summary = {
        "box_name": box["box_name"],
        "option_type": option_type,
        "visual_seed": np.nan,
        "S_min": box["S_min"],
        "S_max": box["S_max"],
        "t_min": box["t_min"],
        "t_max": box["t_max"],
        "n_grid": len(grid),
        "grid_true_containment_rate": covered_flat.mean(),
        "mean_ci_half_width": half_width_flat.mean(),
        "mean_abs_error_to_true": np.mean(np.abs(pred_flat - true_flat)),
        "max_abs_error_to_true": np.max(np.abs(pred_flat - true_flat)),
    }

    return values, grid, summary


def configure_3d_axis(ax, title, values):
    ax.set_title(title)
    ax.set_xlabel("Underlying price S")
    ax.set_ylabel("Time t")
    ax.set_zlabel("Price")
    ax.view_init(elev=24, azim=-132)

    z_min = min(
        np.min(values["lower"]),
        np.min(values["true"]),
        np.min(values["pred"]),
    )
    z_max = max(
        np.max(values["upper"]),
        np.max(values["true"]),
        np.max(values["pred"]),
    )
    pad = 0.08 * max(z_max - z_min, 1e-8)
    ax.set_zlim(z_min - pad, z_max + pad)


def plot_local_3d_containment(grid_n, visual_seed):
    """
    Plot localized 3D confidence bands and true-surface containment.
    """
    fits = {}
    centers_by_option = {}

    for option_type in OPTION_ORDER:
        fit, centers = fit_visual_dataset(option_type, seed=visual_seed)
        fits[option_type] = fit
        centers_by_option[option_type] = centers

    fig = plt.figure(figsize=(15, 5), constrained_layout=True)

    all_grid = []
    all_summary = []

    for idx, box in enumerate(LOCAL_BOXES):
        option_type = box["option_type"]
        values, grid, summary = evaluate_local_box(
            box=box,
            fit=fits[option_type],
            centers=centers_by_option[option_type],
            grid_n=grid_n,
        )
        grid["visual_seed"] = visual_seed
        summary["visual_seed"] = visual_seed

        all_grid.append(grid)
        all_summary.append(summary)

        S_mesh = values["S_mesh"]
        t_mesh = values["t_mesh"]
        covered = values["covered"].ravel()
        S_flat = S_mesh.ravel()
        t_flat = t_mesh.ravel()
        true_flat = values["true"].ravel()

        ax = fig.add_subplot(1, len(LOCAL_BOXES), idx + 1, projection="3d")

        # Transparent confidence surfaces.
        ax.plot_surface(
            S_mesh,
            t_mesh,
            values["upper"],
            color="#f58518",
            linewidth=0,
            alpha=0.16,
            shade=False,
        )
        ax.plot_surface(
            S_mesh,
            t_mesh,
            values["lower"],
            color="#f58518",
            linewidth=0,
            alpha=0.16,
            shade=False,
        )

        # Fitted surface and true surface.
        ax.plot_surface(
            S_mesh,
            t_mesh,
            values["pred"],
            color="#4c78a8",
            linewidth=0,
            alpha=0.46,
            shade=False,
        )
        ax.plot_wireframe(
            S_mesh,
            t_mesh,
            values["true"],
            color="black",
            linewidth=0.45,
            alpha=0.58,
        )

        # True grid points, colored by pointwise containment.
        ax.scatter(
            S_flat[covered],
            t_flat[covered],
            true_flat[covered],
            s=9,
            color="#2ca02c",
            alpha=0.18,
            depthshade=False,
        )
        ax.scatter(
            S_flat[~covered],
            t_flat[~covered],
            true_flat[~covered],
            s=10,
            color="#d62728",
            alpha=0.28,
            depthshade=False,
        )

        title = (
            f"{box['title']}\n"
            f"single-fit containment: {summary['grid_true_containment_rate']:.2f}"
        )
        configure_3d_axis(ax, title, values)

    legend_handles = [
        Patch(facecolor="#4c78a8", alpha=0.46, label=r"Fitted surface $\widehat u$"),
        Patch(facecolor="#f58518", alpha=0.16, label="95% CI surfaces"),
        Line2D([0], [0], color="black", linewidth=1.0, label="True surface"),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color="#2ca02c",
            alpha=0.5,
            label="True grid point covered",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color="#d62728",
            alpha=0.6,
            label="True grid point not covered",
        ),
    ]

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 1.06),
    )

    paths = save_figure(fig, "coverage_local_3d_containment")

    grid_out = pd.concat(all_grid, ignore_index=True)
    summary_out = pd.DataFrame(all_summary)

    save_dataframe(
        grid_out,
        f"{RAW_DIR}/coverage_local_containment_grid.csv",
        index=False,
    )
    save_dataframe(
        summary_out,
        f"{TABLE_DIR}/coverage_local_containment_summary.csv",
        index=False,
    )

    return paths, summary_out


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--skip-simulation",
        action="store_true",
        help="Reuse results/tables/coverage_combined.csv instead of rerunning coverage.",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=N_REP,
        help="Number of coverage replications per option.",
    )
    parser.add_argument(
        "--local-grid",
        type=int,
        default=30,
        help="Grid size per axis for each local 3D containment box.",
    )
    parser.add_argument(
        "--visual-seed",
        type=int,
        default=VISUAL_SEED,
        help="Seed for the single-fit local 3D containment diagnostic.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    ensure_dir(TABLE_DIR)
    ensure_dir(RAW_DIR)
    ensure_dir(FIG_DIR)

    set_seed(SEED)

    coverage = load_or_run_coverage(
        skip_simulation=args.skip_simulation,
        n_rep=args.reps,
    )
    coverage = order_coverage_table(coverage)
    save_dataframe(coverage, f"{TABLE_DIR}/coverage_combined.csv", index=False)

    rounded = write_rounded_table(coverage)

    print_section("Coverage table for thesis")
    print(rounded.to_string(index=False))

    print_section("Static figures")
    saved = []
    saved.extend(plot_coverage_targets(coverage))
    saved.extend(plot_bias_ratio(coverage))

    for path in saved:
        print(path)

    print_section("Local 3D containment diagnostic")
    local_paths, local_summary = plot_local_3d_containment(
        grid_n=args.local_grid,
        visual_seed=args.visual_seed,
    )
    print(local_summary.to_string(index=False))
    for path in local_paths:
        print(path)

    print_section("Saved tables")
    print(f"{TABLE_DIR}/coverage_combined.csv")
    print(f"{TABLE_DIR}/coverage_pointwise_table_rounded.csv")
    print(f"{TABLE_DIR}/coverage_local_containment_summary.csv")
    print(f"{RAW_DIR}/coverage_local_containment_grid.csv")


if __name__ == "__main__":
    main()
