"""
Predictive horse-race benchmark.

This script compares the finite-basis estimators against a model-based PDE
oracle and flexible predictive benchmarks on simulated Black-Scholes call and
digital-call data.

The PDE solver is an oracle reference because it uses the true Black-Scholes
parameters. It is not a noisy-data competitor.

Outputs:
    results/tables/benchmark_horserace_single_seed.csv
    results/tables/benchmark_horserace_combined.csv
    results/tables/benchmark_horserace_robustness_raw.csv
    results/tables/benchmark_horserace_robustness.csv
    results/tables/benchmark_horserace_summary.csv
    results/figures/benchmark_horserace_rmse.png
    results/figures/benchmark_horserace_rmse.pdf
    results/figures/benchmark_horserace_accuracy_runtime.png
    results/figures/benchmark_horserace_accuracy_runtime.pdf
    results/figures/benchmark_horserace_robustness.png
    results/figures/benchmark_horserace_robustness.pdf

Run from repo root:
    python experiments/benchmark_horserace.py

Quick smoke test:
    python experiments/benchmark_horserace.py --skip-neural --robustness-reps 2
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results" / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "results" / ".cache"))

import matplotlib.pyplot as plt


from src.baseline import (
    transport_call_baseline,
    transport_call_pde_residual,
    transport_digital_baseline,
    transport_digital_pde_residual,
)
from src.black_scholes import bs_call_price, bs_digital_price
from src.basis import make_rbf_centers
from src.estimator import fit_finite_basis, predict_price
from src.metrics import rmse
from src.utils import ensure_dir, print_section, save_dataframe, set_seed


# ============================================================
# Optional packages
# ============================================================

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from sklearn.ensemble import RandomForestRegressor

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    from xgboost import XGBRegressor

    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from catboost import CatBoostRegressor

    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


# ============================================================
# Global settings
# ============================================================

SEED = 123

N_TRAIN = 1000
N_TEST = 5000
ROBUSTNESS_REPS = 25

S_MIN = 40.0
S_MAX = 160.0

T_MATURITY = 1.0

K = 100.0
R = 0.05
SIGMA = 0.2

LAMBDA_RIDGE = 0.0

# Finite-difference PDE oracle settings.
PDE_S_MAX = 250.0
PDE_N_S = 400
PDE_N_TAU = 400

# Neural-network settings.
NN_HIDDEN = 64
NN_LAYERS = 3
NN_EPOCHS = 2500
NN_LR = 1e-3
NN_WEIGHT_DECAY = 1e-6
PINN_LAMBDA_PDE = 1e-4

TABLE_DIR = ROOT / "results" / "tables"
FIG_DIR = ROOT / "results" / "figures"

OPTION_ORDER = ["call", "digital"]

OPTION_LABELS = {
    "call": "Call",
    "digital": "Digital call",
}

OPTION_SETTINGS = {
    "call": {
        "noise_sd": 0.05,
        "t_max_sample": 0.95 * T_MATURITY,
        "baseline_smoothness": 15.0,
        "n_basis_s": 14,
        "n_basis_t": 8,
        "rbf_bandwidth": 0.30,
        "lambda_pde": 1e-3,
    },
    "digital": {
        "noise_sd": 0.02,
        "t_max_sample": 0.90 * T_MATURITY,
        "baseline_smoothness": 8.0,
        "n_basis_s": 14,
        "n_basis_t": 8,
        "rbf_bandwidth": 0.35,
        "lambda_pde": 0.1,
    },
}

METHOD_LABELS = {
    "finite_difference_pde_oracle": "PDE oracle",
    "raw_basis": "Raw basis",
    "baseline_only": "Baseline + basis",
    "baseline_pde": "Baseline + basis + PDE",
    "pure_nn": "Pure NN",
    "pinn": "PINN",
    "random_forest": "Random forest",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
}

METHOD_ORDER = [
    "finite_difference_pde_oracle",
    "baseline_pde",
    "baseline_only",
    "raw_basis",
    "pure_nn",
    "pinn",
    "random_forest",
    "xgboost",
    "catboost",
]

METHOD_COLORS = {
    "finite_difference_pde_oracle": "#355C7D",
    "baseline_pde": "#D95F02",
    "baseline_only": "#2A9D8F",
    "raw_basis": "#3A86FF",
    "pure_nn": "#7A5195",
    "pinn": "#C77DFF",
    "random_forest": "#6C757D",
    "xgboost": "#8D6E63",
    "catboost": "#4D908E",
}

METHOD_MARKERS = {
    "finite_difference_pde_oracle": "D",
    "baseline_pde": "o",
    "baseline_only": "s",
    "raw_basis": "P",
    "pure_nn": "^",
    "pinn": "v",
    "random_forest": "X",
    "xgboost": "*",
    "catboost": "h",
}

PROPOSED_METHODS = {"raw_basis", "baseline_only", "baseline_pde"}
ORACLE_METHODS = {"finite_difference_pde_oracle"}
TREE_METHODS = {"random_forest", "xgboost", "catboost"}
NEURAL_METHODS = {"pure_nn", "pinn"}

FAST_ROBUSTNESS_METHODS = [
    "finite_difference_pde_oracle",
    "raw_basis",
    "baseline_only",
    "baseline_pde",
    "random_forest",
    "xgboost",
    "catboost",
]

FD_ORACLE_CACHE = {}


# ============================================================
# Data generation
# ============================================================

def sample_design(n, t_max):
    S = np.random.uniform(S_MIN, S_MAX, size=n)
    t = np.random.uniform(0.0, t_max, size=n)
    return S, t


def get_truth(option_type, S, t):
    if option_type == "call":
        return bs_call_price(S=S, t=t, K=K, r=R, sigma=SIGMA, T=T_MATURITY)

    if option_type == "digital":
        return bs_digital_price(S=S, t=t, K=K, r=R, sigma=SIGMA, T=T_MATURITY)

    raise ValueError(f"Unknown option_type: {option_type}")


def make_dataset(option_type, n_train, n_test, seed):
    set_seed(seed)
    settings = OPTION_SETTINGS[option_type]

    S_train, t_train = sample_design(n_train, settings["t_max_sample"])
    true_train = get_truth(option_type, S_train, t_train)
    y_train = true_train + settings["noise_sd"] * np.random.randn(n_train)

    S_test, t_test = sample_design(n_test, settings["t_max_sample"])
    true_test = get_truth(option_type, S_test, t_test)

    return {
        "S_train": S_train,
        "t_train": t_train,
        "true_train": true_train,
        "y_train": y_train,
        "S_test": S_test,
        "t_test": t_test,
        "true_test": true_test,
    }


def make_features(S, t):
    S = np.asarray(S, dtype=float)
    t = np.asarray(t, dtype=float)

    x = np.log(S / K)
    q = t / T_MATURITY
    tau = (T_MATURITY - t) / T_MATURITY

    return np.column_stack([S / K, x, q, tau])


# ============================================================
# Baseline and finite-basis methods
# ============================================================

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


def fit_predict_finite_basis(option_type, method_name, data):
    settings = OPTION_SETTINGS[option_type]
    centers = make_option_centers(option_type)
    bandwidth = settings["rbf_bandwidth"]

    S_train = data["S_train"]
    t_train = data["t_train"]
    y_train = data["y_train"]
    S_test = data["S_test"]
    t_test = data["t_test"]

    if method_name == "raw_basis":
        baseline_train = np.zeros_like(S_train, dtype=float)
        baseline_pde_train = np.zeros_like(S_train, dtype=float)
        baseline_test = np.zeros_like(S_test, dtype=float)
        lambda_pde = 0.0
    elif method_name == "baseline_only":
        baseline_train, baseline_pde_train = get_baseline(option_type, S_train, t_train)
        baseline_test, _ = get_baseline(option_type, S_test, t_test)
        lambda_pde = 0.0
    elif method_name == "baseline_pde":
        baseline_train, baseline_pde_train = get_baseline(option_type, S_train, t_train)
        baseline_test, _ = get_baseline(option_type, S_test, t_test)
        lambda_pde = settings["lambda_pde"]
    else:
        raise ValueError(f"Unknown finite-basis method: {method_name}")

    fit = fit_finite_basis(
        S_train=S_train,
        t_train=t_train,
        y_train=y_train,
        baseline_train=baseline_train,
        baseline_pde_train=baseline_pde_train,
        centers=centers,
        bandwidth=bandwidth,
        K=K,
        r=R,
        sigma=SIGMA,
        T=T_MATURITY,
        lambda_pde=lambda_pde,
        lambda_ridge=LAMBDA_RIDGE,
    )

    pred_test = predict_price(
        S=S_test,
        t=t_test,
        baseline=baseline_test,
        theta=fit["theta"],
        centers=centers,
        bandwidth=bandwidth,
        K=K,
        T=T_MATURITY,
    )
    pred_train = fit["fitted_price_train"]

    extra = {
        "lambda_pde": lambda_pde,
        "n_basis": centers.shape[0],
        "rbf_bandwidth": bandwidth,
        "baseline_smoothness": settings["baseline_smoothness"],
        "augmented_condition_number": np.linalg.cond(fit["augmented_design"]),
        "category": "proposed",
    }

    return pred_test, pred_train, extra


# ============================================================
# Finite-difference PDE oracle
# ============================================================

def terminal_payoff(option_type, S_grid):
    if option_type == "call":
        return np.maximum(S_grid - K, 0.0)

    if option_type == "digital":
        return (S_grid >= K).astype(float)

    raise ValueError(f"Unknown option_type: {option_type}")


def boundary_values(option_type, tau):
    if option_type == "call":
        return 0.0, PDE_S_MAX - K * np.exp(-R * tau)

    if option_type == "digital":
        return 0.0, np.exp(-R * tau)

    raise ValueError(f"Unknown option_type: {option_type}")


def build_finite_difference_interpolator(option_type):
    S_grid = np.linspace(0.0, PDE_S_MAX, PDE_N_S + 1)
    tau_grid = np.linspace(0.0, T_MATURITY, PDE_N_TAU + 1)

    dS = S_grid[1] - S_grid[0]
    dtau = tau_grid[1] - tau_grid[0]

    interior = S_grid[1:-1]

    a = 0.5 * SIGMA ** 2 * interior ** 2 / dS ** 2 - R * interior / (2.0 * dS)
    b = -SIGMA ** 2 * interior ** 2 / dS ** 2 - R
    c = 0.5 * SIGMA ** 2 * interior ** 2 / dS ** 2 + R * interior / (2.0 * dS)

    matrix = diags(
        diagonals=[-dtau * a[1:], 1.0 - dtau * b, -dtau * c[:-1]],
        offsets=[-1, 0, 1],
        format="csr",
    )

    values = np.zeros((len(tau_grid), len(S_grid)))
    values[0, :] = terminal_payoff(option_type, S_grid)

    lower0, upper0 = boundary_values(option_type, tau_grid[0])
    values[0, 0] = lower0
    values[0, -1] = upper0

    v_inner = values[0, 1:-1].copy()

    for idx in range(1, len(tau_grid)):
        tau_new = tau_grid[idx]
        lower_bc, upper_bc = boundary_values(option_type, tau_new)

        rhs = v_inner.copy()
        rhs[0] += dtau * a[0] * lower_bc
        rhs[-1] += dtau * c[-1] * upper_bc

        v_inner = spsolve(matrix, rhs)

        values[idx, 0] = lower_bc
        values[idx, 1:-1] = v_inner
        values[idx, -1] = upper_bc

    return RegularGridInterpolator(
        (tau_grid, S_grid),
        values,
        bounds_error=False,
        fill_value=None,
    )


def finite_difference_pde_oracle(option_type, S_query, t_query):
    if option_type not in FD_ORACLE_CACHE:
        FD_ORACLE_CACHE[option_type] = build_finite_difference_interpolator(option_type)

    tau_query = T_MATURITY - np.asarray(t_query, dtype=float)
    S_query = np.asarray(S_query, dtype=float)
    points = np.column_stack([tau_query, S_query])

    return FD_ORACLE_CACHE[option_type](points)


def fit_predict_pde_oracle(option_type, data):
    pred_test = finite_difference_pde_oracle(
        option_type,
        data["S_test"],
        data["t_test"],
    )
    pred_train = finite_difference_pde_oracle(
        option_type,
        data["S_train"],
        data["t_train"],
    )

    extra = {
        "lambda_pde": np.nan,
        "n_basis": np.nan,
        "rbf_bandwidth": np.nan,
        "baseline_smoothness": np.nan,
        "augmented_condition_number": np.nan,
        "category": "oracle",
    }

    return pred_test, pred_train, extra


# ============================================================
# Neural network and PINN benchmarks
# ============================================================

if HAS_TORCH:

    class PriceNet(nn.Module):
        def __init__(self, hidden=64, n_layers=3):
            super().__init__()

            layers = []

            for layer_idx in range(n_layers):
                in_dim = 2 if layer_idx == 0 else hidden
                layers.append(nn.Linear(in_dim, hidden))
                layers.append(nn.Tanh())

            layers.append(nn.Linear(hidden, 1))

            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x).squeeze(-1)


def fit_predict_nn(method_name, data, seed, nn_epochs):
    if not HAS_TORCH:
        raise ImportError("PyTorch is not installed.")

    torch.manual_seed(seed)

    S_train = data["S_train"]
    t_train = data["t_train"]
    y_train = data["y_train"]
    S_test = data["S_test"]
    t_test = data["t_test"]

    x_train = np.log(S_train / K)
    q_train = t_train / T_MATURITY

    x_test = np.log(S_test / K)
    q_test = t_test / T_MATURITY

    X_train_np = np.column_stack([x_train, q_train]).astype(np.float32)
    X_test_np = np.column_stack([x_test, q_test]).astype(np.float32)
    y_train_np = y_train.astype(np.float32)

    y_mean = float(np.mean(y_train_np))
    y_scale = float(np.std(y_train_np))
    if y_scale <= 1e-8:
        y_scale = 1.0

    X_train_t = torch.tensor(X_train_np, dtype=torch.float32)
    y_train_t = torch.tensor(y_train_np, dtype=torch.float32)

    model = PriceNet(hidden=NN_HIDDEN, n_layers=NN_LAYERS)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=NN_LR,
        weight_decay=NN_WEIGHT_DECAY,
    )

    use_pde = method_name == "pinn"

    for _ in range(nn_epochs):
        optimizer.zero_grad()

        if use_pde:
            X_req = X_train_t.clone().detach().requires_grad_(True)
        else:
            X_req = X_train_t

        raw = model(X_req)
        price = y_mean + y_scale * raw
        data_loss = torch.mean(((price - y_train_t) / y_scale) ** 2)

        if use_pde:
            grad = torch.autograd.grad(price.sum(), X_req, create_graph=True)[0]
            u_x = grad[:, 0]
            u_q = grad[:, 1]
            u_xx = torch.autograd.grad(u_x.sum(), X_req, create_graph=True)[0][:, 0]

            pde_residual = (
                (1.0 / T_MATURITY) * u_q
                + (R - 0.5 * SIGMA ** 2) * u_x
                + 0.5 * SIGMA ** 2 * u_xx
                - R * price
            )
            pde_loss = torch.mean((pde_residual / y_scale) ** 2)
            loss = data_loss + PINN_LAMBDA_PDE * pde_loss
        else:
            loss = data_loss

        loss.backward()
        optimizer.step()

    model.eval()

    with torch.no_grad():
        train_raw = model(torch.tensor(X_train_np, dtype=torch.float32)).cpu().numpy()
        test_raw = model(torch.tensor(X_test_np, dtype=torch.float32)).cpu().numpy()

    pred_train = y_mean + y_scale * train_raw
    pred_test = y_mean + y_scale * test_raw

    extra = {
        "lambda_pde": PINN_LAMBDA_PDE if use_pde else 0.0,
        "n_basis": np.nan,
        "rbf_bandwidth": np.nan,
        "baseline_smoothness": np.nan,
        "augmented_condition_number": np.nan,
        "category": "neural",
    }

    return pred_test, pred_train, extra


# ============================================================
# Generic ML benchmarks
# ============================================================

def fit_predict_random_forest(data, seed):
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn is not installed.")

    X_train = make_features(data["S_train"], data["t_train"])
    X_test = make_features(data["S_test"], data["t_test"])

    model = RandomForestRegressor(
        n_estimators=500,
        min_samples_leaf=3,
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, data["y_train"])

    extra = {
        "lambda_pde": np.nan,
        "n_basis": np.nan,
        "rbf_bandwidth": np.nan,
        "baseline_smoothness": np.nan,
        "augmented_condition_number": np.nan,
        "category": "ml",
    }

    return model.predict(X_test), model.predict(X_train), extra


def fit_predict_xgboost(data, seed):
    if not HAS_XGBOOST:
        raise ImportError("xgboost is not installed.")

    X_train = make_features(data["S_train"], data["t_train"])
    X_test = make_features(data["S_test"], data["t_test"])

    model = XGBRegressor(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=seed,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, data["y_train"])

    extra = {
        "lambda_pde": np.nan,
        "n_basis": np.nan,
        "rbf_bandwidth": np.nan,
        "baseline_smoothness": np.nan,
        "augmented_condition_number": np.nan,
        "category": "ml",
    }

    return model.predict(X_test), model.predict(X_train), extra


def fit_predict_catboost(data, seed):
    if not HAS_CATBOOST:
        raise ImportError("catboost is not installed.")

    X_train = make_features(data["S_train"], data["t_train"])
    X_test = make_features(data["S_test"], data["t_test"])

    model = CatBoostRegressor(
        iterations=500,
        depth=4,
        learning_rate=0.03,
        loss_function="RMSE",
        random_seed=seed,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(X_train, data["y_train"])

    extra = {
        "lambda_pde": np.nan,
        "n_basis": np.nan,
        "rbf_bandwidth": np.nan,
        "baseline_smoothness": np.nan,
        "augmented_condition_number": np.nan,
        "category": "ml",
    }

    return model.predict(X_test), model.predict(X_train), extra


# ============================================================
# Benchmark dispatch and evaluation
# ============================================================

def method_available(method, skip_neural=False):
    if method in PROPOSED_METHODS or method in ORACLE_METHODS:
        return True
    if method == "pure_nn" or method == "pinn":
        return HAS_TORCH and not skip_neural
    if method == "random_forest":
        return HAS_SKLEARN
    if method == "xgboost":
        return HAS_XGBOOST
    if method == "catboost":
        return HAS_CATBOOST
    return False


def fit_predict_method(option_type, method, data, seed, nn_epochs):
    if method == "finite_difference_pde_oracle":
        return fit_predict_pde_oracle(option_type, data)
    if method in PROPOSED_METHODS:
        return fit_predict_finite_basis(option_type, method, data)
    if method in NEURAL_METHODS:
        return fit_predict_nn(method, data, seed=seed, nn_epochs=nn_epochs)
    if method == "random_forest":
        return fit_predict_random_forest(data, seed=seed)
    if method == "xgboost":
        return fit_predict_xgboost(data, seed=seed)
    if method == "catboost":
        return fit_predict_catboost(data, seed=seed)

    raise ValueError(f"Unknown method: {method}")


def make_result_row(
    option_type,
    method,
    pred_test,
    pred_train,
    data,
    runtime_seconds,
    extra,
    seed,
    rep=None,
):
    row = {
        "option_type": option_type,
        "method": method,
        "method_label": METHOD_LABELS.get(method, method),
        "price_rmse": rmse(pred_test, data["true_test"]),
        "train_price_rmse_noisy": rmse(pred_train, data["y_train"]),
        "runtime_seconds": runtime_seconds,
        "n_train": len(data["S_train"]),
        "n_test": len(data["S_test"]),
        "seed": seed,
        "rep": rep,
    }
    row.update(extra)
    return row


def run_methods_for_option(option_type, data, methods, seed, nn_epochs, rep=None):
    rows = []

    for method in methods:
        print(f"  {option_type}: {method}")
        start = time.perf_counter()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pred_test, pred_train, extra = fit_predict_method(
                option_type=option_type,
                method=method,
                data=data,
                seed=seed,
                nn_epochs=nn_epochs,
            )

        runtime = time.perf_counter() - start

        rows.append(
            make_result_row(
                option_type=option_type,
                method=method,
                pred_test=pred_test,
                pred_train=pred_train,
                data=data,
                runtime_seconds=runtime,
                extra=extra,
                seed=seed,
                rep=rep,
            )
        )

    return rows


def order_result_table(table):
    out = table.copy()
    out["option_type"] = pd.Categorical(out["option_type"], OPTION_ORDER, ordered=True)
    out["method"] = pd.Categorical(out["method"], METHOD_ORDER, ordered=True)
    out = out.sort_values(["option_type", "price_rmse", "method"]).reset_index(drop=True)
    out["option_type"] = out["option_type"].astype(str)
    out["method"] = out["method"].astype(str)
    return out


def run_single_seed(seed, n_train, n_test, nn_epochs, skip_neural):
    methods = [m for m in METHOD_ORDER if method_available(m, skip_neural=skip_neural)]

    print_section("Single-seed predictive horse-race")
    print("Methods:")
    for method in methods:
        print(f"  - {method}")

    rows = []

    for option_type in OPTION_ORDER:
        data = make_dataset(
            option_type=option_type,
            n_train=n_train,
            n_test=n_test,
            seed=seed,
        )
        rows.extend(
            run_methods_for_option(
                option_type=option_type,
                data=data,
                methods=methods,
                seed=seed,
                nn_epochs=nn_epochs,
            )
        )

    table = order_result_table(pd.DataFrame(rows))
    return table


def run_robustness(seed, n_train, n_test, n_rep):
    methods = [
        method
        for method in FAST_ROBUSTNESS_METHODS
        if method_available(method, skip_neural=True)
    ]

    print_section("Repeated-seed benchmark robustness")
    print(f"replications: {n_rep}")
    print("Methods:")
    for method in methods:
        print(f"  - {method}")

    rows = []

    for rep in range(n_rep):
        rep_seed = seed + rep
        print(f"rep {rep + 1}/{n_rep}, seed={rep_seed}")

        for option_type in OPTION_ORDER:
            data = make_dataset(
                option_type=option_type,
                n_train=n_train,
                n_test=n_test,
                seed=rep_seed,
            )
            rows.extend(
                run_methods_for_option(
                    option_type=option_type,
                    data=data,
                    methods=methods,
                    seed=rep_seed,
                    nn_epochs=0,
                    rep=rep,
                )
            )

    raw = order_result_table(pd.DataFrame(rows))
    summary = summarize_robustness(raw)

    return raw, summary


def summarize_robustness(raw):
    grouped = (
        raw.groupby(["option_type", "method", "method_label", "category"], observed=True)
        .agg(
            price_rmse_mean=("price_rmse", "mean"),
            price_rmse_sd=("price_rmse", "std"),
            train_price_rmse_noisy_mean=("train_price_rmse_noisy", "mean"),
            runtime_seconds_mean=("runtime_seconds", "mean"),
            n_rep=("price_rmse", "size"),
        )
        .reset_index()
    )

    grouped["price_rmse_sd"] = grouped["price_rmse_sd"].fillna(0.0)
    grouped = grouped.sort_values(
        ["option_type", "price_rmse_mean", "method"]
    ).reset_index(drop=True)

    return grouped


def build_summary_table(single_seed, robustness):
    rows = []

    for option_type in OPTION_ORDER:
        sub_single = single_seed[single_seed["option_type"] == option_type].copy()
        sub_robust = robustness[robustness["option_type"] == option_type].copy()

        best_single = sub_single.sort_values("price_rmse").iloc[0]

        proposed_single = sub_single[sub_single["method"] == "baseline_pde"].iloc[0]
        oracle_single = sub_single[
            sub_single["method"] == "finite_difference_pde_oracle"
        ].iloc[0]

        row = {
            "option_type": option_type,
            "best_single_seed_method": best_single["method"],
            "best_single_seed_rmse": best_single["price_rmse"],
            "baseline_pde_single_seed_rmse": proposed_single["price_rmse"],
            "pde_oracle_single_seed_rmse": oracle_single["price_rmse"],
        }

        if not sub_robust.empty:
            proposed_robust = sub_robust[sub_robust["method"] == "baseline_pde"].iloc[0]
            best_robust = sub_robust.sort_values("price_rmse_mean").iloc[0]

            row.update(
                {
                    "best_robust_method": best_robust["method"],
                    "best_robust_rmse_mean": best_robust["price_rmse_mean"],
                    "best_robust_rmse_sd": best_robust["price_rmse_sd"],
                    "baseline_pde_robust_rmse_mean": (
                        proposed_robust["price_rmse_mean"]
                    ),
                    "baseline_pde_robust_rmse_sd": proposed_robust["price_rmse_sd"],
                }
            )

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# Plots
# ============================================================

def method_color(method):
    return METHOD_COLORS.get(method, "#444444")


def method_marker(method):
    return METHOD_MARKERS.get(method, "o")


def save_figure(fig, base_name):
    ensure_dir(FIG_DIR)

    png_path = FIG_DIR / f"{base_name}.png"
    pdf_path = FIG_DIR / f"{base_name}.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return [str(png_path), str(pdf_path)]


def plot_rmse_rankings(single_seed):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6), sharex=False)

    for ax, option_type in zip(axes, OPTION_ORDER):
        sub = single_seed[single_seed["option_type"] == option_type].copy()
        sub = sub.sort_values("price_rmse", ascending=False).reset_index(drop=True)

        y = np.arange(len(sub))

        for idx, row in sub.iterrows():
            method = row["method"]
            ax.scatter(
                row["price_rmse"],
                y[idx],
                s=80,
                marker=method_marker(method),
                color=method_color(method),
                alpha=0.95,
                zorder=3,
            )
            ax.hlines(
                y[idx],
                xmin=sub["price_rmse"].min() * 0.8,
                xmax=row["price_rmse"],
                color=method_color(method),
                alpha=0.25,
                linewidth=2,
            )

        ax.set_yticks(y)
        ax.set_yticklabels([METHOD_LABELS[m] for m in sub["method"]])
        ax.set_xscale("log")
        ax.set_xlabel("Test price RMSE, log scale")
        ax.set_title(OPTION_LABELS[option_type])
        ax.grid(axis="x", alpha=0.25, which="both")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Predictive horse-race benchmark", y=1.02, fontsize=14)
    fig.tight_layout()

    return save_figure(fig, "benchmark_horserace_rmse")


def plot_accuracy_runtime(single_seed):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), sharex=False, sharey=False)

    legend_handles = {}

    for ax, option_type in zip(axes, OPTION_ORDER):
        sub = single_seed[single_seed["option_type"] == option_type].copy()
        sub["runtime_seconds"] = sub["runtime_seconds"].clip(lower=1e-6)

        for _, row in sub.iterrows():
            method = row["method"]
            handle = ax.scatter(
                row["runtime_seconds"],
                row["price_rmse"],
                s=80,
                marker=method_marker(method),
                color=method_color(method),
                alpha=0.95,
            )
            legend_handles[method] = handle

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Runtime seconds, log scale")
        ax.set_ylabel("Test price RMSE, log scale")
        ax.set_title(OPTION_LABELS[option_type])
        ax.grid(alpha=0.25, which="both")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    legend_methods = [
        method
        for method in METHOD_ORDER
        if method in legend_handles
    ]
    fig.legend(
        [legend_handles[method] for method in legend_methods],
        [METHOD_LABELS[method] for method in legend_methods],
        loc="lower center",
        ncol=5,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.04),
    )

    fig.suptitle("Accuracy-runtime comparison", y=1.02, fontsize=14)
    fig.tight_layout(rect=[0.0, 0.07, 1.0, 1.0])

    return save_figure(fig, "benchmark_horserace_accuracy_runtime")


def plot_robustness(robustness):
    if robustness.empty:
        return []

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6), sharex=False)

    for ax, option_type in zip(axes, OPTION_ORDER):
        sub = robustness[robustness["option_type"] == option_type].copy()
        sub = sub.sort_values("price_rmse_mean", ascending=False).reset_index(drop=True)

        y = np.arange(len(sub))

        for idx, row in sub.iterrows():
            method = row["method"]
            ax.errorbar(
                row["price_rmse_mean"],
                y[idx],
                xerr=row["price_rmse_sd"],
                fmt=method_marker(method),
                markersize=7,
                color=method_color(method),
                ecolor=method_color(method),
                elinewidth=1.5,
                capsize=3,
                alpha=0.95,
            )

        ax.set_yticks(y)
        ax.set_yticklabels([METHOD_LABELS[m] for m in sub["method"]])
        ax.set_xscale("log")
        ax.set_xlabel("Mean test price RMSE +/- SD, log scale")
        ax.set_title(OPTION_LABELS[option_type])
        ax.grid(axis="x", alpha=0.25, which="both")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Repeated-seed robustness for fast benchmark methods", y=1.02, fontsize=14)
    fig.tight_layout()

    return save_figure(fig, "benchmark_horserace_robustness")


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Run predictive benchmark horse-race.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-train", type=int, default=N_TRAIN)
    parser.add_argument("--n-test", type=int, default=N_TEST)
    parser.add_argument("--nn-epochs", type=int, default=NN_EPOCHS)
    parser.add_argument("--robustness-reps", type=int, default=ROBUSTNESS_REPS)
    parser.add_argument(
        "--skip-neural",
        action="store_true",
        help="Skip pure NN and PINN benchmarks.",
    )
    parser.add_argument(
        "--skip-robustness",
        action="store_true",
        help="Only run the single-seed benchmark.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Write tables without regenerating figures.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    ensure_dir(TABLE_DIR)
    ensure_dir(FIG_DIR)

    print_section("Predictive horse-race benchmark")
    print(f"seed: {args.seed}")
    print(f"training points: {args.n_train}")
    print(f"test points: {args.n_test}")
    print(f"neural benchmarks available: {HAS_TORCH}")
    print(f"random forest available: {HAS_SKLEARN}")
    print(f"xgboost available: {HAS_XGBOOST}")
    print(f"catboost available: {HAS_CATBOOST}")

    single_seed = run_single_seed(
        seed=args.seed,
        n_train=args.n_train,
        n_test=args.n_test,
        nn_epochs=args.nn_epochs,
        skip_neural=args.skip_neural,
    )

    single_path = TABLE_DIR / "benchmark_horserace_single_seed.csv"
    combined_path = TABLE_DIR / "benchmark_horserace_combined.csv"
    save_dataframe(single_seed, single_path, index=False)
    save_dataframe(single_seed, combined_path, index=False)

    robustness_raw = pd.DataFrame()
    robustness = pd.DataFrame()

    if not args.skip_robustness and args.robustness_reps > 0:
        robustness_raw, robustness = run_robustness(
            seed=args.seed,
            n_train=args.n_train,
            n_test=args.n_test,
            n_rep=args.robustness_reps,
        )
        save_dataframe(
            robustness_raw,
            TABLE_DIR / "benchmark_horserace_robustness_raw.csv",
            index=False,
        )
        save_dataframe(
            robustness,
            TABLE_DIR / "benchmark_horserace_robustness.csv",
            index=False,
        )

    summary = build_summary_table(single_seed, robustness)
    summary_path = TABLE_DIR / "benchmark_horserace_summary.csv"
    save_dataframe(summary, summary_path, index=False)

    print("\nSingle-seed ranking:")
    print(
        single_seed[
            [
                "option_type",
                "method",
                "price_rmse",
                "train_price_rmse_noisy",
                "runtime_seconds",
            ]
        ].to_string(index=False)
    )

    if not robustness.empty:
        print("\nRobustness ranking:")
        print(
            robustness[
                [
                    "option_type",
                    "method",
                    "price_rmse_mean",
                    "price_rmse_sd",
                    "runtime_seconds_mean",
                    "n_rep",
                ]
            ].to_string(index=False)
        )

    print("\nSaved tables:")
    print(single_path)
    print(combined_path)
    if not robustness.empty:
        print(TABLE_DIR / "benchmark_horserace_robustness_raw.csv")
        print(TABLE_DIR / "benchmark_horserace_robustness.csv")
    print(summary_path)

    if not args.skip_plots:
        saved = []
        saved.extend(plot_rmse_rankings(single_seed))
        saved.extend(plot_accuracy_runtime(single_seed))
        saved.extend(plot_robustness(robustness))

        print("\nSaved figures:")
        for path in saved:
            print(path)


if __name__ == "__main__":
    main()
