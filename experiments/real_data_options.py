"""
Real option-chain experiment using the local Cboe SPY snapshot.

The empirical goal is market realism rather than payoff diversity. Standard
exchange option-chain data mainly contains liquid vanilla calls and puts, so
we use the current SPY file more fully by fitting calls and puts separately
across maturity buckets.

Outputs:
    results/raw/real_data_options_clean.csv
    results/raw/real_data_options_predictions.csv
    results/raw/real_data_iv_smile_curves.csv
    results/raw/real_data_iv_surface_grid.csv
    results/tables/real_data_sample_summary.csv
    results/tables/real_data_benchmark_summary.csv
    results/tables/real_data_iv_benchmark_summary.csv
    results/figures/real_data_benchmark_by_slice.png
    results/figures/real_data_benchmark_by_slice.pdf
    results/figures/real_data_iv_smiles.png
    results/figures/real_data_iv_smiles.pdf
    results/figures/real_data_iv_surface.png
    results/figures/real_data_iv_surface.pdf

Run from repo root:
    python experiments/real_data_options.py

PINN benchmark, if desired:
    python experiments/real_data_options.py --include-pinn
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq, minimize_scalar
from scipy.stats import norm


ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "results" / ".matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / "results" / ".cache"))

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


from src.utils import ensure_dir, print_section, save_dataframe, set_seed


# ============================================================
# Optional packages
# ============================================================

try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.model_selection import train_test_split

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

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ============================================================
# Settings
# ============================================================

SEED = 123

TICKER = "SPY"
RAW_CBOE_PATH = ROOT / "results" / "raw" / f"cboe_options_{TICKER}_raw.csv"

RAW_DIR = ROOT / "results" / "raw"
TABLE_DIR = ROOT / "results" / "tables"
FIG_DIR = ROOT / "results" / "figures"

CLEAN_OUTPUT_PATH = RAW_DIR / "real_data_options_clean.csv"
PREDICTIONS_OUTPUT_PATH = RAW_DIR / "real_data_options_predictions.csv"
IV_SMILE_CURVES_OUTPUT_PATH = RAW_DIR / "real_data_iv_smile_curves.csv"
IV_SURFACE_GRID_OUTPUT_PATH = RAW_DIR / "real_data_iv_surface_grid.csv"

SAMPLE_SUMMARY_PATH = TABLE_DIR / "real_data_sample_summary.csv"
BENCHMARK_SUMMARY_PATH = TABLE_DIR / "real_data_benchmark_summary.csv"
IV_BENCHMARK_SUMMARY_PATH = TABLE_DIR / "real_data_iv_benchmark_summary.csv"

# Data filters.
MIN_DTE = 14
MAX_DTE = 180
MIN_MONEYNESS = 0.80
MAX_MONEYNESS = 1.20
MIN_MID_PRICE = 0.05
MAX_REL_SPREAD = 0.50

TEST_FRAC = 0.30
MIN_SLICE_QUOTES = 150
MIN_SLICE_EXPIRIES = 3

# Market/model simplifications.
RISK_FREE_RATE = 0.04
DIVIDEND_YIELD = 0.0

# RBF correction settings.
N_BASIS_K = 14
N_BASIS_T = 8
RBF_BANDWIDTH = 0.35
LAMBDA_PDE = "auto"
PDE_LAMBDA_GRID = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
PDE_VALIDATION_FRAC = 0.25
LAMBDA_RIDGE = 0.0
PINV_RCOND = 1e-10

# Quote uncertainty diagnostics used for price-space auxiliary metrics.
Z_VALUE = 1.96
HALF_SPREAD_FLOOR = 0.01

# Implied-volatility diagnostics.
IV_MIN = 1e-4
IV_MAX = 5.0
IV_PRICE_TOL = 1e-7
IV_SMILE_GRID_SIZE = 90
IV_SURFACE_MONEYNESS_GRID_SIZE = 46
IV_SURFACE_DTE_GRID_SIZE = 30

# Optional PINN settings.
PINN_HIDDEN = 64
PINN_LAYERS = 3
PINN_EPOCHS = 2000
PINN_LR = 1e-3
PINN_LAMBDA_PDE = 1e-4
PINN_WEIGHT_DECAY = 1e-6

DTE_BUCKETS = [
    ("near_14_45", "14-45 DTE", 14, 45),
    ("mid_46_90", "46-90 DTE", 46, 90),
    ("long_91_180", "91-180 DTE", 91, 180),
]

OPTION_ORDER = ["call", "put"]
BUCKET_ORDER = [bucket[0] for bucket in DTE_BUCKETS]

SMOOTH_METHODS = ["bs_plus_rbf", "bs_plus_rbf_pde"]

METHOD_ORDER = [
    "black_scholes_const_vol",
    "raw_rbf",
    "bs_plus_rbf",
    "bs_plus_rbf_pde",
    "random_forest",
    "gradient_boosting",
    "xgboost",
    "catboost",
    "pinn",
]

METHOD_LABELS = {
    "black_scholes_const_vol": "Calibrated BS",
    "raw_rbf": "Raw RBF",
    "bs_plus_rbf": "BS + RBF",
    "bs_plus_rbf_pde": "BS + RBF + PDE",
    "random_forest": "Random forest",
    "gradient_boosting": "Gradient boosting",
    "xgboost": "XGBoost",
    "catboost": "CatBoost",
    "pinn": "PINN",
}

METHOD_COLORS = {
    "black_scholes_const_vol": "#355C7D",
    "raw_rbf": "#3A86FF",
    "bs_plus_rbf": "#2A9D8F",
    "bs_plus_rbf_pde": "#D95F02",
    "random_forest": "#6C757D",
    "gradient_boosting": "#8D6E63",
    "xgboost": "#9C6644",
    "catboost": "#4D908E",
    "pinn": "#7A5195",
}

METHOD_MARKERS = {
    "black_scholes_const_vol": "D",
    "raw_rbf": "P",
    "bs_plus_rbf": "s",
    "bs_plus_rbf_pde": "o",
    "random_forest": "X",
    "gradient_boosting": "*",
    "xgboost": "^",
    "catboost": "h",
    "pinn": "v",
}


# ============================================================
# Black-Scholes formulas
# ============================================================

def _d1_d2(S, K, tau, r, sigma, q=0.0):
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    tau = np.maximum(np.asarray(tau, dtype=float), 1e-10)
    sigma = max(float(sigma), 1e-8)

    sqrt_tau = np.sqrt(tau)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * tau) / (sigma * sqrt_tau)
    d2 = d1 - sigma * sqrt_tau
    return d1, d2


def bs_price(option_type, S, K, tau, r, sigma, q=0.0):
    d1, d2 = _d1_d2(S=S, K=K, tau=tau, r=r, sigma=sigma, q=q)
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    tau = np.maximum(np.asarray(tau, dtype=float), 1e-10)
    discounted_spot = S * np.exp(-q * tau)
    discounted_strike = K * np.exp(-r * tau)

    if option_type == "call":
        return discounted_spot * norm.cdf(d1) - discounted_strike * norm.cdf(d2)

    if option_type == "put":
        return discounted_strike * norm.cdf(-d2) - discounted_spot * norm.cdf(-d1)

    raise ValueError(f"Unknown option_type: {option_type}")


def fit_constant_volatility(option_type, S0, K_train, tau_train, y_train, r, q=0.0):
    def objective(sigma):
        pred = bs_price(
            option_type=option_type,
            S=S0,
            K=K_train,
            tau=tau_train,
            r=r,
            sigma=sigma,
            q=q,
        )
        return np.mean((pred - y_train) ** 2)

    result = minimize_scalar(
        objective,
        bounds=(0.03, 2.00),
        method="bounded",
        options={"xatol": 1e-5},
    )
    return float(result.x)


def no_arbitrage_price_bounds(option_type, S, K, tau, r, q=0.0):
    S = float(S)
    K = float(K)
    tau = max(float(tau), 1e-10)

    discounted_spot = S * np.exp(-q * tau)
    discounted_strike = K * np.exp(-r * tau)

    if option_type == "call":
        lower = max(discounted_spot - discounted_strike, 0.0)
        upper = discounted_spot
    elif option_type == "put":
        lower = max(discounted_strike - discounted_spot, 0.0)
        upper = discounted_strike
    else:
        raise ValueError(f"Unknown option_type: {option_type}")

    return lower, upper


def implied_vol_from_price(option_type, price, S, K, tau, r, q=0.0):
    """
    Black-Scholes implied volatility under the same convention used for pricing.

    The inversion is a market convention.  If the input price is outside the
    convention's static bounds, the function returns NaN and the caller reports
    the invalid inversion rate explicitly.
    """
    if not np.isfinite(price) or not np.isfinite(S) or not np.isfinite(K) or not np.isfinite(tau):
        return np.nan
    if price < 0 or S <= 0 or K <= 0 or tau <= 0:
        return np.nan

    lower, upper = no_arbitrage_price_bounds(option_type, S, K, tau, r, q=q)
    if price < lower - IV_PRICE_TOL or price > upper + IV_PRICE_TOL:
        return np.nan

    price = min(max(float(price), lower), upper)
    price_low = float(bs_price(option_type, S, K, tau, r, IV_MIN, q=q))
    price_high = float(bs_price(option_type, S, K, tau, r, IV_MAX, q=q))

    if price <= price_low + IV_PRICE_TOL:
        return IV_MIN
    if price >= price_high - IV_PRICE_TOL:
        return IV_MAX

    try:
        return float(
            brentq(
                lambda sigma: float(bs_price(option_type, S, K, tau, r, sigma, q=q)) - price,
                IV_MIN,
                IV_MAX,
                xtol=1e-8,
                rtol=1e-8,
                maxiter=100,
            )
        )
    except ValueError:
        return np.nan


def implied_vol_array(option_type, price, S, K, tau, r, q=0.0):
    prices = np.asarray(price, dtype=float).ravel()
    spots = np.asarray(S, dtype=float)
    strikes = np.asarray(K, dtype=float).ravel()
    taus = np.asarray(tau, dtype=float).ravel()

    if spots.ndim == 0:
        spots = np.full_like(prices, float(spots), dtype=float)
    else:
        spots = spots.ravel()

    if isinstance(option_type, str):
        option_types = np.full(len(prices), option_type, dtype=object)
    else:
        option_types = np.asarray(option_type, dtype=object).ravel()

    out = np.empty(len(prices), dtype=float)
    for idx, (opt, px, s0, strike, maturity) in enumerate(
        zip(option_types, prices, spots, strikes, taus)
    ):
        out[idx] = implied_vol_from_price(
            str(opt),
            px,
            s0,
            strike,
            maturity,
            r,
            q=q,
        )

    return out


# ============================================================
# Loading and preprocessing
# ============================================================

def load_raw_cboe_data(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find raw option data at {path}. "
            "Expected the local Cboe snapshot in results/raw."
        )

    data = pd.read_csv(path)

    if "spot" not in data.columns:
        raise ValueError("Raw Cboe data must contain a 'spot' column.")

    spot = float(pd.to_numeric(data["spot"], errors="coerce").dropna().iloc[0])
    return data, spot


def assign_dte_bucket(dte):
    for bucket_id, _, low, high in DTE_BUCKETS:
        if low <= dte <= high:
            return bucket_id
    return np.nan


def clean_option_data(raw, spot):
    df = raw.copy()

    needed = ["option_type", "strike", "bid", "ask", "spot", "dte", "tau"]
    missing = [col for col in needed if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required raw columns: {missing}")

    numeric_cols = [
        "strike",
        "bid",
        "ask",
        "spot",
        "dte",
        "tau",
        "bid_size",
        "ask_size",
        "iv",
        "open_interest",
        "volume",
        "delta",
        "gamma",
        "vega",
        "theta",
        "rho",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["option_type"] = df["option_type"].astype(str).str.lower()
    df = df[df["option_type"].isin(OPTION_ORDER)].copy()

    if "expiry" in df.columns:
        df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce").astype(str)

    df = df.dropna(subset=["strike", "bid", "ask", "spot", "dte", "tau"])

    df = df[df["strike"] > 0]
    df = df[df["bid"] >= 0]
    df = df[df["ask"] > df["bid"]]
    df = df[df["tau"] > 0]
    df = df[df["dte"] >= MIN_DTE]
    df = df[df["dte"] <= MAX_DTE]

    df["mid"] = 0.5 * (df["bid"] + df["ask"])
    df["spread"] = df["ask"] - df["bid"]
    df["half_spread"] = 0.5 * df["spread"]
    df["rel_spread"] = df["spread"] / np.maximum(df["mid"], 1e-8)
    df["moneyness"] = df["strike"] / spot
    df["log_moneyness"] = np.log(df["moneyness"])

    df = df[df["mid"] >= MIN_MID_PRICE]
    df = df[df["rel_spread"] <= MAX_REL_SPREAD]
    df = df[df["moneyness"] >= MIN_MONEYNESS]
    df = df[df["moneyness"] <= MAX_MONEYNESS]

    df["quote_noise_var"] = (df["spread"] ** 2) / 12.0
    df["dte_bucket"] = df["dte"].apply(assign_dte_bucket)
    df = df.dropna(subset=["dte_bucket"]).copy()

    df["dte_bucket"] = pd.Categorical(df["dte_bucket"], BUCKET_ORDER, ordered=True)
    df = df.sort_values(["option_type", "dte_bucket", "expiry", "strike"]).copy()
    df["slice_id"] = df["option_type"].astype(str) + "_" + df["dte_bucket"].astype(str)
    df["row_id"] = np.arange(len(df))

    return df.reset_index(drop=True)


def slice_label(option_type, bucket_id):
    bucket_label = {bucket[0]: bucket[1] for bucket in DTE_BUCKETS}[bucket_id]
    return f"{option_type.capitalize()} {bucket_label}"


def iter_slices(clean):
    for option_type in OPTION_ORDER:
        for bucket_id, _, _, _ in DTE_BUCKETS:
            sub = clean[
                (clean["option_type"] == option_type)
                & (clean["dte_bucket"].astype(str) == bucket_id)
            ].copy()

            if len(sub) < MIN_SLICE_QUOTES:
                continue

            n_expiries = sub["expiry"].nunique() if "expiry" in sub.columns else 0
            if n_expiries < MIN_SLICE_EXPIRIES:
                continue

            yield option_type, bucket_id, sub.reset_index(drop=True)


def split_slice(sub, seed):
    train_df, test_df = train_test_split(
        sub,
        test_size=TEST_FRAC,
        random_state=seed,
        shuffle=True,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def summarize_sample(slices, split_lookup):
    rows = []

    for option_type, bucket_id, sub in slices:
        slice_id = f"{option_type}_{bucket_id}"
        train_df, test_df = split_lookup[slice_id]

        rows.append(
            {
                "slice_id": slice_id,
                "slice_label": slice_label(option_type, bucket_id),
                "option_type": option_type,
                "dte_bucket": bucket_id,
                "quotes": len(sub),
                "expiries": sub["expiry"].nunique() if "expiry" in sub.columns else np.nan,
                "train_quotes": len(train_df),
                "test_quotes": len(test_df),
                "dte_min": sub["dte"].min(),
                "dte_max": sub["dte"].max(),
                "moneyness_min": sub["moneyness"].min(),
                "moneyness_max": sub["moneyness"].max(),
                "median_mid": sub["mid"].median(),
                "median_spread": sub["spread"].median(),
                "median_relative_spread": sub["rel_spread"].median(),
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# RBF basis and PDE operator in strike/maturity coordinates
# ============================================================

def make_rbf_centers_from_train(x_train, tau_train, n_k, n_t):
    x_pad = 0.02
    tau_pad = 0.005

    x_grid = np.linspace(x_train.min() - x_pad, x_train.max() + x_pad, n_k)
    tau_grid = np.linspace(tau_train.min() - tau_pad, tau_train.max() + tau_pad, n_t)

    X, T = np.meshgrid(x_grid, tau_grid, indexing="ij")
    return np.column_stack([X.ravel(), T.ravel()])


def rbf_features_strike(K, tau, spot, centers, bandwidth):
    K = np.asarray(K, dtype=float).ravel()
    tau = np.asarray(tau, dtype=float).ravel()
    x = np.log(K / spot)

    dx = x[:, None] - centers[:, 0][None, :]
    dt = tau[:, None] - centers[:, 1][None, :]

    return np.exp(-0.5 * (dx ** 2 + dt ** 2) / (bandwidth ** 2))


def rbf_dK(K, tau, spot, centers, bandwidth):
    K = np.asarray(K, dtype=float).ravel()
    tau = np.asarray(tau, dtype=float).ravel()
    x = np.log(K / spot)

    dx = x[:, None] - centers[:, 0][None, :]
    dt = tau[:, None] - centers[:, 1][None, :]
    h2 = bandwidth ** 2
    Psi = np.exp(-0.5 * (dx ** 2 + dt ** 2) / h2)

    return Psi * (-dx / h2) / K[:, None]


def rbf_dKK(K, tau, spot, centers, bandwidth):
    K = np.asarray(K, dtype=float).ravel()
    tau = np.asarray(tau, dtype=float).ravel()
    x = np.log(K / spot)

    dx = x[:, None] - centers[:, 0][None, :]
    dt = tau[:, None] - centers[:, 1][None, :]
    h2 = bandwidth ** 2
    Psi = np.exp(-0.5 * (dx ** 2 + dt ** 2) / h2)

    return (
        Psi
        / (K[:, None] ** 2)
        * ((dx ** 2) / (h2 ** 2) + (dx - 1.0) / h2)
    )


def rbf_dtau(K, tau, spot, centers, bandwidth):
    K = np.asarray(K, dtype=float).ravel()
    tau = np.asarray(tau, dtype=float).ravel()
    x = np.log(K / spot)

    dx = x[:, None] - centers[:, 0][None, :]
    dt = tau[:, None] - centers[:, 1][None, :]
    h2 = bandwidth ** 2
    Psi = np.exp(-0.5 * (dx ** 2 + dt ** 2) / h2)

    return Psi * (-dt / h2)


def forward_pde_matrix(K, tau, spot, centers, bandwidth, r, sigma):
    """
    Forward Black-Scholes operator in strike and time-to-maturity:

        V_tau + r K V_K - 0.5 sigma^2 K^2 V_KK.

    The same linear operator applies to European calls and puts; payoff and
    boundary behavior differ by option type.
    """
    K = np.asarray(K, dtype=float).ravel()
    Psi_tau = rbf_dtau(K, tau, spot, centers, bandwidth)
    Psi_K = rbf_dK(K, tau, spot, centers, bandwidth)
    Psi_KK = rbf_dKK(K, tau, spot, centers, bandwidth)

    return (
        Psi_tau
        + r * K[:, None] * Psi_K
        - 0.5 * sigma ** 2 * (K[:, None] ** 2) * Psi_KK
    )


# ============================================================
# Linear surface fitting
# ============================================================

def fit_linear_surface(
    K_train,
    tau_train,
    y_train,
    baseline_train,
    spread_train,
    spot,
    centers,
    bandwidth,
    r,
    sigma,
    lambda_pde,
):
    K_train = np.asarray(K_train, dtype=float).ravel()
    tau_train = np.asarray(tau_train, dtype=float).ravel()
    y_train = np.asarray(y_train, dtype=float).ravel()
    baseline_train = np.asarray(baseline_train, dtype=float).ravel()
    spread_train = np.asarray(spread_train, dtype=float).ravel()

    Psi = rbf_features_strike(K_train, tau_train, spot, centers, bandwidth)
    R_mat = forward_pde_matrix(
        K=K_train,
        tau=tau_train,
        spot=spot,
        centers=centers,
        bandwidth=bandwidth,
        r=r,
        sigma=sigma,
    )

    n_basis = Psi.shape[1]
    A = Psi
    b = y_train - baseline_train

    if lambda_pde > 0:
        A = np.vstack([A, np.sqrt(lambda_pde) * R_mat])
        b = np.concatenate([b, np.zeros_like(y_train)])

    if LAMBDA_RIDGE > 0:
        A = np.vstack([A, np.sqrt(LAMBDA_RIDGE) * np.eye(n_basis)])
        b = np.concatenate([b, np.zeros(n_basis)])

    theta, *_ = np.linalg.lstsq(A, b, rcond=None)

    return {
        "theta": theta,
        "A": A,
        "Psi_train": Psi,
        "R_train": R_mat,
        "quote_noise_var": (spread_train ** 2) / 12.0,
        "lambda_pde": lambda_pde,
        "bandwidth": bandwidth,
        "centers": centers,
        "spot": spot,
        "sigma": sigma,
        "r": r,
        "augmented_condition_number": np.linalg.cond(A),
    }


def predict_linear_surface(fit, K, tau, baseline):
    Psi = rbf_features_strike(
        K=K,
        tau=tau,
        spot=fit["spot"],
        centers=fit["centers"],
        bandwidth=fit["bandwidth"],
    )
    return baseline + Psi @ fit["theta"]


def prediction_se_for_mid_quote(fit, K, tau, spread_test):
    Psi_test = rbf_features_strike(
        K=K,
        tau=tau,
        spot=fit["spot"],
        centers=fit["centers"],
        bandwidth=fit["bandwidth"],
    )

    A_pinv = np.linalg.pinv(fit["A"], rcond=PINV_RCOND)
    n_train = fit["Psi_train"].shape[0]
    influence = Psi_test @ A_pinv[:, :n_train]

    model_var = np.sum((influence ** 2) * fit["quote_noise_var"][None, :], axis=1)
    test_var = (np.asarray(spread_test, dtype=float).ravel() ** 2) / 12.0

    return np.sqrt(np.maximum(model_var + test_var, 0.0))


def resolve_pde_lambda(lambda_arg):
    if isinstance(lambda_arg, str):
        if lambda_arg.lower() == "auto":
            return None
        return float(lambda_arg)
    return float(lambda_arg)


def select_pde_lambda(
    K_train,
    tau_train,
    y_train,
    bid_train,
    ask_train,
    baseline_train,
    spread_train,
    spot,
    centers,
    bandwidth,
    r,
    sigma,
    seed,
):
    """
    Select the PDE penalty using an internal training/validation split.

    On market quotes the PDE is a stabilizer rather than an exact pricing law,
    so a fixed penalty can over-smooth smile and microstructure effects.
    """
    idx = np.arange(len(y_train))
    fit_idx, val_idx = train_test_split(
        idx,
        test_size=PDE_VALIDATION_FRAC,
        random_state=seed + 17,
        shuffle=True,
    )

    best_lambda = PDE_LAMBDA_GRID[0]
    best_metric = np.inf

    for lambda_candidate in PDE_LAMBDA_GRID:
        fit = fit_linear_surface(
            K_train=K_train[fit_idx],
            tau_train=tau_train[fit_idx],
            y_train=y_train[fit_idx],
            baseline_train=baseline_train[fit_idx],
            spread_train=spread_train[fit_idx],
            spot=spot,
            centers=centers,
            bandwidth=bandwidth,
            r=r,
            sigma=sigma,
            lambda_pde=lambda_candidate,
        )
        pred_val = predict_linear_surface(
            fit,
            K_train[val_idx],
            tau_train[val_idx],
            baseline=baseline_train[val_idx],
        )
        z_val = spread_normalized_residual(
            y_train[val_idx],
            pred_val,
            bid_train[val_idx],
            ask_train[val_idx],
        )
        metric = float(np.mean(np.abs(z_val)))

        if metric < best_metric:
            best_metric = metric
            best_lambda = lambda_candidate

    return best_lambda, best_metric


# ============================================================
# Metrics and records
# ============================================================

def rmse_np(pred, true):
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    return np.sqrt(np.mean((pred - true) ** 2))


def mae_np(pred, true):
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    return np.mean(np.abs(pred - true))


def spread_normalized_residual(mid, pred, bid, ask):
    half_spread = 0.5 * (np.asarray(ask, dtype=float) - np.asarray(bid, dtype=float))
    denom = np.maximum(half_spread, HALF_SPREAD_FLOOR)
    return (np.asarray(mid, dtype=float) - np.asarray(pred, dtype=float)) / denom


def evaluate_predictions(pred, y_true, bid, ask, se=None):
    pred = np.asarray(pred, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    bid = np.asarray(bid, dtype=float)
    ask = np.asarray(ask, dtype=float)

    z = spread_normalized_residual(y_true, pred, bid, ask)
    inside = (pred >= bid) & (pred <= ask)

    row = {
        "rmse": rmse_np(pred, y_true),
        "mae": mae_np(pred, y_true),
        "median_abs_error": np.median(np.abs(pred - y_true)),
        "rmse_over_mean_mid": rmse_np(pred, y_true) / np.mean(y_true),
        "inside_bid_ask": np.mean(inside),
        "spread_normalized_mae": np.mean(np.abs(z)),
        "spread_normalized_median_abs": np.median(np.abs(z)),
        "share_abs_z_le_1": np.mean(np.abs(z) <= 1.0),
        "share_abs_z_gt_2": np.mean(np.abs(z) > 2.0),
        "mean_signed_z": np.mean(z),
    }

    if se is not None:
        lower = pred - Z_VALUE * se
        upper = pred + Z_VALUE * se
        row["quote_ci_coverage"] = np.mean((lower <= y_true) & (y_true <= upper))
        row["avg_ci_width"] = np.mean(upper - lower)
        row["avg_se"] = np.mean(se)
    else:
        row["quote_ci_coverage"] = np.nan
        row["avg_ci_width"] = np.nan
        row["avg_se"] = np.nan

    return row


def make_prediction_records(test_df, method, pred, se=None, runtime_seconds=np.nan):
    out = test_df.copy()
    out["method"] = method
    out["method_label"] = METHOD_LABELS.get(method, method)
    out["prediction"] = pred
    out["residual"] = out["mid"].to_numpy(dtype=float) - pred
    out["abs_error"] = np.abs(out["residual"])
    out["spread_normalized_residual"] = spread_normalized_residual(
        out["mid"].to_numpy(dtype=float),
        pred,
        out["bid"].to_numpy(dtype=float),
        out["ask"].to_numpy(dtype=float),
    )
    out["abs_spread_normalized_residual"] = np.abs(out["spread_normalized_residual"])
    out["inside_bid_ask"] = (
        (pred >= out["bid"].to_numpy(dtype=float))
        & (pred <= out["ask"].to_numpy(dtype=float))
    )
    out["runtime_seconds"] = runtime_seconds

    if se is not None:
        out["se"] = se
        out["ci_lower"] = pred - Z_VALUE * se
        out["ci_upper"] = pred + Z_VALUE * se
        out["quote_ci_contains_mid"] = (
            (out["ci_lower"] <= out["mid"]) & (out["mid"] <= out["ci_upper"])
        )
    else:
        out["se"] = np.nan
        out["ci_lower"] = np.nan
        out["ci_upper"] = np.nan
        out["quote_ci_contains_mid"] = np.nan

    return out


# ============================================================
# Optional PINN benchmark
# ============================================================

if HAS_TORCH:

    class PricePINN(nn.Module):
        def __init__(self, hidden=64, n_layers=3):
            super().__init__()
            layers = []
            for layer_idx in range(n_layers):
                in_dim = 2 if layer_idx == 0 else hidden
                layers.append(nn.Linear(in_dim, hidden))
                layers.append(nn.Tanh())
            layers.append(nn.Linear(hidden, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, x_tau):
            return self.net(x_tau).squeeze(-1)


def fit_pinn_surface(K_train, tau_train, y_train, spot, r, sigma, seed, epochs):
    if not HAS_TORCH:
        raise ImportError("PyTorch is not installed.")

    torch.manual_seed(seed)

    K_train = np.asarray(K_train, dtype=float).ravel()
    tau_train = np.asarray(tau_train, dtype=float).ravel()
    y_train = np.asarray(y_train, dtype=float).ravel()

    x_train = np.log(K_train / spot)
    X_np = np.column_stack([x_train, tau_train]).astype(np.float32)
    y_np = y_train.astype(np.float32)

    y_mean = float(np.mean(y_np))
    y_scale = float(np.std(y_np))
    if y_scale <= 1e-8:
        y_scale = 1.0

    X = torch.tensor(X_np, dtype=torch.float32)
    y = torch.tensor(y_np, dtype=torch.float32)

    model = PricePINN(hidden=PINN_HIDDEN, n_layers=PINN_LAYERS)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=PINN_LR,
        weight_decay=PINN_WEIGHT_DECAY,
    )

    for _ in range(epochs):
        optimizer.zero_grad()

        X_req = X.clone().detach().requires_grad_(True)
        raw = model(X_req)
        price = y_mean + y_scale * raw

        data_loss = torch.mean(((price - y) / y_scale) ** 2)
        grad = torch.autograd.grad(price.sum(), X_req, create_graph=True)[0]
        V_x = grad[:, 0]
        V_tau = grad[:, 1]
        V_xx = torch.autograd.grad(V_x.sum(), X_req, create_graph=True)[0][:, 0]

        pde_residual = (
            V_tau
            + (r + 0.5 * sigma ** 2) * V_x
            - 0.5 * sigma ** 2 * V_xx
        )
        pde_loss = torch.mean((pde_residual / y_scale) ** 2)
        loss = data_loss + PINN_LAMBDA_PDE * pde_loss

        loss.backward()
        optimizer.step()

    return {
        "model": model,
        "spot": spot,
        "y_mean": y_mean,
        "y_scale": y_scale,
    }


def predict_pinn_surface(fit, K, tau):
    K = np.asarray(K, dtype=float).ravel()
    tau = np.asarray(tau, dtype=float).ravel()
    x = np.log(K / fit["spot"])
    X_np = np.column_stack([x, tau]).astype(np.float32)
    X = torch.tensor(X_np, dtype=torch.float32)

    fit["model"].eval()
    with torch.no_grad():
        raw = fit["model"](X).cpu().numpy()

    return fit["y_mean"] + fit["y_scale"] * raw


# ============================================================
# Model fitting per slice
# ============================================================

def method_available(method, include_pinn):
    if method in {
        "black_scholes_const_vol",
        "raw_rbf",
        "bs_plus_rbf",
        "bs_plus_rbf_pde",
        "random_forest",
        "gradient_boosting",
    }:
        return True
    if method == "xgboost":
        return HAS_XGBOOST
    if method == "catboost":
        return HAS_CATBOOST
    if method == "pinn":
        return include_pinn and HAS_TORCH
    return False


def make_ml_features(df):
    return df[["log_moneyness", "tau"]].to_numpy(dtype=float)


def predict_smooth_price_grid(option_type, method, spot, K, tau, sigma_hat, fit_bs_rbf, fit_pde):
    baseline = bs_price(
        option_type,
        spot,
        K,
        tau,
        RISK_FREE_RATE,
        sigma_hat,
        q=DIVIDEND_YIELD,
    )

    if method == "black_scholes_const_vol":
        return baseline
    if method == "bs_plus_rbf":
        return predict_linear_surface(fit_bs_rbf, K, tau, baseline=baseline)
    if method == "bs_plus_rbf_pde":
        return predict_linear_surface(fit_pde, K, tau, baseline=baseline)

    raise ValueError(f"Unsupported smooth grid method: {method}")


def make_smooth_iv_records(
    option_type,
    bucket_id,
    slice_id,
    slice_name,
    spot,
    K,
    tau,
    dte,
    sigma_hat,
    fit_bs_rbf,
    fit_pde,
    grid_kind,
):
    frames = []
    for method in ["black_scholes_const_vol", "bs_plus_rbf", "bs_plus_rbf_pde"]:
        price = predict_smooth_price_grid(
            option_type=option_type,
            method=method,
            spot=spot,
            K=K,
            tau=tau,
            sigma_hat=sigma_hat,
            fit_bs_rbf=fit_bs_rbf,
            fit_pde=fit_pde,
        )
        implied_vol = implied_vol_array(
            option_type=option_type,
            price=price,
            S=spot,
            K=K,
            tau=tau,
            r=RISK_FREE_RATE,
            q=DIVIDEND_YIELD,
        )

        frames.append(
            pd.DataFrame(
                {
                    "slice_id": slice_id,
                    "slice_label": slice_name,
                    "option_type": option_type,
                    "dte_bucket": bucket_id,
                    "method": method,
                    "method_label": METHOD_LABELS.get(method, method),
                    "grid_kind": grid_kind,
                    "moneyness": np.asarray(K, dtype=float) / spot,
                    "strike": np.asarray(K, dtype=float),
                    "tau": np.asarray(tau, dtype=float),
                    "dte": np.asarray(dte, dtype=float),
                    "price": price,
                    "implied_vol": implied_vol,
                    "sigma_hat": sigma_hat,
                }
            )
        )

    return pd.concat(frames, ignore_index=True)


def build_smooth_iv_grids(
    option_type,
    bucket_id,
    slice_id,
    slice_name,
    train_df,
    test_df,
    spot,
    sigma_hat,
    fit_bs_rbf,
    fit_pde,
):
    combined = pd.concat([train_df, test_df], ignore_index=True)

    moneyness_min = max(
        MIN_MONEYNESS,
        float(combined["moneyness"].quantile(0.03)),
    )
    moneyness_max = min(
        MAX_MONEYNESS,
        float(combined["moneyness"].quantile(0.97)),
    )
    if moneyness_min >= moneyness_max:
        moneyness_min = max(MIN_MONEYNESS, float(combined["moneyness"].min()))
        moneyness_max = min(MAX_MONEYNESS, float(combined["moneyness"].max()))

    rep_dte = float(test_df["dte"].median())
    rep_tau = rep_dte / 365.0
    smile_moneyness = np.linspace(moneyness_min, moneyness_max, IV_SMILE_GRID_SIZE)
    smile_K = spot * smile_moneyness
    smile_tau = np.full_like(smile_K, rep_tau, dtype=float)
    smile_dte = np.full_like(smile_K, rep_dte, dtype=float)

    smile = make_smooth_iv_records(
        option_type=option_type,
        bucket_id=bucket_id,
        slice_id=slice_id,
        slice_name=slice_name,
        spot=spot,
        K=smile_K,
        tau=smile_tau,
        dte=smile_dte,
        sigma_hat=sigma_hat,
        fit_bs_rbf=fit_bs_rbf,
        fit_pde=fit_pde,
        grid_kind="smile",
    )

    surface_moneyness = np.linspace(
        moneyness_min,
        moneyness_max,
        IV_SURFACE_MONEYNESS_GRID_SIZE,
    )
    surface_dte = np.linspace(
        float(combined["dte"].min()),
        float(combined["dte"].max()),
        IV_SURFACE_DTE_GRID_SIZE,
    )
    M, D = np.meshgrid(surface_moneyness, surface_dte)
    surface_K = (spot * M).ravel()
    surface_dte_flat = D.ravel()
    surface_tau = surface_dte_flat / 365.0

    surface = make_smooth_iv_records(
        option_type=option_type,
        bucket_id=bucket_id,
        slice_id=slice_id,
        slice_name=slice_name,
        spot=spot,
        K=surface_K,
        tau=surface_tau,
        dte=surface_dte_flat,
        sigma_hat=sigma_hat,
        fit_bs_rbf=fit_bs_rbf,
        fit_pde=fit_pde,
        grid_kind="surface",
    )

    return smile, surface


def fit_and_evaluate_slice(option_type, bucket_id, train_df, test_df, spot, args):
    slice_id = f"{option_type}_{bucket_id}"
    slice_name = slice_label(option_type, bucket_id)

    K_train = train_df["strike"].to_numpy(dtype=float)
    tau_train = train_df["tau"].to_numpy(dtype=float)
    y_train = train_df["mid"].to_numpy(dtype=float)
    bid_train = train_df["bid"].to_numpy(dtype=float)
    ask_train = train_df["ask"].to_numpy(dtype=float)
    spread_train = train_df["spread"].to_numpy(dtype=float)

    K_test = test_df["strike"].to_numpy(dtype=float)
    tau_test = test_df["tau"].to_numpy(dtype=float)
    y_test = test_df["mid"].to_numpy(dtype=float)
    bid_test = test_df["bid"].to_numpy(dtype=float)
    ask_test = test_df["ask"].to_numpy(dtype=float)
    spread_test = test_df["spread"].to_numpy(dtype=float)

    print_section(f"Real-data slice: {slice_name}")
    print(f"train quotes: {len(train_df)}")
    print(f"test quotes:  {len(test_df)}")

    sigma_hat = fit_constant_volatility(
        option_type=option_type,
        S0=spot,
        K_train=K_train,
        tau_train=tau_train,
        y_train=y_train,
        r=RISK_FREE_RATE,
        q=DIVIDEND_YIELD,
    )
    print(f"calibrated sigma: {sigma_hat:.4f}")

    bs_train = bs_price(
        option_type,
        spot,
        K_train,
        tau_train,
        RISK_FREE_RATE,
        sigma_hat,
        q=DIVIDEND_YIELD,
    )
    bs_test = bs_price(
        option_type,
        spot,
        K_test,
        tau_test,
        RISK_FREE_RATE,
        sigma_hat,
        q=DIVIDEND_YIELD,
    )

    x_train = np.log(K_train / spot)
    centers = make_rbf_centers_from_train(
        x_train=x_train,
        tau_train=tau_train,
        n_k=args.n_basis_k,
        n_t=args.n_basis_t,
    )

    rows = []
    prediction_frames = []

    def add_method_result(method, pred, se, runtime, extra):
        row = {
            "slice_id": slice_id,
            "slice_label": slice_name,
            "option_type": option_type,
            "dte_bucket": bucket_id,
            "method": method,
            "method_label": METHOD_LABELS.get(method, method),
            "sigma_hat": sigma_hat,
            "n_train": len(train_df),
            "n_test": len(test_df),
            "runtime_seconds": runtime,
            **evaluate_predictions(pred, y_test, bid_test, ask_test, se=se),
        }
        row.update(extra)
        rows.append(row)

        prediction_frame = make_prediction_records(
            test_df,
            method,
            pred,
            se=se,
            runtime_seconds=runtime,
        )
        prediction_frame["slice_label"] = slice_name
        prediction_frames.append(prediction_frame)

    # Calibrated Black-Scholes reference.
    start = time.perf_counter()
    runtime = time.perf_counter() - start
    add_method_result(
        method="black_scholes_const_vol",
        pred=bs_test,
        se=None,
        runtime=runtime,
        extra={
            "lambda_pde": np.nan,
            "n_basis": 0,
            "rbf_bandwidth": np.nan,
            "augmented_condition_number": np.nan,
        },
    )

    # Raw RBF.
    start = time.perf_counter()
    fit_raw = fit_linear_surface(
        K_train=K_train,
        tau_train=tau_train,
        y_train=y_train,
        baseline_train=np.zeros_like(y_train),
        spread_train=spread_train,
        spot=spot,
        centers=centers,
        bandwidth=args.rbf_bandwidth,
        r=RISK_FREE_RATE,
        sigma=sigma_hat,
        lambda_pde=0.0,
    )
    pred_raw = predict_linear_surface(
        fit_raw,
        K_test,
        tau_test,
        baseline=np.zeros_like(y_test),
    )
    se_raw = prediction_se_for_mid_quote(fit_raw, K_test, tau_test, spread_test)
    runtime = time.perf_counter() - start
    add_method_result(
        method="raw_rbf",
        pred=pred_raw,
        se=se_raw,
        runtime=runtime,
        extra={
            "lambda_pde": 0.0,
            "n_basis": centers.shape[0],
            "rbf_bandwidth": args.rbf_bandwidth,
            "augmented_condition_number": fit_raw["augmented_condition_number"],
        },
    )

    # BS + RBF.
    start = time.perf_counter()
    fit_bs_rbf = fit_linear_surface(
        K_train=K_train,
        tau_train=tau_train,
        y_train=y_train,
        baseline_train=bs_train,
        spread_train=spread_train,
        spot=spot,
        centers=centers,
        bandwidth=args.rbf_bandwidth,
        r=RISK_FREE_RATE,
        sigma=sigma_hat,
        lambda_pde=0.0,
    )
    pred_bs_rbf = predict_linear_surface(fit_bs_rbf, K_test, tau_test, baseline=bs_test)
    se_bs_rbf = prediction_se_for_mid_quote(fit_bs_rbf, K_test, tau_test, spread_test)
    runtime = time.perf_counter() - start
    add_method_result(
        method="bs_plus_rbf",
        pred=pred_bs_rbf,
        se=se_bs_rbf,
        runtime=runtime,
        extra={
            "lambda_pde": 0.0,
            "n_basis": centers.shape[0],
            "rbf_bandwidth": args.rbf_bandwidth,
            "augmented_condition_number": fit_bs_rbf["augmented_condition_number"],
        },
    )

    # BS + RBF + PDE.
    start = time.perf_counter()
    lambda_pde = resolve_pde_lambda(args.lambda_pde)
    lambda_pde_selection_metric = np.nan
    if lambda_pde is None:
        lambda_pde, lambda_pde_selection_metric = select_pde_lambda(
            K_train=K_train,
            tau_train=tau_train,
            y_train=y_train,
            bid_train=bid_train,
            ask_train=ask_train,
            baseline_train=bs_train,
            spread_train=spread_train,
            spot=spot,
            centers=centers,
            bandwidth=args.rbf_bandwidth,
            r=RISK_FREE_RATE,
            sigma=sigma_hat,
            seed=args.seed,
        )
    fit_pde = fit_linear_surface(
        K_train=K_train,
        tau_train=tau_train,
        y_train=y_train,
        baseline_train=bs_train,
        spread_train=spread_train,
        spot=spot,
        centers=centers,
        bandwidth=args.rbf_bandwidth,
        r=RISK_FREE_RATE,
        sigma=sigma_hat,
        lambda_pde=lambda_pde,
    )
    pred_pde = predict_linear_surface(fit_pde, K_test, tau_test, baseline=bs_test)
    se_pde = prediction_se_for_mid_quote(fit_pde, K_test, tau_test, spread_test)
    runtime = time.perf_counter() - start
    add_method_result(
        method="bs_plus_rbf_pde",
        pred=pred_pde,
        se=se_pde,
        runtime=runtime,
        extra={
            "lambda_pde": lambda_pde,
            "lambda_pde_selection_metric": lambda_pde_selection_metric,
            "n_basis": centers.shape[0],
            "rbf_bandwidth": args.rbf_bandwidth,
            "augmented_condition_number": fit_pde["augmented_condition_number"],
        },
    )

    iv_smile_grid, iv_surface_grid = build_smooth_iv_grids(
        option_type=option_type,
        bucket_id=bucket_id,
        slice_id=slice_id,
        slice_name=slice_name,
        train_df=train_df,
        test_df=test_df,
        spot=spot,
        sigma_hat=sigma_hat,
        fit_bs_rbf=fit_bs_rbf,
        fit_pde=fit_pde,
    )

    X_train = make_ml_features(train_df)
    X_test = make_ml_features(test_df)

    # Random forest.
    start = time.perf_counter()
    rf = RandomForestRegressor(
        n_estimators=300,
        min_samples_leaf=5,
        random_state=args.seed,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    runtime = time.perf_counter() - start
    add_method_result(
        method="random_forest",
        pred=pred_rf,
        se=None,
        runtime=runtime,
        extra={
            "lambda_pde": np.nan,
            "n_basis": np.nan,
            "rbf_bandwidth": np.nan,
            "augmented_condition_number": np.nan,
        },
    )

    # Gradient boosting.
    start = time.perf_counter()
    gbr = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.03,
        max_depth=3,
        random_state=args.seed,
    )
    gbr.fit(X_train, y_train)
    pred_gbr = gbr.predict(X_test)
    runtime = time.perf_counter() - start
    add_method_result(
        method="gradient_boosting",
        pred=pred_gbr,
        se=None,
        runtime=runtime,
        extra={
            "lambda_pde": np.nan,
            "n_basis": np.nan,
            "rbf_bandwidth": np.nan,
            "augmented_condition_number": np.nan,
        },
    )

    if HAS_XGBOOST:
        start = time.perf_counter()
        xgb = XGBRegressor(
            n_estimators=400,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            objective="reg:squarederror",
            random_state=args.seed,
            n_jobs=-1,
            verbosity=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            xgb.fit(X_train, y_train)
        pred_xgb = xgb.predict(X_test)
        runtime = time.perf_counter() - start
        add_method_result(
            method="xgboost",
            pred=pred_xgb,
            se=None,
            runtime=runtime,
            extra={
                "lambda_pde": np.nan,
                "n_basis": np.nan,
                "rbf_bandwidth": np.nan,
                "augmented_condition_number": np.nan,
            },
        )

    if HAS_CATBOOST:
        start = time.perf_counter()
        cat = CatBoostRegressor(
            iterations=400,
            depth=4,
            learning_rate=0.03,
            loss_function="RMSE",
            random_seed=args.seed,
            verbose=False,
            allow_writing_files=False,
        )
        cat.fit(X_train, y_train)
        pred_cat = cat.predict(X_test)
        runtime = time.perf_counter() - start
        add_method_result(
            method="catboost",
            pred=pred_cat,
            se=None,
            runtime=runtime,
            extra={
                "lambda_pde": np.nan,
                "n_basis": np.nan,
                "rbf_bandwidth": np.nan,
                "augmented_condition_number": np.nan,
            },
        )

    if args.include_pinn and HAS_TORCH:
        start = time.perf_counter()
        fit_pinn = fit_pinn_surface(
            K_train=K_train,
            tau_train=tau_train,
            y_train=y_train,
            spot=spot,
            r=RISK_FREE_RATE,
            sigma=sigma_hat,
            seed=args.seed,
            epochs=args.pinn_epochs,
        )
        pred_pinn = predict_pinn_surface(fit_pinn, K_test, tau_test)
        runtime = time.perf_counter() - start
        add_method_result(
            method="pinn",
            pred=pred_pinn,
            se=None,
            runtime=runtime,
            extra={
                "lambda_pde": PINN_LAMBDA_PDE,
                "n_basis": np.nan,
                "rbf_bandwidth": np.nan,
                "augmented_condition_number": np.nan,
            },
        )

    return (
        pd.DataFrame(rows),
        pd.concat(prediction_frames, ignore_index=True),
        iv_smile_grid,
        iv_surface_grid,
    )


# ============================================================
# Implied-volatility summaries
# ============================================================

def add_implied_vol_columns(predictions):
    out = predictions.copy()
    out["market_iv"] = implied_vol_array(
        option_type=out["option_type"].to_numpy(),
        price=out["mid"].to_numpy(dtype=float),
        S=out["spot"].to_numpy(dtype=float),
        K=out["strike"].to_numpy(dtype=float),
        tau=out["tau"].to_numpy(dtype=float),
        r=RISK_FREE_RATE,
        q=DIVIDEND_YIELD,
    )
    out["predicted_iv"] = implied_vol_array(
        option_type=out["option_type"].to_numpy(),
        price=out["prediction"].to_numpy(dtype=float),
        S=out["spot"].to_numpy(dtype=float),
        K=out["strike"].to_numpy(dtype=float),
        tau=out["tau"].to_numpy(dtype=float),
        r=RISK_FREE_RATE,
        q=DIVIDEND_YIELD,
    )
    out["iv_error"] = out["predicted_iv"] - out["market_iv"]
    out["abs_iv_error"] = np.abs(out["iv_error"])
    out["valid_market_iv"] = np.isfinite(out["market_iv"])
    out["valid_predicted_iv"] = np.isfinite(out["predicted_iv"])
    out["valid_iv_pair"] = out["valid_market_iv"] & out["valid_predicted_iv"]
    return out


def build_iv_benchmark_summary(predictions):
    rows = []
    group_cols = [
        "slice_id",
        "slice_label",
        "option_type",
        "dte_bucket",
        "method",
        "method_label",
    ]

    for keys, group in predictions.groupby(group_cols, observed=True):
        valid = group["valid_iv_pair"].to_numpy(dtype=bool)
        row = dict(zip(group_cols, keys))
        row["n_test"] = len(group)
        row["n_valid_iv"] = int(valid.sum())
        row["valid_iv_rate"] = float(valid.mean()) if len(group) else np.nan
        row["price_rmse"] = rmse_np(group["prediction"], group["mid"])
        row["price_mae"] = mae_np(group["prediction"], group["mid"])
        row["inside_bid_ask"] = float(group["inside_bid_ask"].mean())

        if valid.any():
            row["iv_rmse"] = rmse_np(
                group.loc[valid, "predicted_iv"],
                group.loc[valid, "market_iv"],
            )
            row["iv_mae"] = mae_np(
                group.loc[valid, "predicted_iv"],
                group.loc[valid, "market_iv"],
            )
            row["iv_median_abs_error"] = float(
                group.loc[valid, "abs_iv_error"].median()
            )
            row["market_iv_median"] = float(group.loc[valid, "market_iv"].median())
            row["predicted_iv_median"] = float(group.loc[valid, "predicted_iv"].median())
        else:
            row["iv_rmse"] = np.nan
            row["iv_mae"] = np.nan
            row["iv_median_abs_error"] = np.nan
            row["market_iv_median"] = np.nan
            row["predicted_iv_median"] = np.nan

        rows.append(row)

    summary = pd.DataFrame(rows)
    summary["option_type"] = pd.Categorical(
        summary["option_type"], OPTION_ORDER, ordered=True
    )
    summary["dte_bucket"] = pd.Categorical(
        summary["dte_bucket"], BUCKET_ORDER, ordered=True
    )
    summary["method"] = pd.Categorical(summary["method"], METHOD_ORDER, ordered=True)
    summary = summary.sort_values(
        ["option_type", "dte_bucket", "iv_rmse", "method"],
        na_position="last",
    ).reset_index(drop=True)
    summary["option_type"] = summary["option_type"].astype(str)
    summary["dte_bucket"] = summary["dte_bucket"].astype(str)
    summary["method"] = summary["method"].astype(str)
    return summary


def select_best_iv_smooth_methods(iv_benchmark):
    smooth = iv_benchmark[
        iv_benchmark["method"].isin(SMOOTH_METHODS) & iv_benchmark["iv_rmse"].notna()
    ].copy()
    if smooth.empty:
        return pd.DataFrame(columns=["slice_id", "best_smooth_method"])

    idx = smooth.groupby("slice_id")["iv_rmse"].idxmin()
    return smooth.loc[idx, ["slice_id", "method"]].rename(
        columns={"method": "best_smooth_method"}
    )


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


def plot_benchmark_by_slice(benchmark):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=False, sharey=True)

    available_methods = [m for m in METHOD_ORDER if m in set(benchmark["method"])]
    offsets = np.linspace(-0.25, 0.25, max(len(available_methods), 1))
    offset_map = dict(zip(available_methods, offsets))

    for ax, option_type in zip(axes, OPTION_ORDER):
        sub = benchmark[benchmark["option_type"] == option_type].copy()
        y_base = {bucket_id: idx for idx, bucket_id in enumerate(BUCKET_ORDER)}

        for method in available_methods:
            msub = sub[sub["method"] == method]
            if msub.empty:
                continue

            y = [
                y_base[bucket] + offset_map[method]
                for bucket in msub["dte_bucket"].astype(str)
            ]
            ax.scatter(
                msub["rmse"],
                y,
                marker=METHOD_MARKERS.get(method, "o"),
                color=METHOD_COLORS.get(method, "#444444"),
                s=70,
                alpha=0.9,
                label=METHOD_LABELS.get(method, method),
            )

        ax.set_yticks(list(y_base.values()))
        ax.set_yticklabels([bucket[1] for bucket in DTE_BUCKETS])
        ax.set_xscale("log")
        ax.set_xlabel("Held-out mid quote RMSE, log scale")
        ax.set_title(option_type.capitalize())
        ax.grid(axis="x", alpha=0.25, which="both")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    if not handles:
        handles, labels = axes[1].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.05),
    )

    fig.suptitle("Real-data benchmark by option type and maturity", y=1.02, fontsize=14)
    fig.tight_layout(rect=[0.0, 0.08, 1.0, 1.0])

    return save_figure(fig, "real_data_benchmark_by_slice")


def plot_iv_smiles(predictions, iv_smile_curves):
    fig, axes = plt.subplots(
        2,
        3,
        figsize=(14.5, 7.2),
        sharex=True,
        sharey=False,
    )

    bucket_labels = {bucket_id: label for bucket_id, label, _, _ in DTE_BUCKETS}
    line_methods = ["black_scholes_const_vol", "bs_plus_rbf", "bs_plus_rbf_pde"]
    line_styles = {
        "black_scholes_const_vol": "--",
        "bs_plus_rbf": "-",
        "bs_plus_rbf_pde": "-",
    }

    for row, option_type in enumerate(OPTION_ORDER):
        for col, bucket_id in enumerate(BUCKET_ORDER):
            ax = axes[row, col]
            slice_id = f"{option_type}_{bucket_id}"

            market = predictions[
                (predictions["slice_id"] == slice_id)
                & (predictions["method"] == "bs_plus_rbf")
                & predictions["valid_market_iv"]
            ].copy()

            if not market.empty:
                ax.scatter(
                    market["moneyness"],
                    market["market_iv"],
                    s=16,
                    color="#5F6C72",
                    alpha=0.35,
                    linewidths=0,
                    label="Held-out market IV" if row == 0 and col == 0 else None,
                )

            curve = iv_smile_curves[iv_smile_curves["slice_id"] == slice_id].copy()
            for method in line_methods:
                mcurve = curve[
                    (curve["method"] == method) & curve["implied_vol"].notna()
                ].sort_values("moneyness")
                if mcurve.empty:
                    continue

                ax.plot(
                    mcurve["moneyness"],
                    mcurve["implied_vol"],
                    color=METHOD_COLORS.get(method, "#333333"),
                    linestyle=line_styles.get(method, "-"),
                    linewidth=2.0 if method != "black_scholes_const_vol" else 1.5,
                    alpha=0.95,
                    label=METHOD_LABELS.get(method, method) if row == 0 and col == 0 else None,
                )

            panel_values = []
            if not market.empty:
                panel_values.append(market["market_iv"].to_numpy(dtype=float))
            if not curve.empty:
                panel_values.append(curve["implied_vol"].dropna().to_numpy(dtype=float))
            if panel_values:
                values = np.concatenate(panel_values)
                values = values[np.isfinite(values)]
                if values.size:
                    low = max(0.0, float(np.nanquantile(values, 0.02)) - 0.03)
                    high = min(2.0, float(np.nanquantile(values, 0.98)) + 0.04)
                    if high - low < 0.08:
                        high = low + 0.08
                    ax.set_ylim(low, high)

            ax.axvline(1.0, color="#444444", alpha=0.25, linewidth=1)
            ax.set_title(f"{option_type.capitalize()} {bucket_labels[bucket_id]}")
            ax.grid(alpha=0.18)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if col == 0:
                ax.set_ylabel("IV")
            if row == 1:
                ax.set_xlabel("Moneyness K / S0")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=9,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle("Implied-volatility smiles from fitted price surfaces", y=0.97, fontsize=14)
    fig.subplots_adjust(
        left=0.08,
        right=0.99,
        bottom=0.14,
        top=0.86,
        wspace=0.18,
        hspace=0.45,
    )

    return save_figure(fig, "real_data_iv_smiles")


def choose_iv_surface_method(iv_benchmark, slice_id):
    best = select_best_iv_smooth_methods(iv_benchmark)
    sub = best[best["slice_id"] == slice_id]
    if not sub.empty:
        return sub["best_smooth_method"].iloc[0]
    return "bs_plus_rbf"


def plot_iv_surface(predictions, iv_surface_grid, iv_benchmark):
    fig = plt.figure(figsize=(12.5, 5.2))
    panel_positions = [
        [0.03, 0.12, 0.43, 0.74],
        [0.54, 0.12, 0.43, 0.74],
    ]

    for panel, option_type in enumerate(OPTION_ORDER, start=1):
        slice_id = f"{option_type}_mid_46_90"
        method = choose_iv_surface_method(iv_benchmark, slice_id)
        surface = iv_surface_grid[
            (iv_surface_grid["slice_id"] == slice_id)
            & (iv_surface_grid["method"] == method)
            & iv_surface_grid["implied_vol"].notna()
        ].copy()

        ax = fig.add_axes(panel_positions[panel - 1], projection="3d")
        if surface.empty:
            ax.set_axis_off()
            continue

        pivot = surface.pivot_table(
            index="dte",
            columns="moneyness",
            values="implied_vol",
            observed=True,
        )
        X, Y = np.meshgrid(
            pivot.columns.to_numpy(dtype=float),
            pivot.index.to_numpy(dtype=float),
        )
        Z = pivot.to_numpy(dtype=float)

        ax.plot_surface(
            X,
            Y,
            Z,
            color="#9ecae1",
            alpha=0.55,
            linewidth=0,
            antialiased=True,
        )

        market = predictions[
            (predictions["slice_id"] == slice_id)
            & (predictions["method"] == "bs_plus_rbf")
            & predictions["valid_market_iv"]
        ].copy()
        ax.scatter(
            market["moneyness"],
            market["dte"],
            market["market_iv"],
            color="#F7F7F7",
            edgecolors="#333333",
            linewidths=0.35,
            s=22,
            alpha=0.80,
        )

        method_label = METHOD_LABELS.get(method, method)
        ax.set_title(f"{option_type.capitalize()} 46-90 DTE: {method_label}", pad=10)
        ax.set_xlabel("K / S0", labelpad=6)
        ax.set_ylabel("DTE", labelpad=7)
        ax.set_zlabel("Implied volatility", labelpad=7)
        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.set_box_aspect((1.2, 0.9, 0.7))
        ax.view_init(elev=24, azim=-130)

    fig.suptitle(
        "Representative implied-volatility surfaces and held-out market IV",
        y=0.98,
        fontsize=13,
    )

    return save_figure(fig, "real_data_iv_surface")


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description="Run real option-chain experiment.")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--raw-path", type=Path, default=RAW_CBOE_PATH)
    parser.add_argument("--n-basis-k", type=int, default=N_BASIS_K)
    parser.add_argument("--n-basis-t", type=int, default=N_BASIS_T)
    parser.add_argument("--rbf-bandwidth", type=float, default=RBF_BANDWIDTH)
    parser.add_argument(
        "--lambda-pde",
        default=LAMBDA_PDE,
        help="PDE penalty for BS+RBF+PDE; use 'auto' for training-validation selection.",
    )
    parser.add_argument("--include-pinn", action="store_true")
    parser.add_argument("--pinn-epochs", type=int, default=PINN_EPOCHS)
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    ensure_dir(RAW_DIR)
    ensure_dir(TABLE_DIR)
    ensure_dir(FIG_DIR)

    if not HAS_SKLEARN:
        raise ImportError("scikit-learn is required for this real-data experiment.")

    print_section("Load and clean real option-chain data")
    raw, spot = load_raw_cboe_data(args.raw_path)
    clean = clean_option_data(raw, spot)
    save_dataframe(clean, CLEAN_OUTPUT_PATH, index=False)

    print(f"raw path: {args.raw_path}")
    print(f"spot: {spot:.4f}")
    print(f"raw quotes: {len(raw)}")
    print(f"clean quotes: {len(clean)}")
    print("clean option counts:")
    print(clean["option_type"].value_counts().to_string())

    slices = list(iter_slices(clean))
    split_lookup = {}
    for option_type, bucket_id, sub in slices:
        split_lookup[f"{option_type}_{bucket_id}"] = split_slice(sub, args.seed)

    sample_summary = summarize_sample(slices, split_lookup)
    save_dataframe(sample_summary, SAMPLE_SUMMARY_PATH, index=False)

    print("\nSample summary:")
    print(sample_summary.to_string(index=False))

    benchmark_frames = []
    prediction_frames = []
    iv_smile_frames = []
    iv_surface_frames = []

    for option_type, bucket_id, _ in slices:
        slice_id = f"{option_type}_{bucket_id}"
        train_df, test_df = split_lookup[slice_id]
        bench, preds, iv_smile, iv_surface = fit_and_evaluate_slice(
            option_type=option_type,
            bucket_id=bucket_id,
            train_df=train_df,
            test_df=test_df,
            spot=spot,
            args=args,
        )
        benchmark_frames.append(bench)
        prediction_frames.append(preds)
        iv_smile_frames.append(iv_smile)
        iv_surface_frames.append(iv_surface)

    benchmark = pd.concat(benchmark_frames, ignore_index=True)
    predictions = add_implied_vol_columns(pd.concat(prediction_frames, ignore_index=True))
    iv_smile_curves = pd.concat(iv_smile_frames, ignore_index=True)
    iv_surface_grid = pd.concat(iv_surface_frames, ignore_index=True)

    benchmark["option_type"] = pd.Categorical(
        benchmark["option_type"], OPTION_ORDER, ordered=True
    )
    benchmark["dte_bucket"] = pd.Categorical(
        benchmark["dte_bucket"], BUCKET_ORDER, ordered=True
    )
    benchmark["method"] = pd.Categorical(benchmark["method"], METHOD_ORDER, ordered=True)
    benchmark = benchmark.sort_values(
        ["option_type", "dte_bucket", "rmse", "method"]
    ).reset_index(drop=True)
    benchmark["option_type"] = benchmark["option_type"].astype(str)
    benchmark["dte_bucket"] = benchmark["dte_bucket"].astype(str)
    benchmark["method"] = benchmark["method"].astype(str)

    iv_benchmark = build_iv_benchmark_summary(predictions)

    save_dataframe(benchmark, BENCHMARK_SUMMARY_PATH, index=False)
    save_dataframe(predictions, PREDICTIONS_OUTPUT_PATH, index=False)
    save_dataframe(iv_benchmark, IV_BENCHMARK_SUMMARY_PATH, index=False)
    save_dataframe(iv_smile_curves, IV_SMILE_CURVES_OUTPUT_PATH, index=False)
    save_dataframe(iv_surface_grid, IV_SURFACE_GRID_OUTPUT_PATH, index=False)

    print_section("Real-data benchmark summary")
    print(
        benchmark[
            [
                "slice_label",
                "method",
                "rmse",
                "mae",
                "inside_bid_ask",
                "spread_normalized_mae",
                "share_abs_z_gt_2",
            ]
        ].to_string(index=False)
    )

    print_section("Implied-volatility benchmark summary")
    print(
        iv_benchmark[
            [
                "slice_label",
                "method",
                "price_rmse",
                "iv_rmse",
                "iv_mae",
                "iv_median_abs_error",
                "valid_iv_rate",
            ]
        ].to_string(index=False)
    )

    print("\nSaved outputs:")
    for path in [
        CLEAN_OUTPUT_PATH,
        PREDICTIONS_OUTPUT_PATH,
        IV_SMILE_CURVES_OUTPUT_PATH,
        IV_SURFACE_GRID_OUTPUT_PATH,
        SAMPLE_SUMMARY_PATH,
        BENCHMARK_SUMMARY_PATH,
        IV_BENCHMARK_SUMMARY_PATH,
    ]:
        print(path)

    if not args.skip_plots:
        saved = []
        saved.extend(plot_benchmark_by_slice(benchmark))
        saved.extend(plot_iv_smiles(predictions, iv_smile_curves))
        saved.extend(plot_iv_surface(predictions, iv_surface_grid, iv_benchmark))

        print("\nSaved figures:")
        for path in saved:
            print(path)


if __name__ == "__main__":
    main()
