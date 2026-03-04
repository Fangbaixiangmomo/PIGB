# src/models/basis_functions.py
"""
Basis / feature maps for PIGB.

Design goals:
- Provide a consistent transform(X) -> Phi (n, p) used by weak learners.
- Keep it simple and stable tonight.
- Extensible: add new bases without touching PIGB logic.

Conventions:
- X is (n, d) numpy array.
- For PDE pricing, we typically use X = [S, t] (or [logS, tau]).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Literal
import numpy as np


def _as_2d(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D array, got shape {X.shape}")
    return X


@dataclass
class Standardizer:
    """
    Simple standardizer: z = (x - mean) / std.
    Fit on training inputs; reuse for test inputs.

    You can also choose to standardize only certain columns if desired later.
    """
    mean_: Optional[np.ndarray] = None
    std_: Optional[np.ndarray] = None
    eps: float = 1e-12

    def fit(self, X: np.ndarray) -> "Standardizer":
        X = _as_2d(X)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_ = np.maximum(self.std_, self.eps)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = _as_2d(X)
        if self.mean_ is None or self.std_ is None:
            raise RuntimeError("Standardizer must be fit() before transform().")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


class FeatureMap:
    """
    Abstract feature map.
    """
    def fit(self, X: np.ndarray) -> "FeatureMap":
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


@dataclass
class IdentityMap(FeatureMap):
    """
    Phi(X) = X (optionally with a bias column).
    """
    add_bias: bool = True

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = _as_2d(X)
        if self.add_bias:
            return np.column_stack([np.ones(X.shape[0]), X])
        return X


@dataclass
class PolynomialMap(FeatureMap):
    """
    Simple polynomial features up to total degree 'degree' (no cross-term explosion control),
    with optional bias.
    Intended for quick prototypes; for (S,t) only, it's manageable.

    For d=2:
      degree=2 => [1, x1, x2, x1^2, x1*x2, x2^2]
    """
    degree: int = 2
    add_bias: bool = True
    include_interactions: bool = True
    standardize: bool = True
    _scaler: Standardizer = field(default_factory=Standardizer)

    def fit(self, X: np.ndarray) -> "PolynomialMap":
        X = _as_2d(X)
        if self.standardize:
            self._scaler.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = _as_2d(X)
        if self.standardize:
            X = self._scaler.transform(X)

        n, d = X.shape
        feats = []

        if self.add_bias:
            feats.append(np.ones(n))

        # degree 1
        for j in range(d):
            feats.append(X[:, j])

        if self.degree >= 2:
            # squares
            for j in range(d):
                feats.append(X[:, j] ** 2)

            # interactions (only pairwise for now)
            if self.include_interactions and d >= 2:
                for a in range(d):
                    for b in range(a + 1, d):
                        feats.append(X[:, a] * X[:, b])

        if self.degree >= 3:
            # naive higher powers (no cross terms beyond pairwise)
            for p in range(3, self.degree + 1):
                for j in range(d):
                    feats.append(X[:, j] ** p)

        Phi = np.column_stack(feats)
        return Phi


@dataclass
class RBFMap(FeatureMap):
    """
    Radial basis function features:
        Phi_i(x) = exp(-||x - c_i||^2 / (2*sigma^2))

    Centers c_i are chosen by sampling from training X.
    Standardization recommended for stability.

    Notes:
    - Keep n_centers modest (e.g., 50-300) for fast ridge.
    - For PDE with X=(S,t), consider transforming S to logS and/or tau=T-t upstream.
    """
    n_centers: int = 200
    sigma: float = 1.0
    add_bias: bool = True
    standardize: bool = True
    center_seed: int = 0
    _scaler: Standardizer = field(default_factory=Standardizer)
    centers_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "RBFMap":
        X = _as_2d(X)
        if self.standardize:
            Xs = self._scaler.fit_transform(X)
        else:
            Xs = X

        rng = np.random.default_rng(self.center_seed)
        n = Xs.shape[0]
        m = min(self.n_centers, n)
        idx = rng.choice(n, size=m, replace=False)
        self.centers_ = Xs[idx]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = _as_2d(X)
        if self.standardize:
            Xs = self._scaler.transform(X)
        else:
            Xs = X

        if self.centers_ is None:
            raise RuntimeError("RBFMap must be fit() before transform().")

        # Compute squared distances: (n,m)
        # Using (x-c)^2 = x^2 + c^2 - 2 x·c
        x2 = np.sum(Xs**2, axis=1, keepdims=True)               # (n,1)
        c2 = np.sum(self.centers_**2, axis=1, keepdims=True).T  # (1,m)
        xc = Xs @ self.centers_.T                               # (n,m)
        d2 = x2 + c2 - 2.0 * xc

        Phi_rbf = np.exp(-0.5 * d2 / (self.sigma**2))

        if self.add_bias:
            return np.column_stack([np.ones(X.shape[0]), Phi_rbf])
        return Phi_rbf
