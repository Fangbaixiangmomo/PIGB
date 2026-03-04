# src/models/weak_learners.py
"""
Weak learners for PIGB.

Design goals:
- Provide a consistent interface: fit(X, y), predict(X)
- Support at least:
  (A) Ridge regression over a feature map (smooth baseline; very stable)
  (B) Tree regression (CART) as a more "boosting-style" weak learner

PIGB will depend on these interfaces, but should not care about implementation details.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Protocol
import numpy as np

from .basis_functions import FeatureMap, IdentityMap, _as_2d


class WeakLearner(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> "WeakLearner":
        ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        ...


@dataclass
class RidgeBasisLearner:
    """
    Linear ridge regression on top of a feature map:
        minimize ||Phi w - y||^2 + lam ||w||^2

    This is a great "starter weak learner" because:
    - fast closed form
    - smooth predictions (good for Greeks, PDE residual)
    - stable training

    Later, you can swap in TreeLearner without changing PIGB.
    """
    feature_map: FeatureMap = field(default_factory=IdentityMap)
    lam: float = 1e-6
    fit_intercept_already_in_map: bool = True  # if map already includes bias
    coef_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RidgeBasisLearner":
        X = _as_2d(X)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y size mismatch: {X.shape[0]} vs {y.shape[0]}")

        # Fit map (e.g., centers/scaler)
        self.feature_map.fit(X)
        Phi = self.feature_map.transform(X)  # (n,p)

        # Solve (Phi^T Phi + lam I) w = Phi^T y
        n, p = Phi.shape
        A = Phi.T @ Phi
        A.flat[:: p + 1] += self.lam  # add lam to diagonal
        b = Phi.T @ y

        self.coef_ = np.linalg.solve(A, b)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("RidgeBasisLearner must be fit() before predict().")
        X = _as_2d(X)
        Phi = self.feature_map.transform(X)
        return (Phi @ self.coef_).reshape(-1)


@dataclass
class TreeLearner:
    """
    CART regression tree weak learner.
    Uses scikit-learn DecisionTreeRegressor.

    Note:
    - Trees produce piecewise-constant functions; PDE residual derivatives are not smooth.
      That may be fine for some objectives, but for pricing PDE + Greeks, ridge/RBF is often easier.
    - Still useful as an option because you explicitly mentioned tree learners.

    This is a *minimal wrapper* so PIGB can use it interchangeably.
    """
    max_depth: int = 3
    min_samples_leaf: int = 20
    random_state: int = 0
    _model: Optional[object] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "TreeLearner":
        X = _as_2d(X)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y size mismatch: {X.shape[0]} vs {y.shape[0]}")

        try:
            from sklearn.tree import DecisionTreeRegressor
        except Exception as e:
            raise RuntimeError("scikit-learn is required for TreeLearner. Install scikit-learn.") from e

        self._model = DecisionTreeRegressor(
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("TreeLearner must be fit() before predict().")
        X = _as_2d(X)
        return np.asarray(self._model.predict(X), dtype=float).reshape(-1)
