# src/models/pigb.py
"""
Physics-Informed Gradient Boosting (PIGB) with Mixed Learners
============================================================

Implements Algorithm 1 (your pseudocode):

Given:
  - interior collocation set D_PDE
  - boundary/terminal set D_BC
  - interior weak learner class H_int (smooth)
  - boundary weak learner class H_bdry (trees)
  - learning rates {nu_b^int}, {nu_b^bdry}
  - boosting rounds B
  - weights {w_i^PDE}, {w_i^BC}

Initialize f0(z) = 0 (or prior)
For b = 1..B:
  Interior step:
    R_PDE(z_i) = L f_{b-1}(z_i), z_i in D_PDE
    g_b^int(z_i) = - R_PDE(z_i)
    Fit h_b^int approx g_b^int by weighted least squares
    Update f_{b-1/2} = f_{b-1} + nu_b^int * h_b^int

  Boundary step:
    r_BC(z_i) = g(z_i) - f_{b-1/2}(z_i), z_i in D_BC
    g_b^bdry(z_i) = r_BC(z_i)
    Fit h_b^bdry approx g_b^bdry by weighted least squares
    Update f_b = f_{b-1/2} + nu_b^bdry * h_b^bdry

Output:
  f_B

Notes:
- This file is intentionally "PDE-agnostic".
  You must supply:
    * pde_residual_fn(predict_fn, X_pde) -> residual vector
    * bc_target_fn(X_bc) -> g(X_bc)
- For Black–Scholes, pde_residual_fn can be implemented via finite differences
  on predict_fn (tonight) and later upgraded to analytic derivatives.

Interfaces:
- Learners are expected to implement:
    fit(X, y) and predict(X)
  Optionally support:
    fit(X, y, sample_weight=...)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Union, Any
import inspect
import numpy as np


Array = np.ndarray


def _as_2d(X: Array) -> Array:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D array, got shape {X.shape}")
    return X

# ----------------------------
# Sampling utilities (simple baseline)
# ----------------------------

def sample_interior_logS_uniform(
    n: int,
    T: float,
    S_min: float,
    S_max: float,
    rng: np.random.Generator,
) -> Array:
    """
    Sample interior PDE collocation points (S,t):
      t ~ Uniform(0, T)
      log S ~ Uniform(log S_min, log S_max)

    Returns X of shape (n,2) with columns [S, t].
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if S_min <= 0 or S_max <= 0 or S_min >= S_max:
        raise ValueError("Require 0 < S_min < S_max.")
    if T <= 0:
        raise ValueError("Require T > 0.")

    t = rng.uniform(0.0, T, size=n)
    logS = rng.uniform(np.log(S_min), np.log(S_max), size=n)
    S = np.exp(logS)
    return np.column_stack([S, t])


def sample_terminal_logS_uniform(
    n: int,
    T: float,
    S_min: float,
    S_max: float,
    rng: np.random.Generator,
) -> Array:
    """
    Sample terminal/boundary points at t = T:
      log S ~ Uniform(log S_min, log S_max), t = T

    Returns X of shape (n,2) with columns [S, t].
    """
    if n <= 0:
        raise ValueError("n must be positive.")
    if S_min <= 0 or S_max <= 0 or S_min >= S_max:
        raise ValueError("Require 0 < S_min < S_max.")
    if T <= 0:
        raise ValueError("Require T > 0.")

    logS = rng.uniform(np.log(S_min), np.log(S_max), size=n)
    S = np.exp(logS)
    t = np.full(n, T, dtype=float)
    return np.column_stack([S, t])

def _as_1d(y: Array) -> Array:
    y = np.asarray(y, dtype=float).reshape(-1)
    return y


def _expand_lr(lr: Union[float, Sequence[float]], B: int, name: str) -> List[float]:
    """
    Expand a scalar learning rate into a per-round list, or validate list length.
    """
    if isinstance(lr, (float, int)):
        return [float(lr)] * B
    lr_list = [float(x) for x in lr]
    if len(lr_list) != B:
        raise ValueError(f"{name} must have length B={B}, got {len(lr_list)}.")
    return lr_list


def _maybe_fit_with_weights(learner: Any, X: Array, y: Array, w: Optional[Array]) -> Any:
    """
    Try to fit learner with weights, if supported.

    Order:
      1) if w is None: learner.fit(X, y)
      2) if learner.fit supports sample_weight: learner.fit(X, y, sample_weight=w)
      3) else: learner.fit(X, y) (warn handled outside)
    """
    if w is None:
        return learner.fit(X, y)

    # Try signature-based detection
    try:
        sig = inspect.signature(learner.fit)
        if "sample_weight" in sig.parameters:
            return learner.fit(X, y, sample_weight=w)
    except Exception:
        # If inspect fails, fall back
        pass

    # Last attempt: call with sample_weight and catch
    try:
        return learner.fit(X, y, sample_weight=w)
    except TypeError:
        return learner.fit(X, y)


@dataclass
class PIGB:
    """
    Physics-Informed Gradient Boosting with mixed learners.

    Parameters
    ----------
    B : int
        Number of boosting rounds.
    interior_learner_factory : callable
        A function that returns a *fresh* interior weak learner (smooth).
        Example: lambda: RidgeBasisLearner(RBFMap(...), lam=...)
    boundary_learner_factory : callable
        A function that returns a *fresh* boundary weak learner (trees).
        Example: lambda: TreeLearner(max_depth=3, ...)
    pde_residual_fn : callable
        Function computing PDE residuals on interior points:
            pde_residual_fn(predict_fn, X_pde) -> residual vector (n_pde,)
        This implements R_PDE(z) = L f(z) (or your chosen PDE residual).
    bc_target_fn : callable
        Function giving boundary/terminal targets:
            bc_target_fn(X_bc) -> g vector (n_bc,)
    nu_int : float or list[float]
        Interior learning rate(s) nu_b^int.
    nu_bdry : float or list[float]
        Boundary learning rate(s) nu_b^bdry.
    f0 : callable or None
        Initial function f0(X) -> vector. If None, starts from 0.
    verbose : bool
        If True, print a few diagnostics.
    """

    B: int
    interior_learner_factory: Callable[[], Any]
    boundary_learner_factory: Callable[[], Any]
    pde_residual_fn: Callable[[Callable[[Array], Array], Array], Array]
    bc_target_fn: Callable[[Array], Array]
    nu_int: Union[float, Sequence[float]] = 0.1
    nu_bdry: Union[float, Sequence[float]] = 0.1
    f0: Optional[Callable[[Array], Array]] = None
    verbose: bool = False

    # learned components
    interior_learners_: Optional[List[Any]] = None
    boundary_learners_: Optional[List[Any]] = None
    nu_int_list_: Optional[List[float]] = None
    nu_bdry_list_: Optional[List[float]] = None

    # internal bookkeeping
    _warned_weights_int: bool = False
    _warned_weights_bdry: bool = False

    def fit(
        self,
        X_pde: Array,
        X_bc: Array,
        w_pde: Optional[Array] = None,
        w_bc: Optional[Array] = None,
        eval_callback: Optional[Callable[[int, "PIGB"], None]] = None,
    ) -> "PIGB":
        """
        Fit PIGB using Algorithm 1.

        Inputs
        ------
        X_pde : (n_pde, d) array
            Interior collocation points.
        X_bc : (n_bc, d) array
            Boundary/terminal points.
        w_pde : (n_pde,) array or None
            Optional interior weights.
        w_bc : (n_bc,) array or None
            Optional boundary weights.
        eval_callback : callable or None
            If provided, called as eval_callback(b, self) at end of each round.
        """
        X_pde = _as_2d(X_pde)
        X_bc = _as_2d(X_bc)

        if w_pde is not None:
            w_pde = _as_1d(w_pde)
            if w_pde.shape[0] != X_pde.shape[0]:
                raise ValueError("w_pde length mismatch with X_pde.")
        if w_bc is not None:
            w_bc = _as_1d(w_bc)
            if w_bc.shape[0] != X_bc.shape[0]:
                raise ValueError("w_bc length mismatch with X_bc.")

        self.nu_int_list_ = _expand_lr(self.nu_int, self.B, "nu_int")
        self.nu_bdry_list_ = _expand_lr(self.nu_bdry, self.B, "nu_bdry")

        self.interior_learners_ = []
        self.boundary_learners_ = []

        # Precompute boundary targets g(z) (does not depend on f)
        g_bc = _as_1d(self.bc_target_fn(X_bc))
        if g_bc.shape[0] != X_bc.shape[0]:
            raise ValueError("bc_target_fn(X_bc) must return shape (n_bc,).")

        # Define predict_fn closures for residual calculation
        def predict_fn(X: Array) -> Array:
            return self.predict(X)

        for b in range(1, self.B + 1):
            # --------------------------
            # Interior step (PDE residuals)
            # --------------------------
            r_pde = _as_1d(self.pde_residual_fn(predict_fn, X_pde))  # R_PDE(z) = L f(z)
            if r_pde.shape[0] != X_pde.shape[0]:
                raise ValueError("pde_residual_fn must return shape (n_pde,).")

            g_int = -r_pde  # pseudo-residuals

            h_int = self.interior_learner_factory()
            # Try weighted fit if supported
            before = h_int
            h_int = _maybe_fit_with_weights(h_int, X_pde, g_int, w_pde)
            if (w_pde is not None) and (not self._warned_weights_int):
                # Detect if weights were ignored by checking signature
                try:
                    sig = inspect.signature(before.fit)
                    if "sample_weight" not in sig.parameters:
                        if self.verbose:
                            print("[PIGB] Note: interior learner.fit() has no sample_weight; weights may be ignored.")
                        self._warned_weights_int = True
                except Exception:
                    # Can't inspect; just warn once in verbose mode
                    if self.verbose:
                        print("[PIGB] Note: could not verify sample_weight support for interior learner.")
                    self._warned_weights_int = True

            self.interior_learners_.append(h_int)

            # Update f_{b-1/2} = f_{b-1} + nu_b^int h_b^int
            # (implemented implicitly via predict() summation)

            # --------------------------
            # Boundary step (BC residuals)
            # --------------------------
            # f_{b-1/2}(X_bc) = current predict including newly-added interior learner
            f_half = self.predict(X_bc)

            r_bc = g_bc - f_half
            g_bdry = r_bc

            h_bdry = self.boundary_learner_factory()
            before = h_bdry
            h_bdry = _maybe_fit_with_weights(h_bdry, X_bc, g_bdry, w_bc)
            if (w_bc is not None) and (not self._warned_weights_bdry):
                try:
                    sig = inspect.signature(before.fit)
                    if "sample_weight" not in sig.parameters:
                        if self.verbose:
                            print("[PIGB] Note: boundary learner.fit() has no sample_weight; weights may be ignored.")
                        self._warned_weights_bdry = True
                except Exception:
                    if self.verbose:
                        print("[PIGB] Note: could not verify sample_weight support for boundary learner.")
                    self._warned_weights_bdry = True

            self.boundary_learners_.append(h_bdry)

            # Update f_b = f_{b-1/2} + nu_b^bdry h_b^bdry (implicit)

            if self.verbose and (b == 1 or b % max(1, self.B // 10) == 0):
                # Light diagnostics: interior residual RMS and boundary RMS
                rpde_rms = float(np.sqrt(np.mean(r_pde**2)))
                rbc_rms = float(np.sqrt(np.mean(r_bc**2)))
                print(f"[PIGB round {b:4d}] PDE_resid_RMS={rpde_rms:.4g}  BC_resid_RMS={rbc_rms:.4g}")

            if eval_callback is not None:
                eval_callback(b, self)

        return self

    def predict(self, X: Array) -> Array:
        """
        Predict f(X) after training (or during training).

        f(X) = f0(X) + sum_{k=1..B} nu_k^int h_k^int(X) + sum_{k=1..B} nu_k^bdry h_k^bdry(X)

        During fitting, the lists may be partially filled; this still works.
        """
        X = _as_2d(X)
        n = X.shape[0]

        # Base function
        if self.f0 is None:
            y = np.zeros(n, dtype=float)
        else:
            y = _as_1d(self.f0(X))
            if y.shape[0] != n:
                raise ValueError("f0(X) must return shape (n,).")

        if self.interior_learners_ is None or self.boundary_learners_ is None:
            return y

        # Determine how many rounds are currently available
        B_int = len(self.interior_learners_)
        B_bdry = len(self.boundary_learners_)
        B_now = min(B_int, B_bdry)

        # Add contributions round by round (to preserve f_{b-1/2} meaning)
        for k in range(B_now):
            nu_i = self.nu_int_list_[k] if self.nu_int_list_ is not None else float(self.nu_int)
            nu_b = self.nu_bdry_list_[k] if self.nu_bdry_list_ is not None else float(self.nu_bdry)

            y = y + nu_i * _as_1d(self.interior_learners_[k].predict(X))
            y = y + nu_b * _as_1d(self.boundary_learners_[k].predict(X))

        # If interior learners exist but boundary not yet appended (rare), include remaining interior
        for k in range(B_now, B_int):
            nu_i = self.nu_int_list_[k] if self.nu_int_list_ is not None else float(self.nu_int)
            y = y + nu_i * _as_1d(self.interior_learners_[k].predict(X))

        return y

    def staged_predict(self, X: Array) -> List[Array]:
        """
        Return predictions after each *full round* (after both interior + boundary update).
        Useful for plotting training curves.

        Returns a list of length B_now, where entry b-1 is f_b(X).
        """
        X = _as_2d(X)
        preds = []
        if self.interior_learners_ is None or self.boundary_learners_ is None:
            return preds

        n = X.shape[0]
        if self.f0 is None:
            y = np.zeros(n, dtype=float)
        else:
            y = _as_1d(self.f0(X))

        B_now = min(len(self.interior_learners_), len(self.boundary_learners_))
        for k in range(B_now):
            nu_i = self.nu_int_list_[k] if self.nu_int_list_ is not None else float(self.nu_int)
            nu_b = self.nu_bdry_list_[k] if self.nu_bdry_list_ is not None else float(self.nu_bdry)
            y = y + nu_i * _as_1d(self.interior_learners_[k].predict(X))
            y = y + nu_b * _as_1d(self.boundary_learners_[k].predict(X))
            preds.append(y.copy())
        return preds