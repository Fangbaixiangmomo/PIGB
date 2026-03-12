# Experiment Plan for PIGB (Physics-Informed Gradient Boosting)

## Goal

Evaluate whether the proposed **Physics-Informed Gradient Boosting (PIGB)** algorithm can outperform traditional numerical PDE solvers and modern ML-based solvers in terms of:

- Pricing accuracy
- Sensitivity (Greeks) accuracy
- PDE consistency
- Computational efficiency
- Robustness to hyperparameters and randomness
- Performance under non-smooth payoffs and constraints

The experiments should highlight situations where **mesh-based PDE methods or neural PDE solvers struggle**, such as non-smooth payoffs, free boundaries, or higher dimensions.

---

# 1. Accuracy Metrics

## 1.1 Price Error

If a closed-form solution exists (e.g., Black–Scholes European option):

\[
\text{RMSE}_u =
\sqrt{
\frac{1}{M}
\sum_{m=1}^{M}
(\hat{u}(z_m) - u^\*(z_m))^2
}
\]

Where:

- \(u^\*(z)\) = analytical solution
- \(\hat{u}(z)\) = predicted solution
- \(z\) = state variable (e.g. \(S,t\))

If no closed-form solution exists:

- use a **high-resolution finite difference or finite element solution** as reference.

---

## 1.2 Greeks Error

Compute accuracy for sensitivities:

- **Delta**

\[
\Delta = \frac{\partial u}{\partial S}
\]

- **Gamma**

\[
\Gamma = \frac{\partial^2 u}{\partial S^2}
\]

Metric:

\[
\text{RMSE}_{\Delta}, \quad \text{RMSE}_{\Gamma}
\]

These are critical because:

- traders rely on Greeks
- numerical errors often appear in derivatives before prices.

---

## 1.3 Worst-case Error

To capture errors near kinks and boundaries:

- Maximum absolute error

\[
\max_m |\hat{u}(z_m) - u^\*(z_m)|
\]

- 95th percentile absolute error

This helps detect instability near:

- payoff kinks
- barrier regions
- exercise boundaries.

---

# 2. PDE Consistency Metrics

Since PIGB minimizes PDE residuals, we should measure them directly.

---

## 2.1 Interior PDE Residual

Evaluate:

\[
\text{Residual RMS} =
\sqrt{
\frac{1}{N}
\sum_{i=1}^{N}
|L\hat{u}(x_i,t_i)|^2
}
\]

Where:

- \(L\) = differential operator of the PDE
- \((x_i,t_i)\) = interior collocation points.

Lower values indicate better PDE satisfaction.

---

## 2.2 Boundary / Terminal Condition Violation

Measure how well boundary conditions are satisfied:

\[
\text{BC Error} =
\frac{1}{N}
\sum_{i=1}^{N}
|\hat{u}(z_i) - g(z_i)|
\]

Where:

- \(g(z)\) = boundary or terminal payoff function.

---

## 2.3 American Option Constraint Violation (optional)

For American options:

\[
u(S,t) \ge \phi(S)
\]

Measure violation:

\[
\max(0, \phi(S) - \hat{u}(S,t))
\]

This quantifies incorrect early-exercise regions.

---

# 3. Robustness Metrics

Many ML solvers (especially PINNs) suffer from instability.

We explicitly test robustness.

---

## 3.1 Hyperparameter Sensitivity

Vary:

- learning rate
- number of boosting rounds
- RBF width
- tree depth
- boundary penalty weight

Metric:

- standard deviation of final RMSE across hyperparameters.

---

## 3.2 Random Seed Sensitivity

Run the algorithm with multiple seeds.

Report:

| metric | mean | std |
|------|------|------|
| price RMSE | | |
| delta RMSE | | |
| PDE residual | | |

Lower variance indicates higher stability.

---

# 4. Efficiency Metrics

These metrics determine whether the method is computationally competitive.

---

## 4.1 Wall-clock Time

Measure:

- total runtime
- runtime per boosting round

Plot:

Error v.s. Runtime

This is often the most meaningful comparison.

---

## 4.2 Sample Efficiency

Plot:

Error vs Number of Collocation Points


If PIGB requires fewer points than grid methods, this demonstrates efficiency.

---

## 4.3 Dimensional Scaling

Test increasing dimensionality:

- 1D
- 2D
- 5D (toy example)

Report:

| dimension | error | runtime |
|----------|------|------|

This highlights the **curse-of-dimensionality advantage** of mesh-free methods.

---

# 5. Benchmark Problems

## Experiment 1 — European Black–Scholes (1D)

Purpose:

- sanity check
- verify convergence to analytical solution.

Compare:

- PIGB
- Finite Difference
- Finite Element
- PINN (optional)

Metrics:

- price RMSE
- delta RMSE
- runtime.

---

## Experiment 2 — American Put (Free Boundary)

Purpose:

- test inequality constraints
- test non-smooth payoff.

Reference:

- finite difference with PSOR solver.

Metrics:

- price RMSE
- exercise violation
- runtime.

---

## Experiment 3 — 2D Basket Option

Purpose:

- demonstrate scalability.

Compare:

- PIGB
- coarse-grid finite difference
- PINN (optional).

Metrics:

- price RMSE
- runtime
- scaling behavior.

---

# 6. Recommended Plots

Produce the following figures:

1. Error vs runtime
2. Error vs collocation points
3. Delta error vs runtime
4. PDE residual vs boosting rounds
5. Error near payoff kink
6. Dimensional scaling plot

---

# 7. Expected Strengths of PIGB

The method is expected to outperform traditional solvers when:

- domain geometry is irregular
- payoff functions are non-smooth
- dimensionality increases
- mesh generation becomes expensive
- PDE residual minimization is advantageous.

---

# 8. Possible Weaknesses

PIGB may struggle when:

- extremely high accuracy is required
- low-dimensional PDEs where classical methods are highly optimized
- training time becomes large due to boosting iterations.

---

# 9. Minimal Experiment Set (Tonight)

If time is limited, prioritize:

1. European Black–Scholes (1D)
2. American Put (1D)
3. Error vs runtime plots

These three results alone can already demonstrate:

- correctness
- efficiency
- ability to handle constraints.
