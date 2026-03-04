import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.basis_functions import RBFMap
from src.models.weak_learners import RidgeBasisLearner, TreeLearner

def log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    t0 = time.perf_counter()
    log("[1/4] Generating synthetic data...")
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 2))
    y = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2
    log(f"      done in {time.perf_counter() - t0:.3f}s")

    t1 = time.perf_counter()
    log("[2/4] Fitting RidgeBasisLearner...")
    ridge = RidgeBasisLearner(feature_map=RBFMap(n_centers=100, sigma=1.0), lam=1e-4).fit(X, y)
    log(f"      done in {time.perf_counter() - t1:.3f}s")

    t2 = time.perf_counter()
    log("[3/4] Fitting TreeLearner...")
    tree = TreeLearner(max_depth=3, min_samples_leaf=10).fit(X, y)
    log(f"      done in {time.perf_counter() - t2:.3f}s")

    t3 = time.perf_counter()
    log("[4/4] Predicting...")
    print("ridge pred mean:", ridge.predict(X).mean())
    print("tree pred mean:", tree.predict(X).mean())
    log(f"      done in {time.perf_counter() - t3:.3f}s")
    log(f"total runtime: {time.perf_counter() - t0:.3f}s")


if __name__ == "__main__":
    main()
