"""Dual L-BFGS solver for eta-quadratically regularized OT problems."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from base import BaseProblem


History = Dict[str, List[float]]


class DualL2Solver:
    """SciPy L-BFGS-B dual solver for L2-regularized OT.

    The solver assumes the primal objective has the form

        <C, X> + eta * sum_ij R_ij * X_ij^2,

    where ``R`` is read from ``problem.R_safe`` if present. If the problem does
    not provide ``R_safe``, the unweighted case ``R_ij = 1`` is used.

    Parameters
    ----------
    epsilon:
        Tolerance passed to SciPy's optimizer and used when reporting criteria.
    method:
        Optimization method. The default is ``L-BFGS-B``.
    max_iter:
        Default maximum number of optimizer iterations.
    log:
        If ``True``, print the optimizer status and final criteria.
    """

    def __init__(
        self,
        epsilon: float = 1e-3,
        method: str = "L-BFGS-B",
        max_iter: int = 500,
        log: bool = False,
    ) -> None:
        self.epsilon = float(epsilon)
        self.method = str(method)
        self.max_iter = int(max_iter)
        self.log = bool(log)

    @staticmethod
    def _weights(problem: BaseProblem) -> np.ndarray:
        """Return regularization weights for standard or weighted L2 OT."""
        if hasattr(problem, "R_safe"):
            return np.asarray(problem.R_safe, dtype=float)
        return np.ones((problem.n, problem.n), dtype=float)

    @staticmethod
    def _dual_obj_grad(lamu: np.ndarray, problem: BaseProblem) -> Tuple[float, np.ndarray]:
        """Evaluate the dual objective and gradient under the project convention."""
        n = problem.n
        lam = lamu[:n]
        mu = lamu[n:]
        weights = DualL2Solver._weights(problem)

        shifted = problem.C + lam[:, None] + mu[None, :]
        positive_negative_part = np.maximum(-shifted, 0.0)

        # For min_{x >= 0} a*x + eta*R*x^2, the dual contribution is
        # -max(-a, 0)^2 / (4*eta*R).
        phi = -np.sum((positive_negative_part**2) / (4.0 * problem.eta * weights))
        phi -= float(np.dot(lam, problem.p) + np.dot(mu, problem.q))

        x_hat = positive_negative_part / (2.0 * problem.eta * weights)
        grad_lam = np.sum(x_hat, axis=1) - problem.p
        grad_mu = np.sum(x_hat, axis=0) - problem.q
        grad = np.concatenate([grad_lam, grad_mu])

        return float(phi), grad

    def fit(
        self,
        problem: BaseProblem,
        L: Optional[float] = None,
        max_iter: Optional[int] = None,
    ) -> Tuple[np.ndarray, History]:
        """Solve a regularized OT problem by optimizing the dual objective.

        Parameters
        ----------
        problem:
            A concrete ``BaseProblem`` instance.
        L:
            Ignored. Kept for API compatibility with APDAGD.
        max_iter:
            Maximum number of SciPy optimizer iterations. If omitted, the
            instance-level ``max_iter`` is used.

        Returns
        -------
        X_hat:
            Recovered primal transport plan.
        history:
            Final optimization diagnostics and stopping criteria.
        """
        del L

        if max_iter is None:
            max_iter = self.max_iter
        max_iter = int(max_iter)

        history: DefaultDict[str, List[float]] = defaultdict(list)
        start_time = time.perf_counter()

        def objective(lamu: np.ndarray) -> Tuple[float, np.ndarray]:
            phi, grad = self._dual_obj_grad(lamu, problem)
            # SciPy minimizes; the OT dual is maximized.
            return -phi, -grad

        initial_dual = np.zeros(2 * problem.n, dtype=float)
        result = minimize(
            objective,
            initial_dual,
            method=self.method,
            jac=True,
            tol=self.epsilon,
            options={"maxiter": max_iter, "disp": False},
        )

        x_hat = problem.X_hat(result.x)
        criteria_one = float(problem.first_crit(x_hat))
        phi, _ = self._dual_obj_grad(result.x, problem)
        criteria_two = float(problem.second_crit(x_hat, phi))

        history["criteria_one"].append(criteria_one)
        history["criteria_two"].append(criteria_two)
        history["time"].append(time.perf_counter() - start_time)
        history["nit"].append(float(getattr(result, "nit", 0)))
        history["nfev"].append(float(getattr(result, "nfev", 0)))
        history["success"].append(float(bool(getattr(result, "success", False))))
        history["message"].append(str(getattr(result, "message", "")))

        if self.log:
            print(
                "DualL2Solver finished: "
                f"success={result.success}, nit={result.nit}, message={result.message}"
            )
            print(
                f"criteria_one={criteria_one:.6g}, "
                f"criteria_two={criteria_two:.6g}"
            )

        return x_hat, dict(history)
