"""Euclidean and weighted Euclidean regularized OT problem definitions."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from base import BaseProblem


class EuclideanRegularizedOTProblem(BaseProblem):
    r"""
    Quadratically regularized optimal transport problem.

    The primal problem is

        min_{X >= 0, X 1 = p, X^T 1 = q}
            <C, X> + eta * ||X||_F^2.

    The coefficient is ``eta`` exactly. It is not written as ``gamma / 2``.
    """

    def f(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        return float(np.sum(self.C * x) + self.eta * np.sum(x**2))

    def shifted_cost(self, lamu: np.ndarray) -> np.ndarray:
        """Return ``C + lambda_i + mu_j`` as an ``(n, n)`` matrix."""
        lamu = np.asarray(lamu, dtype=float)
        lam = lamu[: self.n]
        mu = lamu[self.n :]
        return self.C + lam[:, None] + mu[None, :]

    def positive_part_of_negative_shifted_cost(self, lamu: np.ndarray) -> np.ndarray:
        """Return ``max(-(C + lambda + mu), 0)`` elementwise."""
        return np.maximum(-self.shifted_cost(lamu), 0.0)

    def phi(self, lamu: np.ndarray) -> float:
        s = self.positive_part_of_negative_shifted_cost(lamu)
        dual_regularizer_term = -np.sum(s**2) / (4.0 * self.eta)
        marginal_term = np.dot(lamu[: self.n], self.p) + np.dot(lamu[self.n :], self.q)
        return float(dual_regularizer_term - marginal_term)

    def phi_approx(self, x: np.ndarray, grad_x: np.ndarray, y: np.ndarray, L: float) -> float:
        delta = np.asarray(y, dtype=float) - np.asarray(x, dtype=float)
        return float(self.phi(x) + np.dot(grad_x, delta) - 0.5 * L * np.sum(delta**2))

    def grad_phi_lambda(self, lamu: np.ndarray) -> np.ndarray:
        X = self.X_hat(lamu)
        return np.sum(X, axis=1) - self.p

    def grad_phi_mu(self, lamu: np.ndarray) -> np.ndarray:
        X = self.X_hat(lamu)
        return np.sum(X, axis=0) - self.q

    def update_lambda(self, lamu: np.ndarray) -> np.ndarray:
        """
        Coordinate update for the lambda block.

        This fixed-point style update is used by Sinkhorn-style experiments and
        is kept compatible with the L2-regularized formulation.
        """
        lambda_, mu_ = lamu[: self.n].copy(), lamu[self.n :].copy()

        for i in range(self.n):
            row = self.C[i, :] + mu_
            active = np.where(row + lambda_[i] < 0.0)[0]
            if len(active) > 0:
                lambda_[i] = -(2.0 * self.eta * self.p[i] + np.sum(row[active])) / len(active)
            else:
                lambda_[i] = -2.0 * self.eta * self.p[i]

        out = np.asarray(lamu, dtype=float).copy()
        out[: self.n] = lambda_
        return out

    def update_mu(self, lamu: np.ndarray) -> np.ndarray:
        """
        Coordinate update for the mu block.

        This fixed-point style update is used by Sinkhorn-style experiments and
        is kept compatible with the L2-regularized formulation.
        """
        lambda_, mu_ = lamu[: self.n].copy(), lamu[self.n :].copy()

        for j in range(self.n):
            column = self.C[:, j] + lambda_
            active = np.where(column + mu_[j] < 0.0)[0]
            if len(active) > 0:
                mu_[j] = -(2.0 * self.eta * self.q[j] + np.sum(column[active])) / len(active)
            else:
                mu_[j] = -2.0 * self.eta * self.q[j]

        out = np.asarray(lamu, dtype=float).copy()
        out[self.n :] = mu_
        return out

    def update_momentum(self, x: np.ndarray, alpha: float, grad: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float) + alpha * np.asarray(grad, dtype=float)

    def X_hat(self, lamu: np.ndarray) -> np.ndarray:
        """Recover the primal plan from dual variables."""
        return self.positive_part_of_negative_shifted_cost(lamu) / (2.0 * self.eta)

    def conv_crit(self, lamu: np.ndarray) -> float:
        X = self.X_hat(lamu)
        row_violation = np.sum(np.abs(np.sum(X, axis=1) - self.p))
        col_violation = np.sum(np.abs(np.sum(X, axis=0) - self.q))
        return float(row_violation + col_violation)

    def apdagd_ls_condition(self, phi_x: float, phi_approx_y: float) -> bool:
        return bool(phi_x >= phi_approx_y)

    def pdaam_ls_condition(self, phi_x: float, phi_y: float, term: float) -> bool:
        return bool(phi_x >= phi_y + term)

    def pdaam_delta(self, phi_x: float, phi_y: float) -> float:
        return float(phi_y - phi_x)

    def first_crit(self, X_hat: np.ndarray) -> float:
        """First stopping criterion based on rounding loss."""
        return float(np.sum(self.C * (self.B_round(X_hat) - X_hat)))

    def second_crit(self, X_hat: np.ndarray, phi_eta: float) -> float:
        """Second stopping criterion based on the primal-dual objective gap."""
        return float(abs(self.f(X_hat) - phi_eta))


class WeightedEuclideanRegularizedOTProblem(EuclideanRegularizedOTProblem):
    r"""
    Weighted quadratically regularized OT problem.

    The primal problem is

        min_{X >= 0, X 1 = p, X^T 1 = q}
            <C, X> + eta * sum_{i,j} R[i,j] * X[i,j]^2.

    In the nested OT experiments, this class is used for the outer coupling
    between ``X_1`` and ``Y_1`` with

        R[i,j] = || pi(X_2, Y_2 | X_1=i, Y_1=j) ||_F^2.
    """

    def __init__(
        self,
        eta: float,
        n: int,
        C: np.ndarray,
        p: np.ndarray,
        q: np.ndarray,
        R: np.ndarray,
        min_weight: float = 1e-16,
    ):
        super().__init__(eta=eta, n=n, C=C, p=p, q=q)

        self.R = np.asarray(R, dtype=float)
        if self.R.shape != (self.n, self.n):
            raise ValueError(f"R must have shape {(self.n, self.n)}, got {self.R.shape}.")
        if np.any(self.R < 0.0):
            raise ValueError("R must be elementwise nonnegative.")
        if min_weight <= 0.0:
            raise ValueError("min_weight must be positive.")

        # Avoid division by zero in primal recovery and dual computations.
        self.R_safe = np.maximum(self.R, float(min_weight))

    def f(self, x: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        return float(np.sum(self.C * x) + self.eta * np.sum(self.R_safe * x**2))

    def phi(self, lamu: np.ndarray) -> float:
        s = self.positive_part_of_negative_shifted_cost(lamu)
        dual_regularizer_term = -np.sum((s**2) / self.R_safe) / (4.0 * self.eta)
        marginal_term = np.dot(lamu[: self.n], self.p) + np.dot(lamu[self.n :], self.q)
        return float(dual_regularizer_term - marginal_term)

    def X_hat(self, lamu: np.ndarray) -> np.ndarray:
        """Recover the primal plan for the weighted quadratic regularizer."""
        return self.positive_part_of_negative_shifted_cost(lamu) / (2.0 * self.eta * self.R_safe)

    def update_lambda(self, lamu: np.ndarray) -> np.ndarray:
        """
        Weighted coordinate update for the lambda block.

        This is the L2 Sinkhorn-style update adjusted by the elementwise weights
        through the factors ``1 / R_safe``.
        """
        lambda_, mu_ = lamu[: self.n].copy(), lamu[self.n :].copy()

        for i in range(self.n):
            row = self.C[i, :] + mu_
            active = np.where(row + lambda_[i] < 0.0)[0]
            if len(active) > 0:
                weights = 1.0 / self.R_safe[i, active]
                lambda_[i] = -(
                    2.0 * self.eta * self.p[i] + np.sum(row[active] * weights)
                ) / np.sum(weights)
            else:
                lambda_[i] = -2.0 * self.eta * self.p[i]

        out = np.asarray(lamu, dtype=float).copy()
        out[: self.n] = lambda_
        return out

    def update_mu(self, lamu: np.ndarray) -> np.ndarray:
        """
        Weighted coordinate update for the mu block.

        This is the L2 Sinkhorn-style update adjusted by the elementwise weights
        through the factors ``1 / R_safe``.
        """
        lambda_, mu_ = lamu[: self.n].copy(), lamu[self.n :].copy()

        for j in range(self.n):
            column = self.C[:, j] + lambda_
            active = np.where(column + mu_[j] < 0.0)[0]
            if len(active) > 0:
                weights = 1.0 / self.R_safe[active, j]
                mu_[j] = -(
                    2.0 * self.eta * self.q[j] + np.sum(column[active] * weights)
                ) / np.sum(weights)
            else:
                mu_[j] = -2.0 * self.eta * self.q[j]

        out = np.asarray(lamu, dtype=float).copy()
        out[self.n :] = mu_
        return out


def solve_nested_euclidean_ot(
    mu1: np.ndarray,
    nu1: np.ndarray,
    mu2: np.ndarray,
    nu2: np.ndarray,
    C1: np.ndarray,
    C2: np.ndarray,
    eta: float,
    solver: Any,
    L: float = 1.0,
    max_iter: int = 2000,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    r"""
    Solve the two-layer nested Euclidean-regularized OT problem.

    For each outer pair ``(i, j)``, the function first solves a conditional OT
    problem between ``P(X_2 | X_1=i)`` and ``P(Y_2 | Y_1=j)``. The conditional
    costs and squared Frobenius norms of these inner plans are then used to
    construct the weighted outer OT problem.

    Parameters
    ----------
    mu1, nu1:
        Marginal distributions of ``X_1`` and ``Y_1`` with shape ``(n,)``.
    mu2, nu2:
        Transition matrices with shape ``(n, n)``. The columns ``mu2[:, i]``
        and ``nu2[:, j]`` are the conditional distributions.
    C1, C2:
        Cost matrices for time 1 and time 2, each with shape ``(n, n)``.
    eta:
        Quadratic regularization coefficient.
    solver:
        Any object exposing ``fit(problem, L=..., max_iter=...)``.
    L:
        Initial smoothness estimate passed to compatible solvers.
    max_iter:
        Maximum number of iterations passed to the solver.

    Returns
    -------
    pi1:
        Outer coupling between ``X_1`` and ``Y_1``.
    conditional_plans:
        Array of shape ``(n, n, n, n)``. The slice ``conditional_plans[i, j]``
        is the conditional plan for ``(X_2, Y_2 | X_1=i, Y_1=j)``.
    details:
        Dictionary containing conditional costs, regularization weights,
        effective outer cost, and solver histories.
    """
    mu1 = np.asarray(mu1, dtype=float)
    nu1 = np.asarray(nu1, dtype=float)
    mu2 = np.asarray(mu2, dtype=float)
    nu2 = np.asarray(nu2, dtype=float)
    C1 = np.asarray(C1, dtype=float)
    C2 = np.asarray(C2, dtype=float)

    n = len(mu1)
    _validate_nested_inputs(mu1, nu1, mu2, nu2, C1, C2, n)

    conditional_plans = np.zeros((n, n, n, n), dtype=float)
    conditional_costs = np.zeros((n, n), dtype=float)
    regularization_weights = np.zeros((n, n), dtype=float)
    conditional_histories: Dict[Tuple[int, int], Any] = {}

    for i in range(n):
        for j in range(n):
            problem_2 = EuclideanRegularizedOTProblem(
                eta=eta,
                n=n,
                C=C2,
                p=mu2[:, i],
                q=nu2[:, j],
            )

            pi_2, history_2 = solver.fit(problem_2, L=L, max_iter=max_iter)

            conditional_plans[i, j] = pi_2
            conditional_costs[i, j] = float(np.sum(C2 * pi_2))
            regularization_weights[i, j] = float(np.sum(pi_2**2))
            conditional_histories[(i, j)] = history_2

    effective_C1 = C1 + conditional_costs

    problem_1 = WeightedEuclideanRegularizedOTProblem(
        eta=eta,
        n=n,
        C=effective_C1,
        p=mu1,
        q=nu1,
        R=regularization_weights,
    )

    pi1, history_1 = solver.fit(problem_1, L=L, max_iter=max_iter)

    details: Dict[str, Any] = {
        "conditional_costs": conditional_costs,
        "regularization_weights": regularization_weights,
        "effective_C1": effective_C1,
        "history_1": history_1,
        "conditional_histories": conditional_histories,
    }
    return pi1, conditional_plans, details


def _validate_nested_inputs(
    mu1: np.ndarray,
    nu1: np.ndarray,
    mu2: np.ndarray,
    nu2: np.ndarray,
    C1: np.ndarray,
    C2: np.ndarray,
    n: int,
) -> None:
    """Validate array shapes for the nested OT construction."""
    if nu1.shape != (n,):
        raise ValueError(f"nu1 must have shape {(n,)}, got {nu1.shape}.")
    for name, matrix in (("mu2", mu2), ("nu2", nu2), ("C1", C1), ("C2", C2)):
        if matrix.shape != (n, n):
            raise ValueError(f"{name} must have shape {(n, n)}, got {matrix.shape}.")
    arrays = {"mu1": mu1, "nu1": nu1, "mu2": mu2, "nu2": nu2, "C1": C1, "C2": C2}
    for name, array in arrays.items():
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must contain only finite values.")
    if np.any(mu1 < 0.0) or np.any(nu1 < 0.0) or np.any(mu2 < 0.0) or np.any(nu2 < 0.0):
        raise ValueError("All marginal and transition probabilities must be nonnegative.")
