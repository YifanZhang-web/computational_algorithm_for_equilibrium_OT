"""Adaptive primal-dual accelerated gradient solver for regularized OT problems."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Tuple

import numpy as np

from base import BaseProblem


History = Dict[str, List[float]]


class APDAGD:
    """Adaptive Primal-Dual Accelerated Gradient Descent.

    The solver is written against the :class:`BaseProblem` interface. It can be
    used for both standard quadratically regularized OT and weighted quadratic
    OT, provided the problem class implements the dual objective, gradients,
    primal recovery, line-search condition, and convergence criteria.

    Parameters
    ----------
    epsilon:
        Target tolerance for the two stopping criteria.
    log:
        If ``True``, print sparse progress messages during optimization.
    """

    def __init__(self, epsilon: float = 1e-3, log: bool = False) -> None:
        self.epsilon = float(epsilon)
        self.log = bool(log)

        if self.log:
            print("-----------------------------")
            print("APDAGD configuration")
            print(f"epsilon = {self.epsilon:g}")
            print("-----------------------------\n")

    def fit(
        self,
        problem: BaseProblem,
        L: Optional[float] = 1.0,
        max_iter: Optional[int] = 2000,
    ) -> Tuple[np.ndarray, History]:
        """Solve a regularized OT problem.

        Parameters
        ----------
        problem:
            A concrete ``BaseProblem`` instance.
        L:
            Initial local Lipschitz estimate used by the line search.
        max_iter:
            Maximum number of outer iterations.

        Returns
        -------
        X_hat:
            Recovered primal transport plan.
        history:
            Iteration history with stopping criteria, elapsed time, line-search
            estimate ``L``, and cumulative inner line-search iterations.
        """
        if L is None:
            L = 1.0
        if max_iter is None:
            max_iter = 2000

        L = float(L)
        max_iter = int(max_iter)

        history: DefaultDict[str, List[float]] = defaultdict(list)

        beta = 0.0
        dzeta = np.zeros(2 * problem.n, dtype=float)
        eta_dual = np.zeros(2 * problem.n, dtype=float)
        x_hat = np.zeros((problem.n, problem.n), dtype=float)

        outer_iter = 0
        total_line_search_steps = 0
        start_time = time.perf_counter()

        while True:
            # Try a smaller local Lipschitz estimate first, then increase it
            # until the problem-specific line-search condition is satisfied.
            L /= 2.0

            while True:
                total_line_search_steps += 1

                alpha_new = (1.0 + np.sqrt(4.0 * L * beta + 1.0)) / (2.0 * L)
                beta_new = beta + alpha_new
                tau = alpha_new / beta_new

                kappa = tau * dzeta + (1.0 - tau) * eta_dual
                grad_phi_kappa = np.concatenate(
                    (problem.grad_phi_lambda(kappa), problem.grad_phi_mu(kappa)),
                    axis=0,
                )

                dzeta_new = problem.update_momentum(dzeta, alpha_new, grad_phi_kappa)
                eta_new = tau * dzeta_new + (1.0 - tau) * eta_dual

                phi_eta_new = problem.phi(eta_new)
                phi_approx = problem.phi_approx(kappa, grad_phi_kappa, eta_new, L)

                if problem.apdagd_ls_condition(phi_eta_new, phi_approx):
                    dzeta = dzeta_new.copy()
                    eta_dual = eta_new.copy()
                    beta = beta_new
                    break

                L *= 2.0

            x_hat = tau * problem.X_hat(kappa) + (1.0 - tau) * x_hat

            criteria_one = float(problem.first_crit(x_hat))
            criteria_two = float(problem.second_crit(x_hat, phi_eta_new))
            elapsed = time.perf_counter() - start_time

            history["criteria_one"].append(criteria_one)
            history["criteria_two"].append(criteria_two)
            history["time"].append(elapsed)
            history["L"].append(float(L))
            history["inner_iterations"].append(float(total_line_search_steps))

            if self.log and outer_iter % 1000 == 0:
                print(
                    f"Iteration {outer_iter}: "
                    f"criteria_one={criteria_one:.6g}, "
                    f"criteria_two={criteria_two:.6g}, "
                    f"epsilon={self.epsilon:.6g}"
                )

            converged = criteria_one <= self.epsilon and criteria_two <= self.epsilon
            if converged or outer_iter >= max_iter:
                if self.log:
                    status = "converged" if converged else "reached max_iter"
                    print(
                        f"APDAGD {status} at iteration {outer_iter}: "
                        f"criteria_one={criteria_one:.6g}, "
                        f"criteria_two={criteria_two:.6g}"
                    )
                return x_hat, dict(history)

            outer_iter += 1
