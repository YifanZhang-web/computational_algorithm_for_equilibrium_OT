"""Coordinate-scaling style solver for L2-regularized OT problems."""

from __future__ import annotations

import time
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Tuple, Union

import numpy as np

from base import BaseProblem


History = Dict[str, List[float]]


class Sinkhorn:
    """Sinkhorn-type coordinate update solver for L2-regularized OT.

    This implementation delegates the actual lambda/mu updates to the supplied
    ``BaseProblem`` instance. Therefore, the same solver can be used for the
    standard L2 problem and the weighted L2 problem, provided the problem class
    implements ``update_lambda``, ``update_mu``, ``X_hat``, and ``conv_crit``
    with the correct regularization weights.

    Parameters
    ----------
    epsilon:
        Target tolerance for the marginal violation criterion.
    log:
        If ``True``, print sparse progress messages during optimization.
    """

    def __init__(self, epsilon: float = 1e-3, log: bool = False) -> None:
        self.epsilon = float(epsilon)
        self.log = bool(log)

        if self.log:
            print("-----------------------------")
            print("Sinkhorn configuration")
            print(f"epsilon = {self.epsilon:g}")
            print("-----------------------------\n")

    def fit(
        self,
        problem: BaseProblem,
        L: Optional[Union[float, np.ndarray]] = None,
        max_iter: Optional[int] = 2000,
        lamu: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, History]:
        """Solve a regularized OT problem by alternating dual updates.

        Parameters
        ----------
        problem:
            A concrete ``BaseProblem`` instance.
        L:
            Ignored. Kept for API compatibility with other solvers. For
            backward compatibility, if an ndarray is passed here it is treated
            as the initial dual vector ``lamu``.
        max_iter:
            Maximum number of alternating lambda/mu updates.
        lamu:
            Optional initial dual vector of shape ``(2 * problem.n,)``. If not
            provided, a zero vector is used.

        Returns
        -------
        X_hat:
            Recovered primal transport plan.
        history:
            Iteration history with marginal violation and elapsed time.
        """
        # Backward compatibility with the old signature:
        # fit(problem, lamu, max_iter=...).
        if isinstance(L, np.ndarray):
            if lamu is not None:
                raise ValueError("Initial dual vector was provided twice.")
            lamu = L
            L = None
        del L

        if max_iter is None:
            max_iter = 2000
        max_iter = int(max_iter)

        if lamu is None:
            dual = np.zeros(2 * problem.n, dtype=float)
        else:
            dual = np.asarray(lamu, dtype=float).copy()
            expected_shape = (2 * problem.n,)
            if dual.shape != expected_shape:
                raise ValueError(f"lamu must have shape {expected_shape}, got {dual.shape}.")

        history: DefaultDict[str, List[float]] = defaultdict(list)
        start_time = time.perf_counter()

        iteration = 0
        while True:
            if iteration % 2 == 0:
                dual = problem.update_lambda(dual)
            else:
                dual = problem.update_mu(dual)

            conv_crit = float(problem.conv_crit(dual))
            elapsed = time.perf_counter() - start_time

            history["conv_crit"].append(conv_crit)
            history["time"].append(elapsed)

            if self.log and iteration % 1000 == 0:
                print(
                    f"Iteration {iteration}: "
                    f"conv_crit={conv_crit:.6g}, epsilon={self.epsilon:.6g}"
                )

            converged = conv_crit < self.epsilon
            if converged or iteration >= max_iter:
                if self.log:
                    status = "converged" if converged else "reached max_iter"
                    print(
                        f"Sinkhorn {status} at iteration {iteration}: "
                        f"conv_crit={conv_crit:.6g}"
                    )
                return problem.X_hat(dual), dict(history)

            iteration += 1
