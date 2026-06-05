"""Base abstractions for quadratically regularized optimal transport problems."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseProblem(ABC):
    r"""
    Abstract base class for regularized OT problems with two dual variables.

    The transport constraints are

        X 1 = p,
        X^T 1 = q,
        X >= 0.

    Subclasses specify the regularizer, the corresponding dual objective,
    primal recovery map, coordinate updates, and convergence criteria used by
    the solvers in this project.
    """

    def __init__(self, eta: float, n: int, C: np.ndarray, p: np.ndarray, q: np.ndarray):
        """
        Parameters
        ----------
        eta:
            Regularization coefficient. In the quadratic cases used here, the
            primal objective contains ``eta * ||X||_F^2`` rather than
            ``gamma / 2 * ||X||_F^2``.
        n:
            Number of support points for each marginal.
        C:
            Cost matrix with shape ``(n, n)``.
        p, q:
            Marginal probability vectors with shape ``(n,)``.
        """
        self.eta = float(eta)

        # Backward-compatible alias for older code that still refers to gamma.
        self.gamma = self.eta

        self.n = int(n)
        self.C = np.asarray(C, dtype=float)
        self.p = np.asarray(p, dtype=float)
        self.q = np.asarray(q, dtype=float)
        self.one = np.ones(self.n, dtype=float)

        self._validate_inputs()

    def _validate_inputs(self) -> None:
        """Validate basic dimensions and numerical assumptions."""
        if self.n <= 0:
            raise ValueError("n must be a positive integer.")
        if self.eta <= 0.0:
            raise ValueError("eta must be positive.")
        if self.C.shape != (self.n, self.n):
            raise ValueError(f"C must have shape {(self.n, self.n)}, got {self.C.shape}.")
        if self.p.shape != (self.n,):
            raise ValueError(f"p must have shape {(self.n,)}, got {self.p.shape}.")
        if self.q.shape != (self.n,):
            raise ValueError(f"q must have shape {(self.n,)}, got {self.q.shape}.")
        if np.any(self.p < 0.0) or np.any(self.q < 0.0):
            raise ValueError("p and q must be elementwise nonnegative.")
        if not np.all(np.isfinite(self.C)):
            raise ValueError("C must contain only finite values.")
        if not np.all(np.isfinite(self.p)) or not np.all(np.isfinite(self.q)):
            raise ValueError("p and q must contain only finite values.")

    def B_round(self, x: np.ndarray) -> np.ndarray:
        r"""
        Round a nonnegative matrix approximately onto the transport polytope.

        This is the standard row/column correction used to obtain a feasible
        coupling from an approximately feasible nonnegative matrix.
        """
        x = np.asarray(x, dtype=float).copy()
        if x.shape != (self.n, self.n):
            raise ValueError(f"x must have shape {(self.n, self.n)}, got {x.shape}.")

        x[x == 0.0] = 1e-16

        row_scale = self.p / x.dot(self.one)
        row_scale[row_scale > 1.0] = 1.0
        F = np.diag(row_scale).dot(x)

        col_scale = self.q / F.T.dot(self.one)
        col_scale[col_scale > 1.0] = 1.0
        F = F.dot(np.diag(col_scale))

        row_error = self.p - F.dot(self.one)
        col_error = self.q - F.T.dot(self.one)

        denom = np.abs(row_error).sum()
        if denom == 0.0:
            return F

        return F + np.outer(row_error, col_error) / denom

    @abstractmethod
    def f(self, x: np.ndarray) -> float:
        """Return the primal objective value at ``x``."""
        raise NotImplementedError

    @abstractmethod
    def phi(self, lamu: np.ndarray) -> float:
        """Return the dual objective value at ``lamu = [lambda, mu]``."""
        raise NotImplementedError

    @abstractmethod
    def phi_approx(self, x: np.ndarray, grad_x: np.ndarray, y: np.ndarray, L: float) -> float:
        """Return the quadratic model of ``phi`` around ``x`` evaluated at ``y``."""
        raise NotImplementedError

    @abstractmethod
    def grad_phi_lambda(self, lamu: np.ndarray) -> np.ndarray:
        """Return the gradient of the dual objective with respect to lambda."""
        raise NotImplementedError

    @abstractmethod
    def grad_phi_mu(self, lamu: np.ndarray) -> np.ndarray:
        """Return the gradient of the dual objective with respect to mu."""
        raise NotImplementedError

    @abstractmethod
    def update_lambda(self, lamu: np.ndarray) -> np.ndarray:
        """Return ``lamu`` after updating the lambda block."""
        raise NotImplementedError

    @abstractmethod
    def update_mu(self, lamu: np.ndarray) -> np.ndarray:
        """Return ``lamu`` after updating the mu block."""
        raise NotImplementedError

    @abstractmethod
    def update_momentum(self, x: np.ndarray, alpha: float, grad: np.ndarray) -> np.ndarray:
        """Return the momentum update used by APDAGD-type solvers."""
        raise NotImplementedError

    @abstractmethod
    def X_hat(self, lamu: np.ndarray) -> np.ndarray:
        """Recover a primal transport plan from dual variables."""
        raise NotImplementedError

    @abstractmethod
    def conv_crit(self, lamu: np.ndarray) -> float:
        """Return a convergence metric for the current dual variables."""
        raise NotImplementedError

    @abstractmethod
    def first_crit(self, X_hat: np.ndarray) -> float:
        """Return the first convergence criterion used in the experiments."""
        raise NotImplementedError

    @abstractmethod
    def second_crit(self, X_hat: np.ndarray, phi_eta: float) -> float:
        """Return the second convergence criterion used in the experiments."""
        raise NotImplementedError

    @abstractmethod
    def apdagd_ls_condition(self, phi_x: float, phi_approx_y: float) -> bool:
        """Return whether the APDAGD line-search condition is satisfied."""
        raise NotImplementedError

    @abstractmethod
    def pdaam_ls_condition(self, phi_x: float, phi_y: float, term: float) -> bool:
        """Return whether the PDAAM line-search condition is satisfied."""
        raise NotImplementedError

    @abstractmethod
    def pdaam_delta(self, phi_x: float, phi_y: float) -> float:
        """Return the PDAAM delta term used in the quadratic equation."""
        raise NotImplementedError
