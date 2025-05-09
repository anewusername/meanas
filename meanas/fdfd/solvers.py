"""
Solvers and solver interface for FDFD problems.
"""

from typing import Any
from collections.abc import Callable
import logging

import numpy
from numpy.typing import ArrayLike, NDArray
from numpy.linalg import norm
import scipy.sparse.linalg

from ..fdmath import dx_lists_t, vfdfield_t, vcfdfield_t
from . import operators


logger = logging.getLogger(__name__)


def _scipy_qmr(
        A: scipy.sparse.csr_matrix,
        b: ArrayLike,
        **kwargs: Any,
        ) -> NDArray[numpy.float64]:
    """
    Wrapper for scipy.sparse.linalg.qmr

    Args:
        A: Sparse matrix
        b: Right-hand-side vector
        kwargs: Passed as **kwargs to the wrapped function

    Returns:
        Guess for solution (returned even if didn't converge)
    """

    #
    #Report on our progress
    #
    ii = 0

    def log_residual(xk: ArrayLike) -> None:
        nonlocal ii
        ii += 1
        if ii % 100 == 0:
            cur_norm = norm(A @ xk - b) / norm(b)
            logger.info(f'Solver residual at iteration {ii} : {cur_norm}')

    if 'callback' in kwargs:
        def augmented_callback(xk: ArrayLike) -> None:
            log_residual(xk)
            kwargs['callback'](xk)

        kwargs['callback'] = augmented_callback
    else:
        kwargs['callback'] = log_residual

    #
    # Run the actual solve
    #
    x, _ = scipy.sparse.linalg.qmr(A, b, **kwargs)
    return x


def generic(
        omega: complex,
        dxes: dx_lists_t,
        J: vcfdfield_t,
        epsilon: vfdfield_t,
        mu: vfdfield_t | None = None,
        *,
        pec: vfdfield_t | None = None,
        pmc: vfdfield_t | None = None,
        adjoint: bool = False,
        matrix_solver: Callable[..., ArrayLike] = _scipy_qmr,
        matrix_solver_opts: dict[str, Any] | None = None,
        E_guess: vcfdfield_t | None = None,
        ) -> vcfdfield_t:
    """
    Conjugate gradient FDFD solver using CSR sparse matrices.

    All ndarray arguments should be 1D arrays, as returned by `meanas.fdmath.vectorization.vec()`.

    Args:
        omega: Complex frequency to solve at.
        dxes: `[[dx_e, dy_e, dz_e], [dx_h, dy_h, dz_h]]` (complex cell sizes) as
            discussed in `meanas.fdmath.types`
        J: Electric current distribution (at E-field locations)
        epsilon: Dielectric constant distribution (at E-field locations)
        mu: Magnetic permeability distribution (at H-field locations)
        pec: Perfect electric conductor distribution
             (at E-field locations; non-zero value indicates PEC is present)
        pmc: Perfect magnetic conductor distribution
             (at H-field locations; non-zero value indicates PMC is present)
        adjoint: If true, solves the adjoint problem.
        matrix_solver: Called as `matrix_solver(A, b, **matrix_solver_opts) -> x`,
                where `A`: `scipy.sparse.csr_matrix`;
                      `b`: `ArrayLike`;
                      `x`: `ArrayLike`;
                Default is a wrapped version of `scipy.sparse.linalg.qmr()`
                 which doesn't return convergence info and logs the residual
                 every 100 iterations.
        matrix_solver_opts: Passed as kwargs to `matrix_solver(...)`
        E_guess: Guess at the solution E-field. `matrix_solver` must accept an
                `x0` argument with the same purpose.

    Returns:
        E-field which solves the system.
    """

    if matrix_solver_opts is None:
        matrix_solver_opts = dict()

    b0 = -1j * omega * J
    A0 = operators.e_full(omega, dxes, epsilon=epsilon, mu=mu, pec=pec, pmc=pmc)

    Pl, Pr = operators.e_full_preconditioners(dxes)

    if adjoint:
        A = (Pl @ A0 @ Pr).H
        b = Pr.H @ b0
    else:
        A = Pl @ A0 @ Pr
        b = Pl @ b0

    if E_guess is not None:
        if adjoint:
            x0 = Pr.H @ E_guess
        else:
            x0 = Pl @ E_guess
        matrix_solver_opts['x0'] = x0

    x = matrix_solver(A.tocsr(), b, **matrix_solver_opts)

    if adjoint:
        x0 = Pl.H @ x
    else:
        x0 = Pr @ x

    return x0
