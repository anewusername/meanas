"""
Solvers for FDFD problems.
"""

from typing import List, Callable, Dict, Any
import logging

import numpy
from numpy.linalg import norm
import scipy.sparse.linalg

from . import operators


logger = logging.getLogger(__name__)


def _scipy_qmr(A: scipy.sparse.csr_matrix,
               b: numpy.ndarray,
               **kwargs
               ) -> numpy.ndarray:
    """
    Wrapper for scipy.sparse.linalg.qmr

    :param A: Sparse matrix
    :param b: Right-hand-side vector
    :param kwargs: Passed as **kwargs to the wrapped function
    :return: Guess for solution (returned even if didn't converge)
    """

    '''
    Report on our progress
    '''
    iter = 0

    def log_residual(xk):
        nonlocal iter
        iter += 1
        if iter % 100 == 0:
            logger.info('Solver residual at iteration {} : {}'.format(iter, norm(A @ xk - b)))

    if 'callback' in kwargs:
        def augmented_callback(xk):
            log_residual(xk)
            kwargs['callback'](xk)

        kwargs['callback'] = augmented_callback
    else:
        kwargs['callback'] = log_residual

    '''
    Run the actual solve
    '''

    x, _ = scipy.sparse.linalg.qmr(A, b, **kwargs)
    return x


def generic(omega: complex,
            dxes: List[List[numpy.ndarray]],
            J: numpy.ndarray,
            epsilon: numpy.ndarray,
            mu: numpy.ndarray = None,
            pec: numpy.ndarray = None,
            pmc: numpy.ndarray = None,
            adjoint: bool = False,
            matrix_solver: Callable[..., numpy.ndarray] = _scipy_qmr,
            matrix_solver_opts: Dict[str, Any] = None,
            ) -> numpy.ndarray:
    """
    Conjugate gradient FDFD solver using CSR sparse matrices.

    All ndarray arguments should be 1D array, as returned by meanas.vec().

    :param omega: Complex frequency to solve at.
    :param dxes: [[dx_e, dy_e, dz_e], [dx_h, dy_h, dz_h]] (complex cell sizes)
    :param J: Electric current distribution (at E-field locations)
    :param epsilon: Dielectric constant distribution (at E-field locations)
    :param mu: Magnetic permeability distribution (at H-field locations)
    :param pec: Perfect electric conductor distribution
        (at E-field locations; non-zero value indicates PEC is present)
    :param pmc: Perfect magnetic conductor distribution
        (at H-field locations; non-zero value indicates PMC is present)
    :param adjoint: If true, solves the adjoint problem.
    :param matrix_solver: Called as matrix_solver(A, b, **matrix_solver_opts) -> x
        Where A: scipy.sparse.csr_matrix
              b: numpy.ndarray
              x: numpy.ndarray
        Default is a wrapped version of scipy.sparse.linalg.qmr()
         which doesn't return convergence info and logs the residual
         every 100 iterations.
    :param matrix_solver_opts: Passed as kwargs to matrix_solver(...)
    :return: E-field which solves the system.
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

    x = matrix_solver(A.tocsr(), b, **matrix_solver_opts)

    if adjoint:
        x0 = Pl.H @ x
    else:
        x0 = Pr @ x

    return x0