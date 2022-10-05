"""
Solvers for eigenvalue / eigenvector problems
"""
from typing import Tuple, Callable, Optional, Union
import numpy
from numpy.typing import NDArray, ArrayLike
from numpy.linalg import norm
from scipy import sparse              # type: ignore
import scipy.sparse.linalg as spalg   # type: ignore


def power_iteration(
        operator: sparse.spmatrix,
        guess_vector: Optional[NDArray[numpy.complex128]] = None,
        iterations: int = 20,
        ) -> Tuple[complex, NDArray[numpy.complex128]]:
    """
    Use power iteration to estimate the dominant eigenvector of a matrix.

    Args:
       operator: Matrix to analyze.
       guess_vector: Starting point for the eigenvector. Default is a randomly chosen vector.
       iterations: Number of iterations to perform. Default 20.

    Returns:
        (Largest-magnitude eigenvalue, Corresponding eigenvector estimate)
    """
    if guess_vector is None:
        v = numpy.random.rand(operator.shape[0]) + 1j * numpy.random.rand(operator.shape[0])
    else:
        v = guess_vector

    for _ in range(iterations):
        v = operator @ v
        v /= numpy.abs(v).sum()     # faster than true norm
    v /= norm(v)

    lm_eigval = v.conj() @ (operator @ v)
    return lm_eigval, v


def rayleigh_quotient_iteration(
        operator: Union[sparse.spmatrix, spalg.LinearOperator],
        guess_vector: NDArray[numpy.complex128],
        iterations: int = 40,
        tolerance: float = 1e-13,
        solver: Optional[Callable[..., NDArray[numpy.complex128]]] = None,
        ) -> Tuple[complex, NDArray[numpy.complex128]]:
    """
    Use Rayleigh quotient iteration to refine an eigenvector guess.

    Args:
        operator: Matrix to analyze.
        guess_vector: Eigenvector to refine.
        iterations: Maximum number of iterations to perform. Default 40.
        tolerance: Stop iteration if `(A - I*eigenvalue) @ v < num_vectors * tolerance`,
                    Default 1e-13.
        solver: Solver function of the form `x = solver(A, b)`.
                By default, use scipy.sparse.spsolve for sparse matrices and
                scipy.sparse.bicgstab for general LinearOperator instances.

    Returns:
        (eigenvalues, eigenvectors)
    """
    try:
        (operator - sparse.eye(operator.shape[0]))

        def shift(eigval: float) -> sparse:
            return eigval * sparse.eye(operator.shape[0])

        if solver is None:
            solver = spalg.spsolve
    except TypeError:
        def shift(eigval: float) -> spalg.LinearOperator:
            return spalg.LinearOperator(
                    shape=operator.shape,
                    dtype=operator.dtype,
                    matvec=lambda v: eigval * v,
                    )
        if solver is None:
            def solver(A: spalg.LinearOperator, b: ArrayLike) -> NDArray[numpy.complex128]:
                return spalg.bicgstab(A, b)[0]
    assert(solver is not None)

    v = numpy.squeeze(guess_vector)
    v /= norm(v)
    for _ in range(iterations):
        eigval = v.conj() @ (operator @ v)
        if norm(operator @ v - eigval * v) < tolerance:
            break

        shifted_operator = operator - shift(eigval)
        v = solver(shifted_operator, v)
        v /= norm(v)
    return eigval, v


def signed_eigensolve(
        operator: Union[sparse.spmatrix, spalg.LinearOperator],
        how_many: int,
        negative: bool = False,
        ) -> Tuple[NDArray[numpy.complex128], NDArray[numpy.complex128]]:
    """
    Find the largest-magnitude positive-only (or negative-only) eigenvalues and
     eigenvectors of the provided matrix.

    Args:
        operator: Matrix to analyze.
        how_many: How many eigenvalues to find.
        negative: Whether to find negative-only eigenvalues.
                  Default False (positive only).

    Returns:
        (sorted list of eigenvalues, 2D ndarray of corresponding eigenvectors)
        `eigenvectors[:, k]` corresponds to the k-th eigenvalue
    """
    # Use power iteration to estimate the dominant eigenvector
    lm_eigval, _ = power_iteration(operator)

    '''
    Shift by the absolute value of the largest eigenvalue, then find a few of the
     largest-magnitude (shifted) eigenvalues. A positive shift ensures that we find the
     largest _positive_ eigenvalues, since any negative eigenvalues will be shifted to the
     range `0 >= neg_eigval + abs(lm_eigval) > abs(lm_eigval)`
    '''
    shift = numpy.abs(lm_eigval)
    if negative:
        shift *= -1

    # Try to combine, use general LinearOperator if we fail
    try:
        shifted_operator = operator + shift * sparse.eye(operator.shape[0])
    except TypeError:
        shifted_operator = operator + spalg.LinearOperator(shape=operator.shape,
                                                           matvec=lambda v: shift * v)

    shifted_eigenvalues, eigenvectors = spalg.eigs(shifted_operator, which='LM', k=how_many, ncv=50)
    eigenvalues = shifted_eigenvalues - shift

    k = eigenvalues.argsort()
    return eigenvalues[k], eigenvectors[:, k]

