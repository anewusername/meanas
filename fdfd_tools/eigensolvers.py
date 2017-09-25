"""
Solvers for eigenvalue / eigenvector problems
"""
from typing import Tuple, List
import numpy
from numpy.linalg import norm
from scipy import sparse
import scipy.sparse.linalg as spalg


def power_iteration(operator: sparse.spmatrix,
                    guess_vector: numpy.ndarray = None,
                    iterations: int = 20,
                    ) -> Tuple[complex, numpy.ndarray]:
    """
    Use power iteration to estimate the dominant eigenvector of a matrix.

    :param operator: Matrix to analyze.
    :param guess_vector: Starting point for the eigenvector. Default is a randomly chosen vector.
    :param iterations: Number of iterations to perform. Default 20.
    :return: (Largest-magnitude eigenvalue, Corresponding eigenvector estimate)
    """
    if numpy.any(numpy.equal(guess_vector, None)):
        v = numpy.random.rand(operator.shape[0])
    else:
        v = guess_vector

    for _ in range(iterations):
        v = operator @ v
        v /= norm(v)

    lm_eigval = v.conj() @ operator @ v
    return lm_eigval, v


def rayleigh_quotient_iteration(operator: sparse.spmatrix,
                                guess_vector: numpy.ndarray,
                                iterations: int = 40,
                                tolerance: float = 1e-13,
                                ) -> Tuple[complex, numpy.ndarray]:
    """
    Use Rayleigh quotient iteration to refine an eigenvector guess.

    :param operator: Matrix to analyze.
    :param guess_vector: Eigenvector to refine.
    :param iterations: Maximum number of iterations to perform. Default 40.
    :param tolerance: Stop iteration if (A - I*eigenvalue) @ v < tolerance.
        Default 1e-13.
    :return: (eigenvalue, eigenvector)
    """
    v = guess_vector
    for _ in range(iterations):
        eigval = v.conj() @ operator @ v
        if norm(operator @ v - eigval * v) < tolerance:
            break
        v = spalg.spsolve(operator - eigval * sparse.eye(operator.shape[0]), v)
        v /= norm(v)

    return eigval, v


def signed_eigensolve(operator: sparse.spmatrix,
                      how_many: int,
                      negative: bool = False,
                      ) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Find the largest-magnitude positive-only (or negative-only) eigenvalues and
     eigenvectors of the provided matrix.

    :param operator: Matrix to analyze.
    :param how_many: How many eigenvalues to find.
    :param negative: Whether to find negative-only eigenvalues.
        Default False (positive only).
    :return: (sorted list of eigenvalues, 2D ndarray of corresponding eigenvectors)
        eigenvectors[:, k] corresponds to the k-th eigenvalue
    """
    # Use power iteration to estimate the dominant eigenvector
    lm_eigval, _ = power_iteration(operator)

    '''
    Shift by the absolute value of the largest eigenvalue, then find a few of the
     largest-magnitude (shifted) eigenvalues. A positive shift ensures that we find the
     largest _positive_ eigenvalues, since any negative eigenvalues will be shifted to the
     range 0 >= neg_eigval + abs(lm_eigval) > abs(lm_eigval)
    '''
    if negative:
        shifted_operator = operator - abs(lm_eigval) * sparse.eye(operator.shape[0])
    else:
        shifted_operator = operator + abs(lm_eigval) * sparse.eye(operator.shape[0])

    eigenvalues, eigenvectors = spalg.eigs(shifted_operator, which='LM', k=how_many, ncv=50)

    k = eigenvalues.argsort()
    return eigenvalues[k], eigenvectors[:, k]

