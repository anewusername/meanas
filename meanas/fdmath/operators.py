"""
Matrix operators for finite difference simulations

Basic discrete calculus etc.
"""
from typing import Sequence, List
import numpy
from numpy.typing import NDArray
import scipy.sparse as sparse   # type: ignore

from .types import vfdfield_t


def shift_circ(
        axis: int,
        shape: Sequence[int],
        shift_distance: int = 1,
        ) -> sparse.spmatrix:
    """
    Utility operator for performing a circular shift along a specified axis by a
     specified number of elements.

    Args:
        axis: Axis to shift along. x=0, y=1, z=2
        shape: Shape of the grid being shifted
        shift_distance: Number of cells to shift by. May be negative. Default 1.

    Returns:
        Sparse matrix for performing the circular shift.
    """
    if len(shape) not in (2, 3):
        raise Exception('Invalid shape: {}'.format(shape))
    if axis not in range(len(shape)):
        raise Exception('Invalid direction: {}, shape is {}'.format(axis, shape))

    shifts = [abs(shift_distance) if a == axis else 0 for a in range(3)]
    shifted_diags = [(numpy.arange(n) + s) % n for n, s in zip(shape, shifts)]
    ijk = numpy.meshgrid(*shifted_diags, indexing='ij')

    n = numpy.prod(shape)
    i_ind = numpy.arange(n)
    j_ind = numpy.ravel_multi_index(ijk, shape, order='C')

    vij = (numpy.ones(n), (i_ind, j_ind.ravel(order='C')))

    d = sparse.csr_matrix(vij, shape=(n, n))

    if shift_distance < 0:
        d = d.T

    return d


def shift_with_mirror(
        axis: int,
        shape: Sequence[int],
        shift_distance: int = 1,
        ) -> sparse.spmatrix:
    """
    Utility operator for performing an n-element shift along a specified axis, with mirror
    boundary conditions applied to the cells beyond the receding edge.

    Args:
        axis: Axis to shift along. x=0, y=1, z=2
        shape: Shape of the grid being shifted
        shift_distance: Number of cells to shift by. May be negative. Default 1.

    Returns:
        Sparse matrix for performing the shift-with-mirror.
    """
    if len(shape) not in (2, 3):
        raise Exception('Invalid shape: {}'.format(shape))
    if axis not in range(len(shape)):
        raise Exception('Invalid direction: {}, shape is {}'.format(axis, shape))
    if shift_distance >= shape[axis]:
        raise Exception('Shift ({}) is too large for axis {} of size {}'.format(
                        shift_distance, axis, shape[axis]))

    def mirrored_range(n: int, s: int) -> NDArray[numpy.int_]:
        v = numpy.arange(n) + s
        v = numpy.where(v >= n, 2 * n - v - 1, v)
        v = numpy.where(v < 0, - 1 - v, v)
        return v

    shifts = [shift_distance if a == axis else 0 for a in range(3)]
    shifted_diags = [mirrored_range(n, s) for n, s in zip(shape, shifts)]
    ijk = numpy.meshgrid(*shifted_diags, indexing='ij')

    n = numpy.prod(shape)
    i_ind = numpy.arange(n)
    j_ind = numpy.ravel_multi_index(ijk, shape, order='C')

    vij = (numpy.ones(n), (i_ind, j_ind.ravel(order='C')))

    d = sparse.csr_matrix(vij, shape=(n, n))
    return d


def deriv_forward(
        dx_e: Sequence[NDArray[numpy.float_]],
        ) -> List[sparse.spmatrix]:
    """
    Utility operators for taking discretized derivatives (forward variant).

    Args:
        dx_e: Lists of cell sizes for all axes
              `[[dx_0, dx_1, ...], [dy_0, dy_1, ...], ...]`.

    Returns:
        List of operators for taking forward derivatives along each axis.
    """
    shape = [s.size for s in dx_e]
    n = numpy.prod(shape)

    dx_e_expanded = numpy.meshgrid(*dx_e, indexing='ij')

    def deriv(axis: int) -> sparse.spmatrix:
        return shift_circ(axis, shape, 1) - sparse.eye(n)

    Ds = [sparse.diags(+1 / dx.ravel(order='C')) @ deriv(a)
          for a, dx in enumerate(dx_e_expanded)]

    return Ds


def deriv_back(
        dx_h: Sequence[NDArray[numpy.float_]],
        ) -> List[sparse.spmatrix]:
    """
    Utility operators for taking discretized derivatives (backward variant).

    Args:
        dx_h: Lists of cell sizes for all axes
              `[[dx_0, dx_1, ...], [dy_0, dy_1, ...], ...]`.

    Returns:
        List of operators for taking forward derivatives along each axis.
    """
    shape = [s.size for s in dx_h]
    n = numpy.prod(shape)

    dx_h_expanded = numpy.meshgrid(*dx_h, indexing='ij')

    def deriv(axis: int) -> sparse.spmatrix:
        return shift_circ(axis, shape, -1) - sparse.eye(n)

    Ds = [sparse.diags(-1 / dx.ravel(order='C')) @ deriv(a)
          for a, dx in enumerate(dx_h_expanded)]

    return Ds


def cross(
        B: Sequence[sparse.spmatrix],
        ) -> sparse.spmatrix:
    """
    Cross product operator

    Args:
        B: List `[Bx, By, Bz]` of sparse matrices corresponding to the x, y, z
           portions of the operator on the left side of the cross product.

    Returns:
        Sparse matrix corresponding to (B x), where x is the cross product.
    """
    n = B[0].shape[0]
    zero = sparse.csr_matrix((n, n))
    return sparse.bmat([[zero, -B[2], B[1]],
                        [B[2], zero, -B[0]],
                        [-B[1], B[0], zero]])


def vec_cross(b: vfdfield_t) -> sparse.spmatrix:
    """
    Vector cross product operator

    Args:
        b: Vector on the left side of the cross product.

    Returns:

        Sparse matrix corresponding to (b x), where x is the cross product.

    """
    B = [sparse.diags(c) for c in numpy.split(b, 3)]
    return cross(B)


def avg_forward(axis: int, shape: Sequence[int]) -> sparse.spmatrix:
    """
    Forward average operator `(x4 = (x4 + x5) / 2)`

    Args:
        axis: Axis to average along (x=0, y=1, z=2)
        shape: Shape of the grid to average

    Returns:
        Sparse matrix for forward average operation.
    """
    if len(shape) not in (2, 3):
        raise Exception('Invalid shape: {}'.format(shape))

    n = numpy.prod(shape)
    return 0.5 * (sparse.eye(n) + shift_circ(axis, shape))


def avg_back(axis: int, shape: Sequence[int]) -> sparse.spmatrix:
    """
    Backward average operator `(x4 = (x4 + x3) / 2)`

    Args:
        axis: Axis to average along (x=0, y=1, z=2)
        shape: Shape of the grid to average

    Returns:
        Sparse matrix for backward average operation.
    """
    return avg_forward(axis, shape).T


def curl_forward(
        dx_e: Sequence[NDArray[numpy.float_]],
        ) -> sparse.spmatrix:
    """
    Curl operator for use with the E field.

    Args:
        dx_e: Lists of cell sizes for all axes
              `[[dx_0, dx_1, ...], [dy_0, dy_1, ...], ...]`.

    Returns:
        Sparse matrix for taking the discretized curl of the E-field
    """
    return cross(deriv_forward(dx_e))


def curl_back(
        dx_h: Sequence[NDArray[numpy.float_]],
        ) -> sparse.spmatrix:
    """
    Curl operator for use with the H field.

    Args:
        dx_h: Lists of cell sizes for all axes
              `[[dx_0, dx_1, ...], [dy_0, dy_1, ...], ...]`.

    Returns:
        Sparse matrix for taking the discretized curl of the H-field
    """
    return cross(deriv_back(dx_h))
