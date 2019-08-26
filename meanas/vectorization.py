"""
Functions for moving between a vector field (list of 3 ndarrays, [f_x, f_y, f_z])
and a 1D array representation of that field [f_x0, f_x1, f_x2,... f_y0,... f_z0,...].
Vectorized versions of the field use row-major (ie., C-style) ordering.
"""

from typing import List
import numpy

from .types import field_t, vfield_t


__author__ = 'Jan Petykiewicz'


def vec(f: field_t) -> vfield_t:
    """
    Create a 1D ndarray from a 3D vector field which spans a 1-3D region.

    Returns None if called with f=None.

    :param f: A vector field, [f_x, f_y, f_z] where each f_ component is a 1 to
        3D ndarray (f_* should all be the same size). Doesn't fail with f=None.
    :return: A 1D ndarray containing the linearized field (or None)
    """
    if numpy.any(numpy.equal(f, None)):
        return None
    return numpy.ravel(f, order='C')


def unvec(v: vfield_t, shape: numpy.ndarray) -> field_t:
    """
    Perform the inverse of vec(): take a 1D ndarray and output a 3D field
     of form [f_x, f_y, f_z] where each of f_* is a len(shape)-dimensional
     ndarray.

    Returns None if called with v=None.

    :param v: 1D ndarray representing a 3D vector field of shape shape (or None)
    :param shape: shape of the vector field
    :return: [f_x, f_y, f_z] where each f_ is a len(shape) dimensional ndarray
     (or None)
    """
    if numpy.any(numpy.equal(v, None)):
        return None
    return v.reshape((3, *shape), order='C')
