"""
Functions for moving between a vector field (list of 3 ndarrays, `[f_x, f_y, f_z]`)
and a 1D array representation of that field `[f_x0, f_x1, f_x2,... f_y0,... f_z0,...]`.
Vectorized versions of the field use row-major (ie., C-style) ordering.
"""

from typing import Optional, overload, Union, Sequence
import numpy
from numpy.typing import ArrayLike

from .types import fdfield_t, vfdfield_t, cfdfield_t, vcfdfield_t


@overload
def vec(f: None) -> None:
    pass

@overload
def vec(f: fdfield_t) -> vfdfield_t:
    pass

@overload
def vec(f: cfdfield_t) -> vcfdfield_t:
    pass

@overload
def vec(f: ArrayLike) -> Union[vfdfield_t, vcfdfield_t]:
    pass

def vec(f: Union[fdfield_t, cfdfield_t, ArrayLike, None]) -> Union[vfdfield_t, vcfdfield_t, None]:
    """
    Create a 1D ndarray from a 3D vector field which spans a 1-3D region.

    Returns `None` if called with `f=None`.

    Args:
        f: A vector field, `[f_x, f_y, f_z]` where each `f_` component is a 1- to
           3-D ndarray (`f_*` should all be the same size). Doesn't fail with `f=None`.

    Returns:
        1D ndarray containing the linearized field (or `None`)
    """
    if f is None:
        return None
    return numpy.ravel(f, order='C')


@overload
def unvec(v: None, shape: Sequence[int]) -> None:
    pass

@overload
def unvec(v: vfdfield_t, shape: Sequence[int]) -> fdfield_t:
    pass

@overload
def unvec(v: vcfdfield_t, shape: Sequence[int]) -> cfdfield_t:
    pass

def unvec(v: Union[vfdfield_t, vcfdfield_t, None], shape: Sequence[int]) -> Union[fdfield_t, cfdfield_t, None]:
    """
    Perform the inverse of vec(): take a 1D ndarray and output a 3D field
     of form `[f_x, f_y, f_z]` where each of `f_*` is a len(shape)-dimensional
     ndarray.

    Returns `None` if called with `v=None`.

    Args:
        v: 1D ndarray representing a 3D vector field of shape shape (or None)
        shape: shape of the vector field

    Returns:
        `[f_x, f_y, f_z]` where each `f_` is a `len(shape)` dimensional ndarray (or `None`)
    """
    if v is None:
        return None
    return v.reshape((3, *shape), order='C')

