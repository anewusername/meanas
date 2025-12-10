"""
Functions for moving between a vector field (list of 3 ndarrays, `[f_x, f_y, f_z]`)
and a 1D array representation of that field `[f_x0, f_x1, f_x2,... f_y0,... f_z0,...]`.
Vectorized versions of the field use row-major (ie., C-style) ordering.
"""

from typing import overload
from collections.abc import Sequence
import numpy
from numpy.typing import ArrayLike, NDArray

from .types import (
    fdfield_t, vfdfield_t, cfdfield_t, vcfdfield_t,
    fdslice_t, vfdslice_t, cfdslice_t, vcfdslice_t,
    fdfield2_t, vfdfield2_t, cfdfield2_t, vcfdfield2_t,
    )


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
def vec(f: fdfield2_t) -> vfdfield2_t:
    pass

@overload
def vec(f: cfdfield2_t) -> vcfdfield2_t:
    pass

@overload
def vec(f: fdslice_t) -> vfdslice_t:
    pass

@overload
def vec(f: cfdslice_t) -> vcfdslice_t:
    pass

@overload
def vec(f: ArrayLike) -> NDArray:
    pass

def vec(
        f: fdfield_t | cfdfield_t | fdfield2_t | cfdfield2_t | fdslice_t | cfdslice_t | ArrayLike | None,
        ) -> vfdfield_t | vcfdfield_t | vfdfield2_t | vcfdfield2_t | vfdslice_t | vcfdslice_t | NDArray | None:
    """
    Create a 1D ndarray from a vector field which spans a 1-3D region.

    Returns `None` if called with `f=None`.

    Args:
        f: A vector field, e.g. `[f_x, f_y, f_z]` where each `f_` component is a 1- to
           3-D ndarray (`f_*` should all be the same size). Doesn't fail with `f=None`.

    Returns:
        1D ndarray containing the linearized field (or `None`)
    """
    if f is None:
        return None
    return numpy.ravel(f, order='C')        # type: ignore


@overload
def unvec(v: None, shape: Sequence[int], nvdim: int = 3) -> None:
    pass

@overload
def unvec(v: vfdfield_t, shape: Sequence[int], nvdim: int = 3) -> fdfield_t:
    pass

@overload
def unvec(v: vcfdfield_t, shape: Sequence[int], nvdim: int = 3) -> cfdfield_t:
    pass

@overload
def unvec(v: vfdfield2_t, shape: Sequence[int], nvdim: int = 3) -> fdfield2_t:
    pass

@overload
def unvec(v: vcfdfield2_t, shape: Sequence[int], nvdim: int = 3) -> cfdfield2_t:
    pass

@overload
def unvec(v: vfdslice_t, shape: Sequence[int], nvdim: int = 3) -> fdslice_t:
    pass

@overload
def unvec(v: vcfdslice_t, shape: Sequence[int], nvdim: int = 3) -> cfdslice_t:
    pass

@overload
def unvec(v: ArrayLike, shape: Sequence[int], nvdim: int = 3) -> NDArray:
    pass

def unvec(
        v: vfdfield_t | vcfdfield_t | vfdfield2_t | vcfdfield2_t | vfdslice_t | vcfdslice_t | ArrayLike | None,
        shape: Sequence[int],
        nvdim: int = 3,
        ) -> fdfield_t | cfdfield_t | fdfield2_t | cfdfield2_t | fdslice_t | cfdslice_t | NDArray | None:
    """
    Perform the inverse of vec(): take a 1D ndarray and output an `nvdim`-component field
     of form e.g. `[f_x, f_y, f_z]` (`nvdim=3`) where each of `f_*` is a len(shape)-dimensional
     ndarray.

    Returns `None` if called with `v=None`.

    Args:
        v: 1D ndarray representing a vector field of shape shape (or None)
        shape: shape of the vector field
        nvdim: Number of components in each vector

    Returns:
        `[f_x, f_y, f_z]` where each `f_` is a `len(shape)` dimensional ndarray (or `None`)
    """
    if v is None:
        return None
    return v.reshape((nvdim, *shape), order='C')  # type: ignore

