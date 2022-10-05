"""
Types shared across multiple submodules
"""
from typing import Sequence, Callable, MutableSequence
import numpy
from numpy.typing import NDArray


# Field types
fdfield_t = NDArray[numpy.float_]
"""Vector field with shape (3, X, Y, Z) (e.g. `[E_x, E_y, E_z]`)"""

vfdfield_t = NDArray[numpy.float_]
"""Linearized vector field (single vector of length 3*X*Y*Z)"""

cfdfield_t = NDArray[numpy.complex_]
"""Complex vector field with shape (3, X, Y, Z) (e.g. `[E_x, E_y, E_z]`)"""

vcfdfield_t = NDArray[numpy.complex_]
"""Linearized complex vector field (single vector of length 3*X*Y*Z)"""


dx_lists_t = Sequence[Sequence[NDArray[numpy.float_]]]
"""
 'dxes' datastructure which contains grid cell width information in the following format:

     [[[dx_e[0], dx_e[1], ...], [dy_e[0], ...], [dz_e[0], ...]],
      [[dx_h[0], dx_h[1], ...], [dy_h[0], ...], [dz_h[0], ...]]]

   where `dx_e[0]` is the x-width of the `x=0` cells, as used when calculating dE/dx,
   and `dy_h[0]` is the y-width of the `y=0` cells, as used when calculating dH/dy, etc.
"""

dx_lists_mut = MutableSequence[MutableSequence[NDArray[numpy.float_]]]
"""Mutable version of `dx_lists_t`"""


fdfield_updater_t = Callable[..., fdfield_t]
"""Convenience type for functions which take and return an fdfield_t"""

cfdfield_updater_t = Callable[..., cfdfield_t]
"""Convenience type for functions which take and return an cfdfield_t"""
