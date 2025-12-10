"""
Types shared across multiple submodules
"""
from typing import NewType
from collections.abc import Sequence, Callable, MutableSequence
from numpy.typing import NDArray
from numpy import floating, complexfloating


# Field types
fdfield_t = NewType('fdfield_t', NDArray[floating])
type fdfield = fdfield_t | NDArray[floating]
"""Vector field with shape (3, X, Y, Z) (e.g. `[E_x, E_y, E_z]`)"""

vfdfield_t = NewType('vfdfield_t', NDArray[floating])
type vfdfield = vfdfield_t | NDArray[floating]
"""Linearized vector field (single vector of length 3*X*Y*Z)"""

cfdfield_t = NewType('cfdfield_t', NDArray[complexfloating])
type cfdfield = cfdfield_t | NDArray[complexfloating]
"""Complex vector field with shape (3, X, Y, Z) (e.g. `[E_x, E_y, E_z]`)"""

vcfdfield_t = NewType('vcfdfield_t', NDArray[complexfloating])
type vcfdfield = vcfdfield_t | NDArray[complexfloating]
"""Linearized complex vector field (single vector of length 3*X*Y*Z)"""


fdslice_t = NewType('fdslice_t', NDArray[floating])
type fdslice = fdslice_t | NDArray[floating]
"""Vector field slice with shape (3, X, Y) (e.g. `[E_x, E_y, E_z]` at a single Z position)"""

vfdslice_t = NewType('vfdslice_t', NDArray[floating])
type vfdslice = vfdslice_t | NDArray[floating]
"""Linearized vector field slice (single vector of length 3*X*Y)"""

cfdslice_t = NewType('cfdslice_t', NDArray[complexfloating])
type cfdslice = cfdslice_t | NDArray[complexfloating]
"""Complex vector field slice with shape (3, X, Y) (e.g. `[E_x, E_y, E_z]` at a single Z position)"""

vcfdslice_t = NewType('vcfdslice_t', NDArray[complexfloating])
type vcfdslice = vcfdslice_t | NDArray[complexfloating]
"""Linearized complex vector field slice (single vector of length 3*X*Y)"""


fdfield2_t = NewType('fdfield2_t', NDArray[floating])
type fdfield2 = fdfield2_t | NDArray[floating]
"""2D Vector field with shape (2, X, Y) (e.g. `[E_x, E_y]`)"""

vfdfield2_t = NewType('vfdfield2_t', NDArray[floating])
type vfdfield2 = vfdfield2_t | NDArray[floating]
"""2D Linearized vector field (single vector of length 2*X*Y)"""

cfdfield2_t = NewType('cfdfield2_t', NDArray[complexfloating])
type cfdfield2 = cfdfield2_t | NDArray[complexfloating]
"""2D Complex vector field with shape (2, X, Y) (e.g. `[E_x, E_y]`)"""

vcfdfield2_t = NewType('vcfdfield2_t', NDArray[complexfloating])
type vcfdfield2 = vcfdfield2_t | NDArray[complexfloating]
"""2D Linearized complex vector field (single vector of length 2*X*Y)"""


dx_lists_t = Sequence[Sequence[NDArray[floating | complexfloating]]]
"""
 'dxes' datastructure which contains grid cell width information in the following format:

     [[[dx_e[0], dx_e[1], ...], [dy_e[0], ...], [dz_e[0], ...]],
      [[dx_h[0], dx_h[1], ...], [dy_h[0], ...], [dz_h[0], ...]]]

   where `dx_e[0]` is the x-width of the `x=0` cells, as used when calculating dE/dx,
   and `dy_h[0]` is the y-width of the `y=0` cells, as used when calculating dH/dy, etc.
"""

dx_lists2_t = Sequence[Sequence[NDArray[floating | complexfloating]]]
"""
 2D 'dxes' datastructure which contains grid cell width information in the following format:

     [[[dx_e[0], dx_e[1], ...], [dy_e[0], ...]],
      [[dx_h[0], dx_h[1], ...], [dy_h[0], ...]]]

   where `dx_e[0]` is the x-width of the `x=0` cells, as used when calculating dE/dx,
   and `dy_h[0]` is the y-width of the `y=0` cells, as used when calculating dH/dy, etc.
"""

dx_lists_mut = MutableSequence[MutableSequence[NDArray[floating | complexfloating]]]
"""Mutable version of `dx_lists_t`"""

dx_lists2_mut = MutableSequence[MutableSequence[NDArray[floating | complexfloating]]]
"""Mutable version of `dx_lists2_t`"""


fdfield_updater_t = Callable[..., fdfield_t]
"""Convenience type for functions which take and return an fdfield_t"""

cfdfield_updater_t = Callable[..., cfdfield_t]
"""Convenience type for functions which take and return an cfdfield_t"""
