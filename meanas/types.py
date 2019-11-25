"""
Types shared across multiple submodules
"""
import numpy
from typing import List, Callable


# Field types
field_t = numpy.ndarray
"""vector field with shape (3, X, Y, Z) (e.g. `[E_x, E_y, E_z]`)"""

vfield_t = numpy.ndarray
"""Linearized vector field (vector of length 3*X*Y*Z)"""

dx_lists_t = List[List[numpy.ndarray]]
'''
 'dxes' datastructure which contains grid cell width information in the following format:
 `[[[dx_e_0, dx_e_1, ...], [dy_e_0, ...], [dz_e_0, ...]],
   [[dx_h_0, dx_h_1, ...], [dy_h_0, ...], [dz_h_0, ...]]]`
 where `dx_e_0` is the x-width of the `x=0` cells, as used when calculating dE/dx,
 and `dy_h_0` is  the y-width of the `y=0` cells, as used when calculating dH/dy, etc.
'''


field_updater = Callable[[field_t], field_t]
