"""
Utilities for running finite-difference time-domain (FDTD) simulations


Timestep
========

From the discussion of "Plane waves and the Dispersion relation" in `meanas.fdmath`,
we have

$$ c^2 \\Delta_t^2 = \\frac{\\Delta_t^2}{\\mu \\epsilon} < 1/(\\frac{1}{\\Delta_x^2} + \\frac{1}{\\Delta_y^2} + \\frac{1}{\\Delta_z^2}) $$

or, if \\( \\Delta_x = \\Delta_y = \\Delta_z \\), then \\( c \\Delta_t < \\frac{\\Delta_x}{\\sqrt{3}} \\).

Based on this, we can set

    dt = sqrt(mu.min() * epsilon.min()) / sqrt(1/dx_min**2 + 1/dy_min**2 + 1/dz_min**2)

The `dx_min`, `dy_min`, `dz_min` should be the minimum value across both the base and derived grids.


Poynting Vector
===============
# TODO

Energy conservation
===================
# TODO

Boundary conditions
===================
# TODO notes about boundaries / PMLs
"""

from .base import maxwell_e, maxwell_h
from .pml import cpml
from .energy import (poynting, poynting_divergence, energy_hstep, energy_estep,
                     delta_energy_h2e, delta_energy_h2e, delta_energy_j)
from .boundaries import conducting_boundary
