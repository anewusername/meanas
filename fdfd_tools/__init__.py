"""
Electromagnetic FDFD simulation tools

Tools for 3D and 2D Electromagnetic Finite Difference Frequency Domain (FDFD)
simulations. These tools handle conversion of fields to/from vector form,
creation of the wave operator matrix, stretched-coordinate PMLs,
field conversion operators, waveguide mode operator, and waveguide mode
solver.

This package only contains a solver for the waveguide mode eigenproblem;
if you want to solve 3D problems you can use your favorite iterative sparse
matrix solver (so long as it can handle complex symmetric [non-Hermitian]
matrices, ideally with double precision).


Dependencies:
- numpy
- scipy

"""

from .vectorization import vec, unvec, field_t, vfield_t
from .grid import dx_lists_t

__author__ = 'Jan Petykiewicz'