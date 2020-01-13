"""
Tools for finite difference frequency-domain (FDFD) simulations and calculations.

These mostly involve picking a single frequency, then setting up and solving a
matrix equation (Ax=b) or eigenvalue problem.


Submodules:

- `operators`, `functional`: General FDFD problem setup.
- `solvers`: Solver interface and reference implementation.
- `scpml`: Stretched-coordinate perfectly matched layer (scpml) boundary conditions
- `waveguide_2d`: Operators and mode-solver for waveguides with constant cross-section.
- `waveguide_3d`: Functions for transforming `waveguide_2d` results into 3D.


===========

# TODO FDFD?
# TODO PML


"""
from . import solvers, operators, functional, scpml, waveguide_2d, waveguide_3d
# from . import farfield, bloch TODO
