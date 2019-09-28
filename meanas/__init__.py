"""
Electromagnetic simulation tools

This package is intended for building simulation inputs, analyzing
simulation outputs, and running short simulations on unspecialized hardware.
It is designed to provide tooling and a baseline for other, high-performance
purpose- and hardware-specific solvers.


**Contents**
- Finite difference frequency domain (FDFD)
    * Library of sparse matrices for representing the electromagnetic wave
    equation in 3D, as well as auxiliary matrices for conversion between fields
    * Waveguide mode operators
    * Waveguide mode eigensolver
    * Stretched-coordinate PML boundaries (SCPML)
    * Functional versions of most operators
    * Anisotropic media (limited to diagonal elements eps_xx, eps_yy, eps_zz, mu_xx, ...)
    * Arbitrary distributions of perfect electric and magnetic conductors (PEC / PMC)
- Finite difference time domain (FDTD)
    * Basic Maxwell time-steps
    * Poynting vector and energy calculation
    * Convolutional PMLs

This package does *not* provide a fast matrix solver, though by default
```meanas.fdfd.solvers.generic(...)``` will call
```scipy.sparse.linalg.qmr(...)``` to perform a solve.
For 2D FDFD problems this should be fine; likewise, the waveguide mode
solver uses scipy's eigenvalue solver, with reasonable results.

For solving large (or 3D) FDFD problems, I recommend a GPU-based iterative
solver, such as [opencl_fdfd](https://mpxd.net/code/jan/opencl_fdfd) or
those included in [MAGMA](http://icl.cs.utk.edu/magma/index.html)). Your
solver will need the ability to solve complex symmetric (non-Hermitian)
linear systems, ideally with double precision.


Dependencies:
- numpy
- scipy

"""

import pathlib

from .types import dx_lists_t, field_t, vfield_t, field_updater
from .vectorization import vec, unvec

__author__ = 'Jan Petykiewicz'

with open(pathlib.Path(__file__).parent / 'VERSION', 'r') as f:
    __version__ = f.read().strip()
