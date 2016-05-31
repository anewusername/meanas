# fdfd_tools

**fdfd_tools** is a python package containing utilities for
creating and analyzing 2D and 3D finite-difference frequency-domain (FDFD)
electromagnetic simulations.


**Contents**
* Library of sparse matrices for representing the electromagnetic wave
 equation in 3D, as well as auxiliary matrices for conversion between fields
* Waveguide mode solver and waveguide mode operators
* Stretched-coordinate PML boundaries (SCPML)
* Functional versions of most operators
* Anisotropic media (eps_xx, eps_yy, eps_zz, mu_xx, ...)

This package does *not* provide a matrix solver. The waveguide mode solver
uses scipy's eigenvalue solver; I recommend a GPU-based iterative solver (eg.
those included in [MAGMA](http://icl.cs.utk.edu/magma/index.html)). You will
need the ability to solve complex symmetric (non-Hermitian) linear systems,
ideally with double precision.

## Installation

**Requirements:**
* python 3 (written and tested with 3.5)
* numpy
* scipy


Install with pip, via git:
```bash
pip install git+https://mpxd.net/gogs/jan/fdfd_tools.git@release
```
