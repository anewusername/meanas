# meanas

**meanas** is a python package for electromagnetic simulations

** UNSTABLE / WORK IN PROGRESS **

Formerly known as [fdfd_tools](https://mpxd.net/code/jan/fdfd_tools).

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
`meanas.fdfd.solvers.generic(...)` will call
`scipy.sparse.linalg.qmr(...)` to perform a solve.
For 2D FDFD problems this should be fine; likewise, the waveguide mode
solver uses scipy's eigenvalue solver, with reasonable results.

For solving large (or 3D) FDFD problems, I recommend a GPU-based iterative
solver, such as [opencl_fdfd](https://mpxd.net/code/jan/opencl_fdfd) or
those included in [MAGMA](http://icl.cs.utk.edu/magma/index.html). Your
solver will need the ability to solve complex symmetric (non-Hermitian)
linear systems, ideally with double precision.

- [Source repository](https://mpxd.net/code/jan/meanas)
- [PyPI](https://pypi.org/project/meanas)
- [Github mirror](https://github.com/anewusername/meanas)


## Installation

**Requirements:**

* python >=3.11
* numpy
* scipy


Install from PyPI with pip:
```bash
pip3 install 'meanas[dev]'
```

### Development install
Install python3 and git:
```bash
# This is for Debian/Ubuntu/other-apt-based systems; you may need an alternative command
sudo apt install python3 build-essential python3-dev git
```

In-place development install:
```bash
# Download using git
git clone https://mpxd.net/code/jan/meanas.git

# If you'd like to create a virtualenv, do so:
python3 -m venv my_venv

# If you are using a virtualenv, activate it
source my_venv/bin/activate

# Install in-place (-e, editable) from ./meanas, including development dependencies ([dev])
pip3 install --user -e './meanas[dev]'

# Run tests
cd meanas
python3 -m pytest -rsxX | tee test_results.txt
```

#### See also:
- [git book](https://git-scm.com/book/en/v2)
- [venv documentation](https://docs.python.org/3/tutorial/venv.html)
- [python language reference](https://docs.python.org/3/reference/index.html)
- [python standard library](https://docs.python.org/3/library/index.html)


## Use

See `examples/` for some simple examples; you may need additional
packages such as [gridlock](https://mpxd.net/code/jan/gridlock)
to run the examples.
